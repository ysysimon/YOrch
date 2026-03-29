#pragma once

#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "executor.hpp"
#include "result.hpp"
#include "task_tree.hpp"

namespace yorch::detail {

/**
 * @brief Compile-time trait that extracts a task's raw return type.
 *
 * This is a low-level `plan` trait whose job is to answer "what raw type does
 * this task expose through `invoke_raw(...)`?" before any later payload
 * normalization is applied.
 *
 * The primary template intentionally provides no `type`. Concrete behavior is
 * supplied through partial specializations for:
 *
 * - tasks that declare `raw_result_type` directly
 * - exception wrappers such as `catch_failure_task<Task>`
 * - policy-based wrappers such as
 *   `catch_failure_with_policy_task<Task, Policy>`
 *
 * @tparam Task Task type being analyzed.
 * @tparam SFINAE hook used for detection-based specialization.
 */
template <typename Task, typename = void>
struct task_raw_result;

/**
 * @brief Uses a task's own `raw_result_type` declaration as the raw return.
 *
 * This is the base case for `task_raw_result`. Tasks such as `bound_task`
 * already expose a canonical `raw_result_type`, so no wrapper unwrapping is
 * required and that declared type becomes the answer directly.
 *
 * `std::remove_cvref_t` strips const/reference qualifiers so detection always
 * targets the normalized task object type.
 *
 * @tparam Task Task type being analyzed.
 */
template <typename Task>
struct task_raw_result<Task, std::void_t<typename std::remove_cvref_t<Task>::raw_result_type>> {
    using type = typename std::remove_cvref_t<Task>::raw_result_type;
};

/**
 * @brief Deduces the raw return exposed by `catch_failure_task<Task>`.
 *
 * The wrapper's public raw return depends on the wrapped task:
 *
 * - if the inner task returns `void`, the wrapper turns both success and
 *   failure paths into `step_result`, so the resulting raw type is
 *   `step_result`
 * - if the inner task already returns a non-void raw type, the wrapper only
 *   catches exceptions and preserves that return category
 *
 * This specialization therefore recursively queries `task_raw_result<Task>` to
 * recover the wrapped task's `inner_type`, then applies the wrapper-specific
 * rewrite rule on top.
 *
 * @tparam Task Task type wrapped by `catch_failure_task`.
 */
template <typename Task>
struct task_raw_result<
    catch_failure_task<Task>,
    std::void_t<typename task_raw_result<std::remove_cvref_t<Task>>::type>
> {
    using inner_type = typename task_raw_result<std::remove_cvref_t<Task>>::type;
    using type = std::conditional_t<std::is_void_v<inner_type>, step_result, inner_type>;
};

/**
 * @brief Deduces the raw return exposed by
 * `catch_failure_with_policy_task<Task, Policy>`.
 *
 * This wrapper depends on two compile-time facts:
 *
 * - the inner task's raw return type
 * - the fallback policy's result type
 *
 * Its rule is:
 *
 * - if the inner task returns `void`, the wrapper's raw return is determined by
 *   the policy, because both success and exception paths must use the policy's
 *   result shape
 * - if the inner task already returns a non-void raw type, the wrapper keeps
 *   that raw type and only requires the policy to produce a compatible
 *   exception-path value
 *
 * The `std::void_t<...>` here performs both checks up front:
 *
 * - `task_raw_result<Task>::type` must be recursively available
 * - `policy_result_t<Policy>` must exist
 *
 * Only when both are valid does this specialization participate in matching.
 *
 * @tparam Task Task type wrapped by the policy-aware exception adapter.
 * @tparam Policy Fallback exception policy type.
 */
template <typename Task, typename Policy>
struct task_raw_result<
    catch_failure_with_policy_task<Task, Policy>,
    std::void_t<
        typename task_raw_result<std::remove_cvref_t<Task>>::type,
        detail::policy_result_t<std::remove_cvref_t<Policy>>
    >
> {
    using inner_type = typename task_raw_result<std::remove_cvref_t<Task>>::type;
    using policy_type = detail::policy_result_t<std::remove_cvref_t<Policy>>;
    using type = std::conditional_t<std::is_void_v<inner_type>, policy_type, inner_type>;
};

/**
 * @brief Convenience alias for a task's raw return type.
 *
 * This avoids repeatedly spelling `typename task_raw_result<T>::type`.
 *
 * @tparam Task Task type being analyzed.
 */
template <typename Task>
using task_raw_result_t = typename task_raw_result<Task>::type;

template <typename R>
struct task_output_type_impl {
    using type = R;
};

template <>
struct task_output_type_impl<step_result> {
    using type = void;
};

template <>
struct task_output_type_impl<task_result<void>> {
    using type = void;
};

template <typename T>
struct task_output_type_impl<task_result<T>> {
    using type = T;
};

template <typename R>
struct task_output_type {
    static_assert(!std::is_reference_v<R>,
                  "Task raw return references are not supported as parent payloads");

    using type = typename task_output_type_impl<std::remove_cv_t<R>>::type;
};

template <>
struct task_output_type<void> {
    using type = void;
};

template <typename R>
using task_output_t = typename task_output_type<R>::type;

template <typename... Nodes>
[[nodiscard]] consteval auto make_level_array() {
    return std::array<std::size_t, sizeof...(Nodes)> {Nodes::level...};
}

template <std::size_t N>
[[nodiscard]] consteval auto make_slot_index_array() {
    std::array<std::size_t, N> indices {};

    for (std::size_t i = 0; i < N; ++i) {
        indices[i] = i;
    }

    return indices;
}

/**
 * @brief Reconstructs each node's direct parent index from the recorded level sequence.
 *
 * For every non-root node, the direct parent is the nearest earlier node whose
 * level is exactly one less than the current node's level. The root keeps the
 * sentinel value `N`, which means "no parent".
 *
 * This rule relies on the current `task_tree` recording convention: nodes are
 * stored in insertion order as a valid tree-shaped level sequence, so scanning
 * backward to the nearest `level - 1` entry recovers the direct parent.
 *
 * @tparam N Number of nodes in the recorded level sequence.
 * @param levels Node levels in insertion order.
 * @return An array of parent indices aligned with `levels`.
 */
template <std::size_t N>
[[nodiscard]] consteval auto make_parent_index_array(const std::array<std::size_t, N>& levels) {
    std::array<std::size_t, N> parents {};
    parents.fill(N);

    if constexpr (N > 0) {
        parents[0] = N;

        for (std::size_t node = 1; node < N; ++node) {
            const auto target_level = levels[node] - 1;

            for (std::size_t cursor = node; cursor > 0; --cursor) {
                const auto candidate = cursor - 1;

                if (levels[candidate] == target_level) {
                    parents[node] = candidate;
                    break;
                }
            }
        }
    }

    return parents;
}

/**
 * @brief Flat direct-child adjacency storage for a tree-shaped plan.
 *
 * For each node, this layout records:
 *
 * - `counts[node]`: how many direct children the node has
 * - `offsets[node]`: where that node's child range begins in `indices`
 * - `indices`: all child node indices concatenated into one flat buffer
 *
 * This keeps child traversal compact and allocation-free while matching the
 * current invariant that the compiled structure is a single-parent tree.
 *
 * @tparam N Number of nodes in the compiled plan.
 */
template <std::size_t N>
struct child_layout {
    std::array<std::size_t, N> counts {};
    std::array<std::size_t, N> offsets {};
    std::array<std::size_t, (N > 0 ? N - 1 : 0)> indices {};
};

/**
 * @brief Builds a flat direct-child adjacency layout from per-node parent indices.
 *
 * The returned layout stores, for each node:
 *
 * - how many direct children it has (`counts`)
 * - where that node's child range begins in the flattened `indices` array (`offsets`)
 * - the concatenated child indices for all nodes (`indices`)
 *
 * This relies on the tree invariant: every non-root node has exactly
 * one direct parent, so the full set of child edges can be reconstructed from
 * the `parents` array alone.
 *
 * @tparam N Number of nodes in the plan.
 * @param parents Direct parent index for each node. The root is expected to use
 * the sentinel value `N`.
 * @return A compact child adjacency layout derived from `parents`.
 */
template <std::size_t N>
[[nodiscard]] consteval auto make_child_layout(const std::array<std::size_t, N>& parents) {
    child_layout<N> layout {};

    if constexpr (N > 0) {
        for (std::size_t node = 1; node < N; ++node) {
            ++layout.counts[parents[node]];
        }

        std::size_t offset = 0;
        for (std::size_t node = 0; node < N; ++node) {
            layout.offsets[node] = offset;
            offset += layout.counts[node];
        }

        auto next_offsets = layout.offsets;

        for (std::size_t node = 1; node < N; ++node) {
            const auto parent = parents[node];
            layout.indices[next_offsets[parent]++] = node;
        }
    }

    return layout;
}

template <typename... Nodes>
inline constexpr auto compiled_levels_v = make_level_array<Nodes...>();

template <typename... Nodes>
inline constexpr auto compiled_parent_indices_v =
    make_parent_index_array(compiled_levels_v<Nodes...>);

template <typename... Nodes>
inline constexpr auto compiled_child_layout_v =
    make_child_layout(compiled_parent_indices_v<Nodes...>);

template <typename... Nodes>
inline constexpr auto compiled_slot_indices_v =
    make_slot_index_array<sizeof...(Nodes)>();

} // namespace yorch::detail

namespace yorch {

/**
 * @brief Static plan compiled from a `task_tree_builder`.
 *
 * The compiled plan keeps the original node tuple while materializing the
 * tree's structural metadata into constexpr arrays that later executors can
 * consume directly.
 *
 * @tparam Nodes Recorded task-tree node types in insertion order.
 */
template <typename... Nodes>
struct compiled_plan {
    using tuple_type = std::tuple<Nodes...>;

    static constexpr std::size_t node_count = sizeof...(Nodes);
    static constexpr std::size_t max_level = detail::max_level_v<Nodes...>;
    static constexpr std::size_t no_parent = node_count;
    static constexpr std::size_t slot_count = node_count;

    /// Stored plan nodes in insertion order.
    tuple_type nodes {};

    /// Node levels in insertion order.
    static constexpr auto levels = detail::compiled_levels_v<Nodes...>;
    /// Direct parent index for each node; the root stores `no_parent`.
    static constexpr auto parent_indices = detail::compiled_parent_indices_v<Nodes...>;
    /// Number of direct children for each node.
    static constexpr auto child_counts = detail::compiled_child_layout_v<Nodes...>.counts;
    /// Start offset of each node's child list inside `child_indices`.
    static constexpr auto child_offsets = detail::compiled_child_layout_v<Nodes...>.offsets;
    /// Flattened direct-child adjacency storage.
    static constexpr auto child_indices = detail::compiled_child_layout_v<Nodes...>.indices;
    /// Logical output slot assigned to each node by now.
    static constexpr auto slot_indices = detail::compiled_slot_indices_v<Nodes...>;

    template <std::size_t I>
    using node_type = std::tuple_element_t<I, tuple_type>;

    template <std::size_t I>
    using task_type = typename node_type<I>::task_type;

    template <std::size_t I>
    using raw_result_type = detail::task_raw_result_t<task_type<I>>;

    template <std::size_t I>
    using output_type = detail::task_output_t<raw_result_type<I>>;

    template <std::size_t I>
    static constexpr std::size_t level = levels[I];

    template <std::size_t I>
    static constexpr std::size_t parent_index = parent_indices[I];

    template <std::size_t I>
    static constexpr std::size_t child_count = child_counts[I];

    template <std::size_t I>
    static constexpr std::size_t child_offset = child_offsets[I];

    template <std::size_t I>
    static constexpr std::size_t slot_index = slot_indices[I];

    template <std::size_t I, std::size_t ChildOrdinal>
    static constexpr std::size_t child_index =
        child_indices[child_offset<I> + ChildOrdinal];

    template <std::size_t I>
    [[nodiscard]] constexpr auto& entry() & noexcept {
        return std::get<I>(nodes);
    }

    template <std::size_t I>
    [[nodiscard]] constexpr const auto& entry() const& noexcept {
        return std::get<I>(nodes);
    }

    template <std::size_t I>
    [[nodiscard]] constexpr auto&& entry() && noexcept {
        return std::get<I>(std::move(nodes));
    }
};

/**
 * @brief Compiles a populated `task_tree_builder` into a static plan.
 *
 * This recovers direct-parent relations, direct-child adjacency, and
 * per-node output slot metadata while preserving the stored task objects.
 *
 * @tparam Nodes Recorded builder node types.
 * @param tree Source task tree builder.
 * @return A `compiled_plan` carrying the same node tuple and static metadata.
 */
template <typename... Nodes>
    requires (sizeof...(Nodes) > 0)
[[nodiscard]] constexpr auto compile_plan(task_tree_builder<Nodes...>&& tree) {
    return compiled_plan<Nodes...> {
        std::move(tree.nodes)
    };
}

template <typename... Nodes>
    requires (sizeof...(Nodes) > 0)
[[nodiscard]] constexpr auto compile_plan(const task_tree_builder<Nodes...>& tree) {
    return compiled_plan<Nodes...> {
        tree.nodes
    };
}

} // namespace yorch
