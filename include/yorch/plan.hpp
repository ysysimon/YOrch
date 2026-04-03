#pragma once

#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "result.hpp"
#include "detail/slot_policy.hpp"
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
 * supplied by opt-in specializations or, in the common case, by tasks that
 * declare `raw_result_type` directly.
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
 * @brief Convenience alias for a task's raw return type.
 *
 * This avoids repeatedly spelling `typename task_raw_result<T>::type`.
 *
 * @tparam Task Task type being analyzed.
 */
template <typename Task>
using task_raw_result_t = typename task_raw_result<Task>::type;

template <typename Task>
concept plannable_task =
    requires {
        typename task_raw_result<std::remove_cvref_t<Task>>::type;
    };

template <typename Task, typename = void>
struct plan_declared_task_output;

template <typename Task>
struct plan_declared_task_output<
    Task,
    std::void_t<typename std::remove_cvref_t<Task>::output_type>
> {
    using type = typename std::remove_cvref_t<Task>::output_type;
};

template <typename Task, typename = void>
struct has_plan_declared_task_output : std::false_type {};

template <typename Task>
struct has_plan_declared_task_output<
    Task,
    std::void_t<typename std::remove_cvref_t<Task>::output_type>
> : std::true_type {};

template <typename R>
struct task_output_type_impl {
    using type = R;
};

template <>
struct task_output_type_impl<step_result> {
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

template <typename Task, typename = void>
struct task_output_for {
    using type = task_output_t<task_raw_result_t<Task>>;
};

template <typename Task>
struct task_output_for<
    Task,
    std::void_t<typename plan_declared_task_output<Task>::type>
> {
    using type = typename plan_declared_task_output<Task>::type;
};

template <typename Task>
using task_output_for_t = typename task_output_for<Task>::type;

template <typename Task, typename = void>
struct task_slot_policy {
private:
    using raw_t = task_raw_result_t<Task>;

public:
    static constexpr slot_policy value =
        std::is_void_v<raw_t> ||
                std::is_same_v<raw_t, step_result>
            ? slot_policy::none
        : is_task_result_v<raw_t>
            ? slot_policy::maybe_payload
            : slot_policy::must_payload;
};

template <typename Task>
struct task_slot_policy<
    Task,
    std::void_t<typename plan_declared_task_output<Task>::type>
> {
    static constexpr slot_policy value = slot_policy::maybe_payload;
};

template <typename Task>
inline constexpr slot_policy task_slot_policy_v =
    task_slot_policy<Task>::value;

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

template <typename Node>
concept plannable_plan_node =
    requires {
        typename Node::task_type;
    } &&
    plannable_task<typename Node::task_type>;

template <typename... Nodes>
concept plannable_plan_nodes =
    (plannable_plan_node<Nodes> && ...);

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
    using output_type = detail::task_output_for_t<task_type<I>>;

    template <std::size_t I>
    static constexpr detail::slot_policy slot_policy_for =
        detail::task_slot_policy_v<task_type<I>>;

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

template <typename Tree>
struct compiled_plan_from;

/**
 * @brief Maps a `task_tree_builder` type to its corresponding compiled plan type.
 *
 * This public trait exists so user code can name the plan type in class
 * members, aliases, and other declaration contexts without spelling a
 * `decltype(compile_plan(...))` expression.
 *
 * @tparam Tree Task tree builder type.
 */
template <typename... Nodes>
struct compiled_plan_from<task_tree_builder<Nodes...>> {
    using type = compiled_plan<Nodes...>;
};

/**
 * @brief Convenience alias for the compiled plan type produced from a tree type.
 *
 * `compiled_plan_t<Tree>` removes cv/ref qualifiers from `Tree` before mapping
 * it to `compiled_plan<...>`.
 *
 * @tparam Tree Task tree builder type.
 */
template <typename Tree>
using compiled_plan_t = typename compiled_plan_from<std::remove_cvref_t<Tree>>::type;

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
    requires (sizeof...(Nodes) > 0) &&
             detail::plannable_plan_nodes<Nodes...>
[[nodiscard]] constexpr auto compile_plan(task_tree_builder<Nodes...>&& tree) {
    return compiled_plan<Nodes...> {
        std::move(tree.nodes)
    };
}

template <typename... Nodes>
    requires (sizeof...(Nodes) > 0) &&
             detail::plannable_plan_nodes<Nodes...>
[[nodiscard]] constexpr auto compile_plan(const task_tree_builder<Nodes...>& tree) {
    return compiled_plan<Nodes...> {
        tree.nodes
    };
}

} // namespace yorch
