#pragma once

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "../detail/executor/plan_validation.hpp"
#include "../task_tree.hpp"
#include "layout.hpp"
#include "traits.hpp"

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
    using fanout_policy_type = typename node_type<I>::fanout_policy_type;

    template <std::size_t I>
    using raw_result_type = detail::task_raw_result_t<task_type<I>>;

    template <std::size_t I>
    using output_type = detail::task_output_for_t<task_type<I>>;

    template <std::size_t I>
    static constexpr detail::output_storage_mode output_storage_mode_for =
        detail::task_output_storage_mode_v<task_type<I>>;

    template <std::size_t I>
    static constexpr detail::slot_logical_policy slot_logical_policy_for =
        detail::task_slot_logical_policy_v<task_type<I>>;

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
    detail::emit_plan_diagnostic<compiled_plan<Nodes...>>();

    return compiled_plan<Nodes...> {
        std::move(tree.nodes)
    };
}

template <typename... Nodes>
    requires (sizeof...(Nodes) > 0) &&
             detail::plannable_plan_nodes<Nodes...>
[[nodiscard]] constexpr auto compile_plan(const task_tree_builder<Nodes...>& tree) {
    detail::emit_plan_diagnostic<compiled_plan<Nodes...>>();

    return compiled_plan<Nodes...> {
        tree.nodes
    };
}

} // namespace yorch
