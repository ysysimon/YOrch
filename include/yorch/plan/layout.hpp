#pragma once

#include <array>
#include <cstddef>

namespace yorch::detail {

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
