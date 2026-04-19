#pragma once

#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "../../slots/policies.hpp"
#include "erased_slot.hpp"
#include "policy.hpp"
#include "typed_slot.hpp"

namespace yorch::detail {

/**
 * @brief Builds the compile-time node -> has-payload table for a plan.
 *
 * Each entry is `true` when the corresponding node needs owned physical slot
 * storage. Nodes whose logical output is statically forwarded from their
 * direct parent do not contribute storage here.
 */
template <typename Plan, std::size_t... I>
[[nodiscard]] consteval auto make_payload_node_array(std::index_sequence<I...>) {
    return std::array<bool, sizeof...(I)> {
        (Plan::template output_storage_mode_for<I> == output_storage_mode::owned)...
    };
}

/**
 * @brief Builds the compile-time node -> logical slot policy table.
 *
 * Each entry mirrors `Plan::slot_logical_policy_for<I>`. Layouts use this 
 * collected table when multiple logical nodes map to the same physical slot
 * and their logical policies must be joined into one storage-level 
 * `slot_physical_policy`. 
 */
template <typename Plan, std::size_t... I>
[[nodiscard]] consteval auto make_slot_logical_policy_array(std::index_sequence<I...>) {
    return std::array<slot_logical_policy, sizeof...(I)> {
        Plan::template slot_logical_policy_for<I>...
    };
}

template <typename Plan>
inline constexpr auto payload_node_array_v =
    make_payload_node_array<Plan>(std::make_index_sequence<Plan::node_count> {});

template <typename Plan>
inline constexpr auto slot_logical_policy_array_v =
    make_slot_logical_policy_array<Plan>(std::make_index_sequence<Plan::node_count> {});

template <typename Plan>
inline constexpr std::size_t payload_node_count_v = [] {
    std::size_t count = 0;
    for (const bool payload : payload_node_array_v<Plan>) {
        if (payload) {
            ++count;
        }
    }
    return count;
}();

/**
 * @brief Builds the one-to-one node -> physical slot index table.
 *
 * In this layout, every owned-storage node gets its own distinct physical
 * slot. Nodes without owned storage keep the
 * `no_physical_slot` sentinel instead.
 */
template <typename Plan>
[[nodiscard]] consteval auto make_one_to_one_physical_slot_indices() {
    std::array<std::size_t, Plan::node_count> indices {};
    indices.fill(no_physical_slot);

    std::size_t next = 0;
    for (std::size_t node = 0; node < Plan::node_count; ++node) {
        if (payload_node_array_v<Plan>[node]) {
            indices[node] = next++;
        }
    }

    return indices;
}

/**
 * @brief Intermediate index data for the serial-DFS compact slot layout.
 */
template <typename Plan>
struct compact_slot_index_layout {
    // Node -> physical slot mapping; non-owned nodes keep `no_physical_slot`.
    std::array<std::size_t, Plan::node_count> physical_slot_indices {};
    // Payload depth along each node's root-to-node path, including the node.
    std::array<std::size_t, Plan::node_count> payload_depths {};
    // Total physical slots needed, equal to the maximum payload depth.
    std::size_t physical_slot_count = 0;
};

/**
 * @brief Builds the serial-DFS compact node -> physical slot layout.
 *
 * The serial executor only keeps owned payload slots on the current root-to-node DFS path
 * live at the same time. Sibling subtrees run one after another, so payload
 * nodes at the same path payload depth can safely reuse the same physical slot.
 *
 * For example, given:
 *
 * ```
 * 0: int
 * |- 1: string
 * |  `- 2: void
 * `- 3: void
 *    `- 4: bool
 * ```
 *
 * the payload depths are `node 0 -> 1`, `node 1 -> 2`, `node 2 -> 2`,
 * `node 3 -> 1`, and `node 4 -> 2`. Payload nodes use `depth - 1` as the
 * physical slot index, so `node 0 -> slot 0`, while `node 1` and `node 4`
 * both map to `slot 1`. The final `physical_slot_count` is the maximum payload
 * depth across the tree.
 */
template <typename Plan>
[[nodiscard]] consteval auto make_serial_dfs_compact_slot_layout() {
    compact_slot_index_layout<Plan> layout {};
    layout.physical_slot_indices.fill(no_physical_slot);

    for (std::size_t node = 0; node < Plan::node_count; ++node) {
        const auto parent = Plan::parent_indices[node];
        const auto parent_depth =
            parent == Plan::no_parent ? 0 : layout.payload_depths[parent];
        layout.payload_depths[node] =
            parent_depth + (payload_node_array_v<Plan>[node] ? 1U : 0U);

        if (payload_node_array_v<Plan>[node]) {
            layout.physical_slot_indices[node] = layout.payload_depths[node] - 1;
            if (layout.payload_depths[node] > layout.physical_slot_count) {
                layout.physical_slot_count = layout.payload_depths[node];
            }
        }
    }

    return layout;
}

/**
 * @brief Joins one node's logical policy into a physical slot policy.
 *
 * `left` is the physical policy accumulated for a physical slot so far, and
 * `right` is the next logical owner mapped to that slot. Logical `none` does
 * not contribute. An empty physical policy adopts the first payload owner, and
 * any `maybe_payload` owner upgrades the whole physical slot to
 * `maybe_payload`, because the shared storage must carry an engagement flag if
 * any possible owner may skip materializing a payload. The result stays
 * `must_payload` only when every contributing owner is logically `must_payload`.
 */
constexpr slot_physical_policy join_slot_physical_policy(
    slot_physical_policy left,
    slot_logical_policy right) noexcept {
    if (right == slot_logical_policy::none) {
        return left;
    }

    if (left == slot_physical_policy::none) {
        return right == slot_logical_policy::must_payload
            ? slot_physical_policy::must_payload
            : slot_physical_policy::maybe_payload;
    }

    if (left == slot_physical_policy::maybe_payload ||
        right == slot_logical_policy::maybe_payload) {
        return slot_physical_policy::maybe_payload;
    }

    return slot_physical_policy::must_payload;
}

/**
 * @brief Computes the required raw storage size for each physical slot.
 *
 * `indices` is the node -> physical slot mapping produced by a slot layout.
 * The fold expression expands over node indices, skips `void` nodes, then uses
 * `indices[I]` to update the size for the physical slot owned by node `I`.
 * When multiple nodes map to the same physical slot, the stored size is the
 * maximum `sizeof(output_type<I>)` among those owners.
 *
 * @tparam PhysicalSlotCount Number of physical slots in the layout.
 * @tparam I Node indices supplied by the `std::index_sequence<I...>` argument.
 * @param indices Node -> physical slot mapping; `no_physical_slot` entries are
 * ignored.
 */
template <typename Plan, std::size_t PhysicalSlotCount, std::size_t... I>
[[nodiscard]] consteval auto make_physical_slot_sizes(
    const std::array<std::size_t, Plan::node_count>& indices,
    std::index_sequence<I...>) {
    std::array<std::size_t, PhysicalSlotCount> sizes {};
    ([&] {
        if constexpr (!std::is_void_v<typename Plan::template output_type<I>>) {
            const auto slot = indices[I];
            if (slot != no_physical_slot) {
                sizes[slot] = sizes[slot] < sizeof(typename Plan::template output_type<I>)
                    ? sizeof(typename Plan::template output_type<I>)
                    : sizes[slot];
            }
        }
    }(), ...);
    return sizes;
}

/**
 * @brief Computes the required alignment for each physical slot.
 *
 * This mirrors `make_physical_slot_sizes(...)`, but records the maximum
 * `alignof(output_type<I>)` for every physical slot. When multiple nodes share
 * one erased slot, the slot must satisfy the strictest alignment requirement
 * among all mapped payload owners.
 *
 * @tparam PhysicalSlotCount Number of physical slots in the layout.
 * @tparam I Node indices supplied by the `std::index_sequence<I...>` argument.
 * @param indices Node -> physical slot mapping; `no_physical_slot` entries are
 * ignored.
 */
template <typename Plan, std::size_t PhysicalSlotCount, std::size_t... I>
[[nodiscard]] consteval auto make_physical_slot_alignments(
    const std::array<std::size_t, Plan::node_count>& indices,
    std::index_sequence<I...>) {
    std::array<std::size_t, PhysicalSlotCount> alignments {};
    ([&] {
        if constexpr (!std::is_void_v<typename Plan::template output_type<I>>) {
            const auto slot = indices[I];
            if (slot != no_physical_slot) {
                alignments[slot] = alignments[slot] < alignof(typename Plan::template output_type<I>)
                    ? alignof(typename Plan::template output_type<I>)
                    : alignments[slot];
            }
        }
    }(), ...);
    return alignments;
}

/**
 * @brief Computes the storage policy for each physical slot.
 *
 * The fold expression expands over node indices, uses `indices[I]` to find the
 * physical slot for node `I`, and joins that node's
 * `slot_logical_policy_for<I>` into the accumulated `slot_physical_policy`.
 * Nodes mapped to `no_physical_slot` are ignored. When multiple nodes share one
 * physical slot, the result is `maybe_payload` if any owner is logically
 * `maybe_payload`; otherwise it remains `must_payload`.
 *
 * @tparam PhysicalSlotCount Number of physical slots in the layout.
 * @tparam I Node indices supplied by the `std::index_sequence<I...>` argument.
 * @param indices Node -> physical slot mapping; `no_physical_slot` entries are
 * ignored.
 */
template <typename Plan, std::size_t PhysicalSlotCount, std::size_t... I>
[[nodiscard]] consteval auto make_physical_slot_policies(
    const std::array<std::size_t, Plan::node_count>& indices,
    std::index_sequence<I...>) {
    std::array<slot_physical_policy, PhysicalSlotCount> policies {};
    policies.fill(slot_physical_policy::none);
    (((indices[I] != no_physical_slot)
          ? policies[indices[I]] = join_slot_physical_policy(
                policies[indices[I]],
                Plan::template slot_logical_policy_for<I>)
          : slot_physical_policy::none),
     ...);
    return policies;
}

template <typename Plan, typename LayoutPolicy>
struct plan_slot_layout;

template <typename Plan>
struct plan_slot_layout<Plan, yorch::slot_layout_one_to_one_policy> {
    static constexpr auto physical_slot_indices =
        make_one_to_one_physical_slot_indices<Plan>();
    static constexpr std::size_t physical_slot_count = payload_node_count_v<Plan>;
    static constexpr auto physical_slot_sizes =
        make_physical_slot_sizes<Plan, physical_slot_count>(
            physical_slot_indices,
            std::make_index_sequence<Plan::node_count> {});
    static constexpr auto physical_slot_alignments =
        make_physical_slot_alignments<Plan, physical_slot_count>(
            physical_slot_indices,
            std::make_index_sequence<Plan::node_count> {});
    static constexpr auto physical_slot_policies =
        make_physical_slot_policies<Plan, physical_slot_count>(
            physical_slot_indices,
            std::make_index_sequence<Plan::node_count> {});

    template <std::size_t I>
    static constexpr std::size_t physical_slot_index = physical_slot_indices[I];

    template <std::size_t K>
    static constexpr slot_physical_policy physical_slot_policy = physical_slot_policies[K];
};

template <typename Plan>
struct plan_slot_layout<Plan, yorch::slot_layout_serial_dfs_compact_policy> {
    static constexpr auto compact_layout =
        make_serial_dfs_compact_slot_layout<Plan>();
    static constexpr auto physical_slot_indices = compact_layout.physical_slot_indices;
    static constexpr auto payload_depths = compact_layout.payload_depths;
    static constexpr std::size_t physical_slot_count = compact_layout.physical_slot_count;
    static constexpr auto physical_slot_sizes =
        make_physical_slot_sizes<Plan, physical_slot_count>(
            physical_slot_indices,
            std::make_index_sequence<Plan::node_count> {});
    static constexpr auto physical_slot_alignments =
        make_physical_slot_alignments<Plan, physical_slot_count>(
            physical_slot_indices,
            std::make_index_sequence<Plan::node_count> {});
    static constexpr auto physical_slot_policies =
        make_physical_slot_policies<Plan, physical_slot_count>(
            physical_slot_indices,
            std::make_index_sequence<Plan::node_count> {});

    template <std::size_t I>
    static constexpr std::size_t physical_slot_index = physical_slot_indices[I];

    template <std::size_t K>
    static constexpr slot_physical_policy physical_slot_policy = physical_slot_policies[K];
};

/**
 * @brief Type factory for a layout's erased-slot storage tuple.
 *
 * This declaration is intentionally body-less: callers use it only in
 * `decltype(...)` to expand physical slot indices `K...` into
 * `std::tuple<erased_slot<size, alignment, policy>...>`.
 */
template <typename Layout, std::size_t... K>
[[nodiscard]] auto make_erased_slots_tuple(std::index_sequence<K...>)
    -> std::tuple<erased_slot<
        Layout::physical_slot_sizes[K],
        Layout::physical_slot_alignments[K],
        Layout::template physical_slot_policy<K>>...>;

/**
 * @brief Erased backend storage tuple for compact/layout-driven plan slots.
 */
template <typename Layout>
using plan_erased_slots_tuple_t =
    decltype(make_erased_slots_tuple<Layout>(
        std::make_index_sequence<Layout::physical_slot_count> {}));

/**
 * @brief Finds the unique node that owns a one-to-one physical slot.
 *
 * The one-to-one layout maps each payload-producing node to a distinct
 * physical slot and skips `void` nodes. The typed backend tuple is therefore
 * expanded by physical slot index, not by node index; this helper reverses the
 * mapping so `make_typed_slots_tuple(...)` can recover the owner node's
 * `output_type` and `slot_logical_policy`.
 *
 * For example, if nodes `0:int`, `1:string`, `2:void`, `3:void`, `4:bool` map 
 * to slots `0`, `1`, `no_physical_slot`, `no_physical_slot`, `2`. 
 * then physical slot `2` is owned by node `4`.
 */
template <typename Plan, std::size_t PhysicalSlot>
[[nodiscard]] consteval std::size_t one_to_one_physical_slot_owner_node() {
    constexpr auto indices = make_one_to_one_physical_slot_indices<Plan>();
    for (std::size_t node = 0; node < Plan::node_count; ++node) {
        if (indices[node] == PhysicalSlot) {
            return node;
        }
    }

    return no_physical_slot;
}

/**
 * @brief Type factory for a plan's typed one-to-one slot tuple.
 *
 * This declaration is intentionally body-less: callers use it only in
 * `decltype(...)` to expand physical slot indices `K...` into typed slots for
 * their unique payload-owner node. Void-output nodes do not appear in this
 * tuple, so its size matches the one-to-one layout's `physical_slot_count`.
 */
template <typename Plan, std::size_t... K>
[[nodiscard]] auto make_typed_slots_tuple(std::index_sequence<K...>)
    -> std::tuple<typed_slot<
        typename Plan::template output_type<one_to_one_physical_slot_owner_node<Plan, K>()>,
        Plan::template slot_logical_policy_for<one_to_one_physical_slot_owner_node<Plan, K>()>>...>;

/**
 * @brief Typed backend storage tuple for one-to-one owned payload slots.
 */
template <typename Plan>
using plan_typed_slots_tuple_t =
    decltype(make_typed_slots_tuple<Plan>(
        std::make_index_sequence<payload_node_count_v<Plan>> {}));

/**
 * @brief True when node `I` owns physical storage for its output.
 */
template <typename Plan, std::size_t I>
concept payload_node =
    Plan::template output_storage_mode_for<I> == output_storage_mode::owned;

} // namespace yorch::detail
