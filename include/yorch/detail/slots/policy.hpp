#pragma once

#include <cstddef>
#include <limits>

namespace yorch::detail {

/**
 * @brief Node-local slot semantics inferred from a task's result protocol.
 *
 * This policy answers whether a plan node logically owns a payload slot:
 * `none` is used for `void` output nodes; `maybe_payload` is used when a
 * successful-looking task invocation may still skip materializing a payload,
 * such as direct-output tasks; `must_payload` is used when success always
 * materializes a payload value.
 */
enum class slot_logical_policy : unsigned char {
    none,
    maybe_payload,
    must_payload,
};

/**
 * @brief Storage-level policy for a physical slot after layout selection.
 *
 * This policy is derived from every logical owner mapped to the same physical
 * slot. `none` is only the empty/default state before any owner contributes;
 * a physical slot is `must_payload` only when all owners are logically
 * `must_payload`; if any owner is logically `maybe_payload`, the physical slot
 * becomes `maybe_payload` and carries an engagement flag.
 */
enum class slot_physical_policy : unsigned char {
    none,
    maybe_payload,
    must_payload,
};

/**
 * @brief Sentinel physical slot index meaning "this node has no storage".
 *
 * Layouts assign real physical slot indices only to payload-producing nodes.
 * Nodes with `void` output keep this value instead, and compact slot storage
 * also reuses it to mark that no node currently owns an erased slot.
 */
inline constexpr std::size_t no_physical_slot = std::numeric_limits<std::size_t>::max();

} // namespace yorch::detail
