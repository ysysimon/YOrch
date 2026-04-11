#pragma once

#include <type_traits>

namespace yorch {

/// Default fanout policy.
///
/// This preserves the current implicit behavior:
/// - if a node has at most one direct child that uses prev access, that child may
///   use any declared prev access semantics supported by the task;
/// - if a node fans out to multiple direct children that use prev access, only
///   shared access is allowed, namely `borrow_prev` and `copy_prev`.
///
/// Use this policy when you want the existing behavior and do not need to make
/// the fanout contract explicit.
struct fanout_auto_policy {};

/// Explicit readonly fanout policy.
///
/// A node marked with this policy always exposes its output as a shared,
/// non-exclusive upstream source. Direct children may use:
/// - `borrow_prev`
/// - `copy_prev`
///
/// Direct children may not use:
/// - `borrow_prev_mut`
/// - `consume_prev`
///
/// This is useful when a parent result is intended to be broadcast to multiple
/// consumers without allowing any child to mutate or consume the original source.
struct fanout_shared_readonly_policy {};

/// Mixed fanout policy that allows one consumer plus any number of copies.
///
/// A node marked with this policy may have direct children that use prev access
/// only under the following rules:
/// - at most one direct child may use `consume_prev`;
/// - every other direct child that uses prev access must use `copy_prev`;
/// - `borrow_prev` and `borrow_prev_mut` are not allowed.
///
/// At runtime the executor stages the fanout before children begin execution:
/// copy payloads are materialized first for `copy_prev` children, and the
/// original upstream payload is then reserved for the single `consume_prev`
/// child. This keeps the semantics independent of sibling execution order.
struct fanout_consume_with_copies_policy {};

} // namespace yorch

namespace yorch::detail {

template <typename FanoutPolicy>
concept fanout_policy =
    std::is_same_v<std::remove_cvref_t<FanoutPolicy>, yorch::fanout_auto_policy> ||
    std::is_same_v<std::remove_cvref_t<FanoutPolicy>, yorch::fanout_shared_readonly_policy> ||
    std::is_same_v<std::remove_cvref_t<FanoutPolicy>, yorch::fanout_consume_with_copies_policy>;

} // namespace yorch::detail
