#pragma once

#include <concepts> // IWYU pragma: keep
#include <cstddef>
#include <tuple>
#include <type_traits>

#include "../../bind.hpp"
#include "../../specs.hpp"
#include "../../task_adapters/catch_as_failure.hpp"
#include "../../task_adapters/retry.hpp"

namespace yorch::detail {

template <typename Spec>
struct is_from_prev_spec : std::false_type {};

template <typename T>
struct is_from_prev_spec<from_prev_t<T>> : std::true_type {};

template <typename Spec>
inline constexpr bool is_from_prev_spec_v =
    is_from_prev_spec<std::remove_cvref_t<Spec>>::value;

template <typename Arg>
inline constexpr bool exact_const_lvalue_ref_v =
    std::is_lvalue_reference_v<Arg> &&
    std::is_const_v<std::remove_reference_t<Arg>>;

/**
 * @brief Checks whether every `from_prev(...)` binding inside a `bound_task`
 * maps to an exact `const T&` callable parameter.
 *
 * This helper is used by the current fan-out validation path. When a parent
 * node has multiple direct children, those children may only read the parent
 * payload through `from_prev(...)`; they must not take a mutable reference,
 * take the value by copy, or request an rvalue-style consuming parameter.
 *
 * The check is performed position-by-position across the bound task's spec
 * tuple and callable signature:
 *
 * - if the `I`-th spec is not `from_prev(...)`, that parameter position is
 *   irrelevant to this validation and therefore accepted
 * - if the `I`-th spec is `from_prev(...)`, then the callable's `I`-th
 *   parameter type must be exactly `const T&`
 *
 * The final result is the logical AND of all parameter positions.
 *
 * @tparam F Callable type stored inside `bound_task`.
 * @tparam SpecsTuple Tuple type storing one bound spec per callable parameter.
 * @tparam I Compile-time parameter indices used to walk the callable/spec pair.
 * @param Unused compile-time index sequence selecting the parameter positions.
 * @return `true` when all `from_prev(...)` positions bind as exact
 * `const T&`; otherwise `false`.
 */
template <typename F, typename SpecsTuple, std::size_t... I>
[[nodiscard]] consteval bool bound_task_fanout_prev_valid_impl(std::index_sequence<I...>) {
    return (((!is_from_prev_spec_v<std::tuple_element_t<I, SpecsTuple>>) ||
             exact_const_lvalue_ref_v<nth_arg_t<I, F>>) && ...);
}

template <typename Task>
struct fanout_prev_task_valid : std::true_type {};

template <typename F, typename... Specs>
struct fanout_prev_task_valid<bound_task<F, Specs...>>
    : std::bool_constant<
          bound_task_fanout_prev_valid_impl<
              std::remove_cvref_t<F>,
              std::tuple<Specs...>>(
              std::index_sequence_for<Specs...> {})> {};

template <typename F, typename T, typename... Specs>
struct fanout_prev_task_valid<bound_output_task<F, T, Specs...>>
    : std::bool_constant<
          bound_task_fanout_prev_valid_impl<
              std::remove_cvref_t<F>,
              std::tuple<Specs...>>(
              std::index_sequence_for<Specs...> {})> {};

template <typename Task>
struct fanout_prev_task_valid<catch_failure_task<Task>>
    : fanout_prev_task_valid<Task> {};

template <typename Task, typename Policy>
struct fanout_prev_task_valid<catch_failure_with_policy_task<Task, Policy>>
    : fanout_prev_task_valid<Task> {};

template <typename Task, typename Policy>
struct fanout_prev_task_valid<retry_task<Task, Policy>>
    : fanout_prev_task_valid<Task> {};

template <typename Task>
inline constexpr bool fanout_prev_task_valid_v =
    fanout_prev_task_valid<std::remove_cvref_t<Task>>::value;

template <typename SpecsTuple, std::size_t... I>
[[nodiscard]] consteval bool bound_task_uses_from_prev_impl(std::index_sequence<I...>) {
    return (is_from_prev_spec_v<std::tuple_element_t<I, SpecsTuple>> || ...);
}

template <typename Task>
struct task_uses_from_prev : std::false_type {};

template <typename F, typename... Specs>
struct task_uses_from_prev<bound_task<F, Specs...>>
    : std::bool_constant<
          bound_task_uses_from_prev_impl<
              std::tuple<Specs...>>(
              std::index_sequence_for<Specs...> {})> {};

template <typename F, typename T, typename... Specs>
struct task_uses_from_prev<bound_output_task<F, T, Specs...>>
    : std::bool_constant<
          bound_task_uses_from_prev_impl<
              std::tuple<Specs...>>(
              std::index_sequence_for<Specs...> {})> {};

template <typename Task>
struct task_uses_from_prev<catch_failure_task<Task>>
    : task_uses_from_prev<Task> {};

template <typename Task, typename Policy>
struct task_uses_from_prev<catch_failure_with_policy_task<Task, Policy>>
    : task_uses_from_prev<Task> {};

template <typename Task, typename Policy>
struct task_uses_from_prev<retry_task<Task, Policy>>
    : task_uses_from_prev<Task> {};

template <typename Task>
inline constexpr bool task_uses_from_prev_v =
    task_uses_from_prev<std::remove_cvref_t<Task>>::value;

/**
 * @brief Minimal static-plan protocol required by the fan-out `from_prev`
 * validation path for a specific node index.
 *
 * This intentionally checks only the plan surface consumed by
 * `node_fanout_prev_valid()`: a root sentinel, per-node parent lookup,
 * per-node task type exposure, and per-node child count exposure.
 *
 * The concept is purposefully interface-based rather than tied to
 * `compiled_plan<...>` so future plan backends can reuse the same validation
 * logic as long as they expose the same compile-time protocol.
 *
 * @tparam Plan Candidate plan type.
 * @tparam I Node index being validated.
 */
template <typename Plan, std::size_t I>
concept fanout_validatable_plan_node =
    requires {
        { Plan::no_parent } -> std::convertible_to<std::size_t>;
        { Plan::template parent_index<I> } -> std::convertible_to<std::size_t>;
        typename Plan::template task_type<I>;
    };

/**
 * @brief Checks whether node `I` is structurally allowed to use
 * `from_prev(...)` at all.
 *
 * This validation is intentionally narrower than full `from_prev` type
 * checking. It only answers the question "does this node have a readable
 * direct-parent payload source?" and therefore rejects two cases early at the
 * `run_plan(...)` boundary:
 *
 * - the root node uses `from_prev(...)`
 * - the direct parent exists but its compiled output type is `void`
 *
 * It does not attempt to validate the requested payload type `T` inside
 * `from_prev<T>()`, nor whether that `T` matches the direct parent payload
 * exactly. Those finer-grained binding checks remain in the resolution layer,
 * so this helper stays focused on plan-structure legality instead of becoming
 * a full duplicate of `resolve_as(from_prev_t<...>, ...)`.
 *
 * @tparam Plan Compiled plan type being validated.
 * @tparam I Node index within `Plan`.
 * @return `true` when node `I` either does not use `from_prev(...)` or has a
 * non-void direct parent payload source; otherwise `false`.
 */
template <typename Plan, std::size_t I>
    requires fanout_validatable_plan_node<Plan, I>
[[nodiscard]] consteval bool node_prev_source_valid() {
    using task_t = typename Plan::template task_type<I>;

    if constexpr (Plan::template parent_index<I> == Plan::no_parent) {
        return !task_uses_from_prev_v<task_t>;
    } else {
        constexpr auto parent = Plan::template parent_index<I>;

        if constexpr (std::is_void_v<typename Plan::template output_type<parent>>) {
            return !task_uses_from_prev_v<task_t>;
        } else {
            return true;
        }
    }
}

template <typename Plan, std::size_t I>
    requires fanout_validatable_plan_node<Plan, I>
[[nodiscard]] consteval bool node_fanout_prev_valid() {
    if constexpr (Plan::template parent_index<I> == Plan::no_parent) {
        return true;
    } else {
        constexpr auto parent = Plan::template parent_index<I>;

        if constexpr (Plan::template child_count<parent> <= 1) {
            return true;
        } else {
            return fanout_prev_task_valid_v<typename Plan::template task_type<I>>;
        }
    }
}

template <typename Plan, std::size_t... I>
    requires (fanout_validatable_plan_node<Plan, I> && ...)
[[nodiscard]] consteval bool plan_prev_source_valid_impl(std::index_sequence<I...>) {
    return (node_prev_source_valid<Plan, I>() && ...);
}

template <typename Plan, std::size_t... I>
    requires (fanout_validatable_plan_node<Plan, I> && ...)
[[nodiscard]] consteval bool plan_fanout_prev_valid_impl(std::index_sequence<I...>) {
    return (node_fanout_prev_valid<Plan, I>() && ...);
}

template <typename Plan>
inline constexpr bool plan_prev_source_valid_v =
    plan_prev_source_valid_impl<Plan>(std::make_index_sequence<Plan::node_count> {});

template <typename Plan>
inline constexpr bool plan_fanout_prev_valid_v =
    plan_fanout_prev_valid_impl<Plan>(std::make_index_sequence<Plan::node_count> {});

} // namespace yorch::detail
