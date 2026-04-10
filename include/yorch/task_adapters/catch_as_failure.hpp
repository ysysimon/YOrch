#pragma once

#include "concepts.hpp" // IWYU pragma: keep
#include "../detail/task_adapters/result_helpers.hpp"

namespace yorch {

/**
 * @brief Exception-catching adapter for tasks that rely on the default failure
 * mapping.
 *
 * The wrapped task itself may throw. This adapter catches any exception,
 * converts it into a compatible failure raw result, and exposes a `noexcept`
 * `invoke_raw(...)` protocol to the outer executor.
 *
 * @tparam Task Stored task type.
 */
template <typename Task>
struct catch_failure_task
    : detail::catch_failure_task_raw_result_base<Task>,
      detail::forwarded_task_output_base<Task> {
    Task task;

    constexpr explicit catch_failure_task(Task&& stored_task)
        noexcept(std::is_nothrow_move_constructible_v<Task>)
        : task(std::move(stored_task)) {}

    constexpr explicit catch_failure_task(const Task& stored_task)
        noexcept(std::is_nothrow_copy_constructible_v<Task>)
        : task(stored_task) {}

    /**
     * @brief Executes the wrapped task and maps thrown exceptions to failure.
     *
     * @tparam Ctx Context schema.
     * @tparam Prev Direct-parent slot view type.
     * @param ec Borrowed execution context.
     * @return The wrapped task's raw result on success, or a synthesized
     * failure result on exception.
     */
    template <typename Ctx, typename Prev>
        requires default_catchable_task<Task, Ctx, Prev>
    constexpr auto invoke_raw(exec_context<Ctx, Prev>& ec) noexcept {
        using raw_result_t = detail::raw_task_result_t<Task&, Ctx, Prev>;

        static_assert(detail::declared_task_raw_result_matches_invoke_raw_v<Task, Ctx, Prev>,
                      "Wrapped task raw_result_type must match invoke_raw(exec_context<...>&) return type");
        static_assert(detail::default_catch_supported_v<raw_result_t>,
                      "Default catch_as_failure only supports tasks returning void or step_result");

        try {
            if constexpr (std::is_void_v<raw_result_t>) {
                task.invoke_raw(ec);
                return step_result::success();
            } else {
                return task.invoke_raw(ec);
            }
        } catch (...) {
            return detail::default_catch_failure_result<raw_result_t>();
        }
    }

    template <typename Ctx, typename Prev, typename U = Task>
        requires direct_output_task<U, Ctx, Prev>
    constexpr step_result invoke_into(
        exec_context<Ctx, Prev>& ec,
        direct_out<detail::declared_task_output_t<U>> out) noexcept {
        try {
            return task.invoke_into(ec, out);
        } catch (...) {
            return step_result::failure();
        }
    }
};

/**
 * @brief Exception-catching adapter for tasks with a user-supplied fallback
 * policy.
 *
 * The policy must be `noexcept` and support a captured exception either by
 * value (`std::exception_ptr`) or by const lvalue reference
 * (`const std::exception_ptr&`), allowing the wrapper to present a
 * `noexcept invoke_raw(...)` surface to the executor while still giving users
 * control over failure-path raw results.
 *
 * @tparam Task Stored task type.
 * @tparam Policy Stored fallback policy type.
 */
template <typename Task, typename Policy>
struct catch_failure_with_policy_task
    : detail::catch_failure_with_policy_task_raw_result_base<Task, Policy>,
      detail::forwarded_task_output_base<Task> {
    Task task;
    Policy policy;

    constexpr catch_failure_with_policy_task(Task&& stored_task, Policy&& stored_policy)
        noexcept(std::is_nothrow_move_constructible_v<Task> &&
                 std::is_nothrow_move_constructible_v<Policy>)
        : task(std::move(stored_task)),
          policy(std::move(stored_policy)) {}

    constexpr catch_failure_with_policy_task(const Task& stored_task, const Policy& stored_policy)
        noexcept(std::is_nothrow_copy_constructible_v<Task> &&
                 std::is_nothrow_copy_constructible_v<Policy>)
        : task(stored_task),
          policy(stored_policy) {}

    /**
     * @brief Executes the wrapped task and delegates exception handling to the
     * stored fallback policy.
     *
     * @tparam Ctx Context schema.
     * @tparam Prev Direct-parent slot view type.
     * @param ec Borrowed execution context.
     * @return The wrapped task's raw result on success, or the policy-produced
     * raw result on exception.
     */
    template <typename Ctx, typename Prev>
        requires catch_policy_compatible_task<Task, Policy, Ctx, Prev>
    constexpr auto invoke_raw(exec_context<Ctx, Prev>& ec) noexcept {
        using raw_result_t = detail::raw_task_result_t<Task&, Ctx, Prev>;

        static_assert(detail::declared_task_raw_result_matches_invoke_raw_v<Task, Ctx, Prev>,
                      "Wrapped task raw_result_type must match invoke_raw(exec_context<...>&) return type");
        static_assert(detail::catch_policy_supported_v<raw_result_t, Policy>,
                      "catch_as_failure(task, policy) requires a noexcept fallback taking std::exception_ptr and returning a compatible result");

        try {
            if constexpr (std::is_void_v<raw_result_t>) {
                task.invoke_raw(ec);
                return detail::void_task_success_result<detail::policy_result_t<Policy>>();
            } else {
                return task.invoke_raw(ec);
            }
        } catch (...) {
            return detail::policy_catch_failure_result<raw_result_t>(policy, std::current_exception());
        }
    }

    template <typename Ctx, typename Prev, typename U = Task>
        requires catch_policy_compatible_direct_output_task<U, Policy, Ctx, Prev>
    constexpr step_result invoke_into(
        exec_context<Ctx, Prev>& ec,
        direct_out<detail::declared_task_output_t<U>> out) noexcept {
        try {
            return task.invoke_into(ec, out);
        } catch (...) {
            return std::invoke(policy, std::current_exception());
        }
    }
};

/**
 * @brief Wraps a task so thrown exceptions become default failure results.
 *
 * This overload is intentionally narrow: it only supports raw return
 * categories for which the library can synthesize a failure result without a
 * user-provided payload.
 *
 * Note that the exception source is not limited to the final callable body.
 * For bound tasks, argument resolution, implicit conversions, and temporary
 * argument construction all happen inside `invoke_raw(...)` and therefore also
 * participate in the task's effective `noexcept` surface. If any of those
 * steps may throw, the resulting task is no longer `noexcept` and may need to
 * be adapted through `catch_as_failure(...)` before it can re-enter the main
 * executor path.
 *
 * @tparam Task Task type to wrap.
 * @param task Task object exposing `invoke_raw(...)`.
 * @return A `catch_failure_task` wrapper.
 */
template <typename Task>
constexpr auto catch_as_failure(Task&& task) {
    return catch_failure_task<std::decay_t<Task>>{
        std::forward<Task>(task)
    };
}

/**
 * @brief Wraps a task so thrown exceptions are routed to a user fallback
 * policy.
 *
 * The policy contract is a `noexcept` call accepting the captured exception
 * either by value (`std::exception_ptr`) or by const lvalue reference
 * (`const std::exception_ptr&`). Its return type must be compatible with the
 * wrapped task's raw return category.
 *
 * As with the default overload, exceptions may originate anywhere inside the
 * wrapped task's `invoke_raw(...)` path, including parameter resolution and
 * conversion work performed before the final callable body is entered. This
 * overload is therefore suitable both for callables that throw directly and
 * for bound tasks whose argument construction path can fail.
 *
 * @tparam Task Task type to wrap.
 * @tparam Policy Fallback policy type.
 * @param task Task object exposing `invoke_raw(...)`.
 * @param policy Exception fallback policy.
 * @return A `catch_failure_with_policy_task` wrapper.
 */
template <typename Task, typename Policy>
constexpr auto catch_as_failure(Task&& task, Policy&& policy) {
    return catch_failure_with_policy_task<
        std::decay_t<Task>,
        std::decay_t<Policy>
    >{
        std::forward<Task>(task),
        std::forward<Policy>(policy)
    };
}

}  // namespace yorch
