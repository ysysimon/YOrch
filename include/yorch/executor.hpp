#pragma once
#include <exception>
#include <functional>
#include <type_traits>
#include <utility>

#include "context.hpp"
#include "result.hpp"

namespace yorch {

namespace detail {

/**
 * @brief Normalizes a raw task return into scheduler-facing `step_result`.
 *
 * Plain value payloads are treated as successful completion, `step_result`
 * passes through unchanged, and `task_result<T>` contributes only its control
 * status here. Payload extraction itself is handled by later execution stages.
 *
 * @tparam R Raw return object type.
 * @param r Raw return object emitted by `invoke_raw(...)`.
 * @return Normalized `step_result`.
 */
template <typename R>
[[nodiscard]] constexpr step_result normalize_task_result(R&& r) { // NOLINT(readability-identifier-length)
    using raw_t = std::remove_cvref_t<R>;

    if constexpr (std::is_same_v<raw_t, step_result>) {
        return std::forward<R>(r);
    } else if constexpr (is_task_result_v<raw_t>) {
        return std::forward<R>(r).step;
    } else {
        static_cast<void>(r);
        return step_result::success();
    }
}

template <typename Task, typename Ctx, typename Prev>
using raw_task_result_t =
    decltype(std::declval<Task>().invoke_raw(std::declval<exec_context<Ctx, Prev>&>()));

/**
 * @brief Reports whether the default `catch_as_failure(task)` adapter can
 * synthesize a failure result for raw return type `R`.
 *
 * The no-policy overload intentionally stays conservative: it only supports
 * return categories where the library can manufacture a failure result without
 * inventing a payload object.
 *
 * @tparam R Raw task return type.
 */
template <typename R>
inline constexpr bool default_catch_supported_v =
    std::is_void_v<R> ||
    std::is_same_v<R, step_result> ||
    std::is_same_v<R, task_result<void>>;

/**
 * @brief Produces the default failure result used by `catch_as_failure(task)`.
 *
 * This helper is valid only for raw result categories accepted by
 * `default_catch_supported_v`.
 *
 * @tparam R Raw task return type.
 * @return A failure result compatible with `R`.
 */
template <typename R>
[[nodiscard]] constexpr auto default_catch_failure_result() noexcept {
    static_assert(default_catch_supported_v<R>,
                  "Default catch_as_failure only supports void, step_result, and task_result<void>");

    if constexpr (std::is_void_v<R> || std::is_same_v<R, step_result>) {
        return step_result::failure();
    } else {
        return task_result<void>{step_result::failure()};
    }
}

/**
 * @brief Extracts the result type returned by an exception fallback policy.
 *
 * The policy contract is `policy(std::exception_ptr) noexcept`.
 *
 * @tparam Policy Policy type stored inside `catch_as_failure(task, policy)`.
 * @tparam Enable SFINAE hook used to detect whether the policy is invocable.
 */
template <typename Policy, typename = void>
struct policy_result;

template <typename Policy>
struct policy_result<Policy, std::void_t<std::invoke_result_t<Policy&, std::exception_ptr>>> {
    using type = std::invoke_result_t<Policy&, std::exception_ptr>;
};

template <typename Policy>
using policy_result_t = typename policy_result<Policy>::type;

/**
 * @brief Trait that reports whether a fallback policy is compatible with raw
 * task result type `R`.
 *
 * Compatibility means:
 * - the policy is invocable as `policy(std::exception_ptr)` and is `noexcept`
 * - for `void` tasks, the policy returns `step_result` or `task_result<void>`
 * - for non-void tasks, the policy result is convertible to the raw result
 *
 * @tparam R Raw task return type.
 * @tparam Policy Policy type.
 * @tparam Enable SFINAE hook used to suppress diagnostics for non-invocable
 * policies.
 */
template <typename R, typename Policy, typename = void>
struct catch_policy_supported : std::false_type {};

template <typename R, typename Policy>
struct catch_policy_supported<R, Policy, std::void_t<std::invoke_result_t<Policy&, std::exception_ptr>>>
    : std::bool_constant<
          std::is_nothrow_invocable_v<Policy&, std::exception_ptr> &&
          (std::is_void_v<R>
               ? (std::is_same_v<policy_result_t<Policy>, step_result> ||
                  std::is_same_v<policy_result_t<Policy>, task_result<void>>)
               : (!std::is_reference_v<R> &&
                  std::is_convertible_v<policy_result_t<Policy>, R>))> {};

template <typename R, typename Policy>
inline constexpr bool catch_policy_supported_v =
    catch_policy_supported<R, Policy>::value;

/**
 * @brief Invokes a fallback policy to produce the exception-path raw result.
 *
 * @tparam R Raw task return type of the wrapped task.
 * @tparam Policy Policy type.
 * @param policy Stored fallback policy.
 * @param ep Captured exception from the wrapped task.
 * @return A raw result compatible with `R`.
 */
template <typename R, typename Policy>
[[nodiscard]] inline auto policy_catch_failure_result(Policy& policy, std::exception_ptr ep) noexcept {
    static_assert(catch_policy_supported_v<R, Policy>,
                  "catch_as_failure(task, policy) requires a noexcept fallback taking std::exception_ptr and returning a compatible result");

    if constexpr (std::is_void_v<R>) {
        return std::invoke(policy, ep);
    } else {
        return static_cast<R>(std::invoke(policy, ep));
    }
}

/**
 * @brief Produces the success-path raw result for a wrapped `void` task when a
 * custom fallback policy is installed.
 *
 * The policy decides whether exception paths produce `step_result` or
 * `task_result<void>`, so successful `void` execution must mirror that same raw
 * result shape.
 *
 * @tparam PolicyResult Raw result type returned by the policy.
 * @return Success result matching `PolicyResult`.
 */
template <typename PolicyResult>
[[nodiscard]] constexpr auto void_task_success_result() noexcept {
    static_assert(std::is_same_v<PolicyResult, step_result> ||
                  std::is_same_v<PolicyResult, task_result<void>>,
                  "Void task catch policy must return step_result or task_result<void>");

    if constexpr (std::is_same_v<PolicyResult, step_result>) {
        return step_result::success();
    } else {
        return task_result<void>{step_result::success()};
    }
}

}  // namespace detail

/**
 * @brief Describes the main execution protocol accepted by `run_task(...)`.
 *
 * A task is executable on the main path only when it exposes
 * `invoke_raw(exec_context<...>&)` and that call is itself `noexcept`.
 * Throwing tasks are intentionally excluded from this concept; they must be
 * adapted first, for example with `catch_as_failure(...)`, before they can be
 * handed to the executor.
 *
 * @tparam Task Task object type.
 * @tparam Ctx Context schema.
 * @tparam Prev Direct-parent slot view type.
 */
template <typename Task, typename Ctx, typename Prev = no_prev>
concept executable_task =
    requires(Task&& task, exec_context<Ctx, Prev>& ec) {
        std::forward<Task>(task).invoke_raw(ec);
    } &&
    requires(Task&& task, exec_context<Ctx, Prev>& ec) {
        requires noexcept(std::forward<Task>(task).invoke_raw(ec));
    };

/**
 * @brief Reports whether a task object exposes the raw execution protocol for a
 * concrete `exec_context`.
 *
 * Unlike `executable_task`, this concept only checks structural invocability
 * and does not require the wrapped task itself to be `noexcept`. It is used as
 * the lower-level boundary for exception-catching adapters.
 *
 * @tparam Task Wrapped task type.
 * @tparam Ctx Context schema.
 * @tparam Prev Direct-parent slot view type.
 */
template <typename Task, typename Ctx, typename Prev = no_prev>
concept catch_wrappable_task =
    requires(Task& task, exec_context<Ctx, Prev>& ec) {
        task.invoke_raw(ec);
    };

/**
 * @brief Reports whether `catch_as_failure(task)` can wrap `Task` for a
 * concrete execution context without a custom policy.
 *
 * @tparam Task Wrapped task type.
 * @tparam Ctx Context schema.
 * @tparam Prev Direct-parent slot view type.
 */
template <typename Task, typename Ctx, typename Prev = no_prev>
concept default_catchable_task =
    catch_wrappable_task<Task, Ctx, Prev> &&
    detail::default_catch_supported_v<detail::raw_task_result_t<Task&, Ctx, Prev>>;

/**
 * @brief Reports whether `catch_as_failure(task, policy)` can wrap `Task`
 * using the provided fallback `Policy`.
 *
 * @tparam Task Wrapped task type.
 * @tparam Policy Fallback policy type.
 * @tparam Ctx Context schema.
 * @tparam Prev Direct-parent slot view type.
 */
template <typename Task, typename Policy, typename Ctx, typename Prev = no_prev>
concept catch_policy_compatible_task =
    catch_wrappable_task<Task, Ctx, Prev> &&
    detail::catch_policy_supported_v<detail::raw_task_result_t<Task&, Ctx, Prev>, Policy>;

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
struct catch_failure_task {
    Task task;

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

        static_assert(detail::default_catch_supported_v<raw_result_t>,
                      "Default catch_as_failure only supports tasks returning void, step_result, or task_result<void>");

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
};

/**
 * @brief Exception-catching adapter for tasks with a user-supplied fallback
 * policy.
 *
 * The policy is invoked with `std::exception_ptr` and must itself be
 * `noexcept`, allowing the wrapper to present a `noexcept invoke_raw(...)`
 * surface to the executor while still giving users control over failure-path
 * raw results.
 *
 * @tparam Task Stored task type.
 * @tparam Policy Stored fallback policy type.
 */
template <typename Task, typename Policy>
struct catch_failure_with_policy_task {
    Task task;
    Policy policy;

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
};

/**
 * @brief Wraps a task so thrown exceptions become default failure results.
 *
 * This overload is intentionally narrow: it only supports raw return
 * categories for which the library can synthesize a failure result without a
 * user-provided payload.
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
 * The policy contract is `policy(std::exception_ptr) noexcept`, and its return
 * type must be compatible with the wrapped task's raw return category.
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

/**
 * @brief Executes a ready-to-run task against the provided execution context.
 *
 * In the current design, `run_task(...)` remains thin: it resolves no
 * arguments by itself and only normalizes the raw return emitted by the task's
 * `invoke_raw(...)` protocol. The function intentionally accepts only the
 * no-throw execution surface modeled by `executable_task`; potentially
 * throwing tasks must first be wrapped into a no-throw adapter such as
 * `catch_as_failure(...)`.
 *
 * @tparam Task Executable task type, typically `bound_task`.
 * @tparam Ctx Borrowed execution-context schema.
 * @param task Task object to execute.
 * @param ec Borrowed execution context.
 * @return The normalized `step_result` derived from the raw task return.
 */
template <typename Task, typename Ctx, typename Prev = no_prev>
    requires executable_task<Task, Ctx, Prev>
[[nodiscard]] constexpr step_result run_task(Task&& task, exec_context<Ctx, Prev>& ec)
    noexcept(
        noexcept(std::forward<Task>(task).invoke_raw(ec)) &&
        (std::is_void_v<detail::raw_task_result_t<Task&&, Ctx, Prev>> ||
         noexcept(detail::normalize_task_result(
             std::forward<Task>(task).invoke_raw(ec)))))
{
    using raw_result_t = detail::raw_task_result_t<Task&&, Ctx, Prev>;

    if constexpr (std::is_void_v<raw_result_t>) {
        std::forward<Task>(task).invoke_raw(ec);
        return step_result::success();
    } else {
        return detail::normalize_task_result(
            std::forward<Task>(task).invoke_raw(ec)
        );
    }
}

}  // namespace yorch
