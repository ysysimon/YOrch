#pragma once

#include <exception>
#include <functional>
#include <type_traits>
#include <utility>

#include "context.hpp"
#include "result.hpp"

namespace yorch::detail {

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
        return task_result<void>::failure();
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

template <typename Task, typename = void>
struct declared_task_raw_result;

template <typename Task>
struct declared_task_raw_result<
    Task,
    std::void_t<typename std::remove_cvref_t<Task>::raw_result_type>
> {
    using type = typename std::remove_cvref_t<Task>::raw_result_type;
};

template <typename Task>
using declared_task_raw_result_t = typename declared_task_raw_result<Task>::type;

/**
 * @brief Reports whether a task's declared `raw_result_type` agrees with the
 * concrete return type of `invoke_raw(exec_context<Ctx, Prev>&)`.
 *
 * This trait is intentionally permissive for tasks that do not declare
 * `raw_result_type`: in that case it yields `true`, so runtime-only task
 * shapes can still participate in lower-level execution checks. When a task
 * does opt into the declared raw-result contract, however, this trait enforces
 * that the declaration remains consistent with the actual `invoke_raw(...)`
 * expression type observed for the concrete execution context.
 *
 * The executor and task adapters use this as a protocol guard so plan-facing
 * metadata and execution-time behavior cannot silently drift apart.
 *
 * @tparam Task Task type being checked.
 * @tparam Ctx Context schema for the concrete invocation.
 * @tparam Prev Direct-parent slot view type for the concrete invocation.
 * @tparam Enable SFINAE hook that activates strict checking only when
 * `Task::raw_result_type` is declared.
 */
template <typename Task, typename Ctx, typename Prev, typename = void>
struct declared_task_raw_result_matches_invoke_raw : std::true_type {};

template <typename Task, typename Ctx, typename Prev>
struct declared_task_raw_result_matches_invoke_raw<
    Task,
    Ctx,
    Prev,
    std::void_t<declared_task_raw_result_t<Task>>
> : std::bool_constant<
        std::same_as<
            raw_task_result_t<Task&, Ctx, Prev>,
            declared_task_raw_result_t<Task>>> {};

template <typename Task, typename Ctx, typename Prev>
inline constexpr bool declared_task_raw_result_matches_invoke_raw_v =
    declared_task_raw_result_matches_invoke_raw<Task, Ctx, Prev>::value;

template <typename Task, typename = void>
struct catch_failure_task_raw_result_base {};

template <typename Task>
struct catch_failure_task_raw_result_base<
    Task,
    std::void_t<declared_task_raw_result_t<Task>>
> {
    using inner_type = declared_task_raw_result_t<Task>;
    using raw_result_type = std::conditional_t<std::is_void_v<inner_type>, step_result, inner_type>;
};

template <typename Task, typename Policy, typename = void>
struct catch_failure_with_policy_task_raw_result_base {};

template <typename Task, typename Policy>
struct catch_failure_with_policy_task_raw_result_base<
    Task,
    Policy,
    std::void_t<
        declared_task_raw_result_t<Task>,
        policy_result_t<std::remove_cvref_t<Policy>>
    >
> {
    using inner_type = declared_task_raw_result_t<Task>;
    using policy_type = policy_result_t<std::remove_cvref_t<Policy>>;
    using raw_result_type = std::conditional_t<std::is_void_v<inner_type>, policy_type, inner_type>;
};

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
        return task_result<void>::success();
    }
}

}  // namespace yorch::detail

namespace yorch {

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
    } &&
    detail::declared_task_raw_result_matches_invoke_raw_v<Task, Ctx, Prev>;

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
struct catch_failure_task : detail::catch_failure_task_raw_result_base<Task> {
    Task task;

    constexpr explicit catch_failure_task(Task&& stored_task)
        noexcept(std::is_nothrow_move_constructible_v<Task>)
        : task(std::forward<Task>(stored_task)) {}

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
struct catch_failure_with_policy_task
    : detail::catch_failure_with_policy_task_raw_result_base<Task, Policy> {
    Task task;
    Policy policy;

    constexpr catch_failure_with_policy_task(Task&& stored_task, Policy&& stored_policy)
        noexcept(std::is_nothrow_move_constructible_v<Task> &&
                 std::is_nothrow_move_constructible_v<Policy>)
        : task(std::forward<Task>(stored_task)),
          policy(std::forward<Policy>(stored_policy)) {}

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
 * The policy contract is `policy(std::exception_ptr) noexcept`, and its return
 * type must be compatible with the wrapped task's raw return category.
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
