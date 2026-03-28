#pragma once
#include <functional>
#include <type_traits>
#include <utility>

#include "context.hpp"
#include "result.hpp"

namespace yorch {

namespace detail {

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

template <typename R>
inline constexpr bool default_catch_supported_v =
    std::is_void_v<R> ||
    std::is_same_v<R, step_result> ||
    std::is_same_v<R, task_result<void>>;

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

template <typename Policy>
using policy_result_t = std::invoke_result_t<Policy&>;

template <typename R, typename Policy>
inline constexpr bool catch_policy_supported_v =
    std::is_nothrow_invocable_v<Policy&> &&
    (std::is_void_v<R>
         ? (std::is_same_v<policy_result_t<Policy>, step_result> ||
            std::is_same_v<policy_result_t<Policy>, task_result<void>>)
         : (!std::is_reference_v<R> &&
            std::is_convertible_v<policy_result_t<Policy>, R>));

template <typename R, typename Policy>
[[nodiscard]] constexpr auto policy_catch_failure_result(Policy& policy) noexcept {
    static_assert(catch_policy_supported_v<R, Policy>,
                  "catch_as_failure(task, policy) requires a noexcept fallback returning a compatible result");

    if constexpr (std::is_void_v<R>) {
        return std::invoke(policy);
    } else {
        return static_cast<R>(std::invoke(policy));
    }
}

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

template <typename Task, typename Ctx, typename Prev = no_prev>
concept executable_task =
    requires(Task&& task, exec_context<Ctx, Prev>& ec) {
        std::forward<Task>(task).invoke_raw(ec);
    } &&
    requires(Task&& task, exec_context<Ctx, Prev>& ec) {
        requires noexcept(std::forward<Task>(task).invoke_raw(ec));
    };

template <typename Task>
struct catch_failure_task {
    Task task;

    template <typename Ctx, typename Prev>
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

template <typename Task, typename Policy>
struct catch_failure_with_policy_task {
    Task task;
    Policy policy;

    template <typename Ctx, typename Prev>
    constexpr auto invoke_raw(exec_context<Ctx, Prev>& ec) noexcept {
        using raw_result_t = detail::raw_task_result_t<Task&, Ctx, Prev>;

        static_assert(detail::catch_policy_supported_v<raw_result_t, Policy>,
                      "catch_as_failure(task, policy) requires a noexcept fallback returning a compatible result");

        try {
            if constexpr (std::is_void_v<raw_result_t>) {
                task.invoke_raw(ec);
                return detail::void_task_success_result<detail::policy_result_t<Policy>>();
            } else {
                return task.invoke_raw(ec);
            }
        } catch (...) {
            return detail::policy_catch_failure_result<raw_result_t>(policy);
        }
    }
};

template <typename Task>
constexpr auto catch_as_failure(Task&& task) {
    return catch_failure_task<std::decay_t<Task>>{
        std::forward<Task>(task)
    };
}

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
 * `invoke_raw(...)` protocol.
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
