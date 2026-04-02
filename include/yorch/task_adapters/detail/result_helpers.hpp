#pragma once

#include <concepts> // IWYU pragma: keep
#include <functional>
#include <utility>

#include "traits.hpp"

namespace yorch::detail {

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
 * @brief Invokes a fallback policy to produce the exception-path raw result.
 *
 * @tparam R Raw task return type of the wrapped task.
 * @tparam Policy Policy type.
 * @param policy Stored fallback policy.
 * @param ep Captured exception from the wrapped task.
 * @return A raw result compatible with `R`.
 */
template <typename R, typename Policy>
[[nodiscard]] inline auto policy_catch_failure_result(
    Policy& policy,
    const std::exception_ptr& ep) noexcept {
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

template <typename R>
[[nodiscard]] constexpr bool raw_result_requests_retry(const R& r) noexcept {
    using raw_t = std::remove_cvref_t<R>;

    if constexpr (std::is_same_v<raw_t, step_result>) {
        return r.status == step_status::retry;
    } else if constexpr (is_task_result_v<raw_t>) {
        return r.step.status == step_status::retry;
    } else {
        static_cast<void>(r);
        return false;
    }
}

template <typename R>
inline constexpr bool retry_status_capable_result_v =
    std::is_same_v<std::remove_cvref_t<R>, step_result> ||
    is_task_result_v<std::remove_cvref_t<R>>;

template <typename Raw>
[[nodiscard]] constexpr auto retry_exhausted_as_failure(const Raw& raw) noexcept {
    using raw_t = std::remove_cvref_t<Raw>;

    static_assert(retry_status_capable_result_v<raw_t>,
                  "Retry exhaustion fallback is only valid for results that carry step_status");

    static_cast<void>(raw);

    if constexpr (std::is_same_v<raw_t, step_result>) {
        return step_result::failure();
    } else if constexpr (std::is_same_v<raw_t, task_result<void>>) {
        return task_result<void>::failure();
    } else {
        return raw_t::failure();
    }
}

template <typename Policy, typename Raw>
concept retry_exhaustion_policy =
    requires(const std::remove_cvref_t<Policy>& policy, Raw&& raw) {
        { policy.on_exhausted(std::forward<Raw>(raw)) } noexcept
            -> std::convertible_to<std::remove_cvref_t<Raw>>;
    };

template <typename Policy, typename Raw>
[[nodiscard]] constexpr auto handle_retry_exhausted(const Policy& policy, Raw&& raw) noexcept {
    if constexpr (retry_exhaustion_policy<Policy, Raw>) {
        return policy.on_exhausted(std::forward<Raw>(raw));
    } else {
        return std::forward<Raw>(raw);
    }
}

}  // namespace yorch::detail
