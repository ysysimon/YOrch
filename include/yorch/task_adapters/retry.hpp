#pragma once

#include "concepts.hpp" // IWYU pragma: keep
#include "detail/result_helpers.hpp"

namespace yorch {

/**
 * @brief Retry policy that allows a fixed number of additional retries.
 *
 * `max_retries` counts retries after the initial attempt. For example,
 * `retry_fixed_policy {2}` means "run once, then allow up to two more attempts
 * if the task keeps returning `retry`". When that retry budget is exhausted,
 * the adapter converts the final retry result into `failure`.
 */
struct retry_fixed_policy {
    std::size_t max_retries = 0;

    [[nodiscard]] constexpr bool should_retry(std::size_t retry_count) const noexcept {
        return retry_count < max_retries;
    }

    template <typename Raw>
    [[nodiscard]] static constexpr auto on_exhausted(Raw&& raw) noexcept {
        return detail::retry_exhausted_as_failure(std::forward<Raw>(raw));
    }
};

/**
 * @brief Retry policy that allows a fixed number of retries and then preserves
 * the final `retry` result unchanged.
 */
struct retry_fixed_passthrough_policy {
    std::size_t max_retries = 0;

    [[nodiscard]] constexpr bool should_retry(std::size_t retry_count) const noexcept {
        return retry_count < max_retries;
    }

    template <typename Raw>
    [[nodiscard]] static constexpr std::remove_cvref_t<Raw> on_exhausted(Raw&& raw) noexcept {
        return std::forward<Raw>(raw);
    }
};

/**
 * @brief Retry policy that keeps retrying for as long as the task requests it.
 */
struct retry_forever_policy {
    [[nodiscard]] static constexpr bool should_retry(std::size_t) noexcept {
        return true;
    }
};

/**
 * @brief Retry adapter that re-invokes a task when it returns `retry`.
 *
 * This wrapper leaves the task's raw result type unchanged. Only results that
 * can explicitly carry `step_status::retry` participate in the retry loop:
 * `step_result` and `task_result<T>`. Plain value and `void` results are
 * forwarded unchanged because they cannot request retry through the current
 * status model.
 *
 * @tparam Task Stored task type.
 * @tparam Policy Stored retry policy type.
 */
template <typename Task, typename Policy>
struct retry_task
    : detail::retry_task_raw_result_base<Task>,
      detail::forwarded_task_output_base<Task> {
    Task task;
    Policy policy;

    constexpr retry_task(Task&& stored_task, Policy&& stored_policy)
        noexcept(std::is_nothrow_move_constructible_v<Task> &&
                 std::is_nothrow_move_constructible_v<Policy>)
        : task(std::move(stored_task)),
          policy(std::move(stored_policy)) {}

    constexpr retry_task(const Task& stored_task, const Policy& stored_policy)
        noexcept(std::is_nothrow_copy_constructible_v<Task> &&
                 std::is_nothrow_copy_constructible_v<Policy>)
        : task(stored_task),
          policy(stored_policy) {}

    /**
     * @brief Executes the wrapped task and re-runs it while policy permits.
     *
     * Each retry attempt re-enters the wrapped task from scratch with the same
     * borrowed execution context. Side effects from earlier attempts are not
     * rolled back by this adapter.
     *
     * @tparam Ctx Context schema.
     * @tparam Prev Direct-parent slot view type.
     * @param ec Borrowed execution context.
     * @return The first non-`retry` raw result, or the policy-selected result
     * once the retry budget is exhausted.
     */
    template <typename Ctx, typename Prev>
        requires adapter_wrappable_task<Task, Ctx, Prev> && retry_policy<Policy>
    constexpr decltype(auto) invoke_raw(exec_context<Ctx, Prev>& ec)
        noexcept(noexcept(task.invoke_raw(ec)) &&
                 noexcept(policy.should_retry(std::size_t {})) &&
                 noexcept(detail::handle_retry_exhausted(
                     policy,
                     std::declval<std::remove_cvref_t<detail::raw_task_result_t<Task&, Ctx, Prev>>>()))) {
        using raw_result_t = detail::raw_task_result_t<Task&, Ctx, Prev>;
        using raw_value_t = std::remove_cvref_t<raw_result_t>;

        static_assert(detail::declared_task_raw_result_matches_invoke_raw_v<Task, Ctx, Prev>,
                      "Wrapped task raw_result_type must match invoke_raw(exec_context<...>&) return type");

        if constexpr (!detail::retry_status_capable_result_v<raw_result_t>) {
            return task.invoke_raw(ec);
        } else {
            static_assert(!std::is_reference_v<raw_result_t>,
                          "with_retry(...) does not support retry-capable reference raw results");

            std::size_t retry_count = 0;

            while (true) {
                raw_value_t raw = task.invoke_raw(ec);

                if (!detail::raw_result_requests_retry(raw)) {
                    return raw;
                }

                if (!policy.should_retry(retry_count)) {
                    return detail::handle_retry_exhausted(policy, std::move(raw));
                }

                ++retry_count;
            }
        }
    }

    template <typename Ctx, typename Prev, typename U = Task>
        requires direct_output_task<U, Ctx, Prev> && retry_policy<Policy>
    constexpr step_result invoke_into(
        exec_context<Ctx, Prev>& ec,
        result_out<detail::declared_task_output_t<U>> out) noexcept(
            noexcept(task.invoke_into(ec, out)) &&
            noexcept(policy.should_retry(std::size_t {})) &&
            noexcept(detail::handle_retry_exhausted(
                policy,
                step_result::retry()))) {
        std::size_t retry_count = 0;

        while (true) {
            const auto step = task.invoke_into(ec, out);

            if (step.status != step_status::retry) {
                return step;
            }

            if (!policy.should_retry(retry_count)) {
                return detail::handle_retry_exhausted(policy, step);
            }

            // if the task requested retry, it should not have produced an output value, but just in case, destroy it before the next attempt to avoid leaking resources or causing double-destruction on the next success
            if (out.has_value()) {
                out.destroy();
            }

            ++retry_count;
        }
    }
};

/**
 * @brief Wraps a task so `retry` results are handled by a user retry policy.
 *
 * The wrapped task keeps its original raw result type. The adapter simply
 * re-invokes it while the raw result requests retry and the policy approves
 * another attempt.
 *
 * @tparam Task Task type to wrap.
 * @tparam Policy Retry policy type.
 * @param task Task object exposing `invoke_raw(...)`.
 * @param policy Retry policy object.
 * @return A `retry_task` wrapper.
 */
template <typename Task, typename Policy>
    requires retry_policy<std::decay_t<Policy>>
constexpr auto with_retry(Task&& task, Policy&& policy) {
    return retry_task<
        std::decay_t<Task>,
        std::decay_t<Policy>
    >{
        std::forward<Task>(task),
        std::forward<Policy>(policy)
    };
}

}  // namespace yorch
