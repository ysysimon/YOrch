#pragma once

#include "detail/traits.hpp"
#include "../slots.hpp"

namespace yorch {

/**
 * @brief Reports whether a task object exposes the raw execution protocol for a
 * concrete `exec_context`.
 *
 * Unlike `executable_task`, this concept only checks structural invocability
 * and does not require the wrapped task itself to be `noexcept`. It is used as
 * the lower-level boundary shared by task adapters.
 *
 * @tparam Task Wrapped task type.
 * @tparam Ctx Context schema.
 * @tparam Prev Direct-parent slot view type.
 */
template <typename Task, typename Ctx, typename Prev = no_prev>
concept adapter_wrappable_task =
    requires(Task& task, exec_context<Ctx, Prev>& ec) {
        task.invoke_raw(ec);
    } &&
    detail::declared_task_raw_result_matches_invoke_raw_v<Task, Ctx, Prev>;

template <typename Task, typename Ctx, typename Prev = no_prev>
concept direct_output_task =
    detail::has_declared_task_output_v<Task> &&
    requires(Task& task, exec_context<Ctx, Prev>& ec) {
        {
            task.invoke_into(
                ec,
                result_out<detail::declared_task_output_t<Task>> {
                    std::declval<detail::typed_slot<detail::declared_task_output_t<Task>>&>()})
        } -> std::same_as<step_result>;
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
    adapter_wrappable_task<Task, Ctx, Prev> &&
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
    adapter_wrappable_task<Task, Ctx, Prev> &&
    detail::catch_policy_supported_v<detail::raw_task_result_t<Task&, Ctx, Prev>, Policy>;

template <typename Task, typename Policy, typename Ctx, typename Prev = no_prev>
concept catch_policy_compatible_direct_output_task =
    direct_output_task<Task, Ctx, Prev> &&
    std::is_nothrow_invocable_r_v<step_result, Policy&, const std::exception_ptr&>;

/**
 * @brief Reports whether a retry policy exposes the minimal protocol required
 * by `with_retry(...)`.
 *
 * The policy is asked whether one more retry should be attempted after the
 * wrapped task has already returned `step_status::retry`. The argument counts
 * how many retries have already been granted for the current execution of the
 * wrapped node.
 *
 * @tparam Policy Candidate retry policy type.
 */
template <typename Policy>
concept retry_policy =
    requires(const std::remove_cvref_t<Policy>& policy, std::size_t retry_count) {
        { policy.should_retry(retry_count) } noexcept -> std::convertible_to<bool>;
    };

}  // namespace yorch
