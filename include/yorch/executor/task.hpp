#pragma once

#include <type_traits>
#include <utility>

#include "../detail/executor/result.hpp"
#include "../detail/task_adapters/traits.hpp"
#include "../slots/result_out.hpp"
#include "concepts.hpp" // IWYU pragma: keep

namespace yorch {

/**
 * @brief Executes a ready-to-run task against the provided execution context.
 *
 * In the current design, `run_task(...)` remains thin: it resolves no
 * arguments by itself and only normalizes the raw return emitted by the task's
 * `invoke_raw(...)` protocol. Direct-output tasks that write into a slot-like
 * sink should instead use `run_task_into(...)`.
 *
 * The function intentionally accepts only the no-throw execution surface
 * modeled by `executable_task`; potentially throwing tasks must first be
 * wrapped into a no-throw adapter such as `catch_as_failure(...)`.
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

    static_assert(detail::declared_task_raw_result_matches_invoke_raw_v<Task, Ctx, Prev>,
                  "Task raw_result_type must match invoke_raw(exec_context<...>&) return type");

    if constexpr (std::is_void_v<raw_result_t>) {
        std::forward<Task>(task).invoke_raw(ec);
        return step_result::success();
    } else {
        return detail::normalize_task_result(
            std::forward<Task>(task).invoke_raw(ec)
        );
    }
}

/**
 * @brief Executes a direct-output task against the provided execution context.
 *
 * Unlike `run_task(...)`, this overload is for tasks whose payload is written
 * into a caller-provided output sink instead of returned through `invoke_raw(...)`.
 * The caller owns the destination slot lifetime; this helper only enforces the
 * direct-output success/non-success contract on top of that sink.
 *
 * @tparam Task Direct-output task type exposing `invoke_into(...)`.
 * @tparam Ctx Borrowed execution-context schema.
 * @tparam Prev Direct-parent slot view type.
 * @param task Task object to execute.
 * @param ec Borrowed execution context.
 * @param out Output sink receiving the task payload on success.
 * @return The resulting `step_result` from the direct-output execution.
 */
template <typename Task, typename Ctx, typename Prev = no_prev>
    requires executable_direct_output_task<Task, Ctx, Prev>
[[nodiscard]] constexpr step_result run_task_into(
    Task&& task,
    exec_context<Ctx, Prev>& ec,
    result_out<detail::declared_task_output_t<Task>> out)
    noexcept(noexcept(std::forward<Task>(task).invoke_into(ec, out)))
{
    const auto step = std::forward<Task>(task).invoke_into(ec, out);

    if (step.ok()) {
        YORCH_ASSERT(out.has_value());
    } else if (out.has_value()) {
        out.destroy();
    }

    return step;
}

} // namespace yorch
