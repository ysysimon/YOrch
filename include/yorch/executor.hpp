#pragma once
#include <type_traits>
#include <utility>

#include "task_adapters.hpp"

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
    detail::declared_task_raw_result_matches_invoke_raw_v<Task, Ctx, Prev> &&
    requires(Task&& task, exec_context<Ctx, Prev>& ec) {
        requires noexcept(std::forward<Task>(task).invoke_raw(ec));
    };

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

}  // namespace yorch
