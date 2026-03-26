#pragma once
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
    } else if constexpr (std::is_same_v<raw_t, bool>) {
        return r ? step_result::success()
                 : step_result::failure();
    } else if constexpr (is_task_result_v<raw_t>) {
        return std::forward<R>(r).step;
    } else {
        static_assert(std::is_same_v<raw_t, void>,
                      "Unsupported task return type");
    }
}

template <typename Task, typename Ctx, typename Prev>
using raw_task_result_t =
    decltype(std::declval<Task>().invoke_raw(std::declval<exec_context<Ctx, Prev>&>()));

}  // namespace detail

template <typename Task, typename Ctx, typename Prev = no_prev>
concept executable_task =
    requires(Task&& task, exec_context<Ctx, Prev>& ec) {
        std::forward<Task>(task).invoke_raw(ec);
    };

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
