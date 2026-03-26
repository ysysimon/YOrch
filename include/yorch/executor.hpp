#pragma once

#include <utility>

#include "context.hpp"
#include "result.hpp"

namespace yorch {

template <typename Task, typename Ctx, typename Prev = no_prev>
concept executable_task =
    requires(Task&& task, exec_context<Ctx, Prev>& ec) {
        { std::forward<Task>(task)(ec) } -> std::same_as<step_result>;
    };

/**
 * @brief Executes a ready-to-run task against the provided execution context.
 *
 * In the current design, `run_task(...)` is intentionally thin: it only
 * invokes a task object that already matches the execution protocol and returns
 * `step_result`. Argument binding and user-return normalization stay in
 * `bind.hpp`.
 *
 * @tparam Task Executable task type, typically `bound_task`.
 * @tparam Ctx Borrowed execution-context schema.
 * @param task Task object to execute.
 * @param ec Borrowed execution context.
 * @return The task's normalized `step_result`.
 */
template <typename Task, typename Ctx, typename Prev = no_prev>
    requires executable_task<Task, Ctx, Prev>
[[nodiscard]] constexpr step_result run_task(Task&& task, exec_context<Ctx, Prev>& ec)
    noexcept(noexcept(std::forward<Task>(task)(ec))) {
    return std::forward<Task>(task)(ec);
}

}  // namespace yorch
