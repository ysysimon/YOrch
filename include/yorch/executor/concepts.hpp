#pragma once

#include <concepts> // IWYU pragma: keep
#include <type_traits> // IWYU pragma: keep
#include <utility>

#include "../context.hpp"
#include "../slots/result_out.hpp"
#include "../task_adapters/concepts.hpp" // IWYU pragma: keep

namespace yorch {

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

template <typename Task, typename Ctx, typename Prev = no_prev>
concept executable_direct_output_task =
    direct_output_task<Task, Ctx, Prev> &&
    requires(
        Task&& task,
        exec_context<Ctx, Prev>& ec,
        detail::typed_slot<detail::declared_task_output_t<Task>>& slot) {
        {
            std::forward<Task>(task).invoke_into(
                ec,
                result_out<detail::declared_task_output_t<Task>> {slot})
        } -> std::same_as<step_result>;
        requires noexcept(std::forward<Task>(task).invoke_into(
            ec,
            result_out<detail::declared_task_output_t<Task>> {slot}));
    };

}  // namespace yorch
