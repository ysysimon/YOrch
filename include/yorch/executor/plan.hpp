#pragma once

#include "../detail/executor/prev_validation.hpp"
#include "../detail/executor/serial_dfs.hpp"
#include "../slots.hpp"

namespace yorch {

/**
 * @brief Executes a compiled plan using serial depth-first traversal.
 *
 * `abort_branch` is consumed locally by the parent and only terminates the
 * current child subtree; all other non-success statuses stop the remaining DFS
 * and bubble to the caller.
 *
 * The call stack shape is:
 *
 * - `run_plan(...)` creates per-run slot storage and enters the root
 * - `run_node<I>(...)` prepares node `I`'s direct-parent view and
 *   `exec_context`
 * - `run_node_body<I>(...)` executes node `I`, stores its payload if any, and
 *   delegates to child traversal
 * - `run_children<I, Ord>(...)` enumerates node `I`'s children and descends
 *   into each child via `run_node<child>(...)`
 *
 * This separation keeps node-local execution, context assembly, and sibling
 * traversal as distinct steps in the DFS executor.
 *
 * @tparam Plan Compiled plan type.
 * @param plan Mutable plan whose stored task objects are executed in DFS order.
 * @return Final `step_result` of the whole execution.
 */
template <typename LayoutPolicy = slot_layout_one_to_one_policy, typename Plan>
    requires detail::slot_layout_policy<LayoutPolicy> &&
             (Plan::node_count > 0) &&
             detail::plan_prev_source_valid_v<Plan> &&
             detail::plan_fanout_prev_valid_v<Plan>
[[nodiscard]] constexpr step_result run_plan(Plan& plan) {
    plan_exec_slots<Plan, LayoutPolicy> slots;
    return detail::run_node<0>(plan, slots);
}

/**
 * @brief Executes a compiled plan using serial depth-first traversal with an
 * external typed execution context.
 *
 * @tparam Plan Compiled plan type.
 * @tparam Ctx Borrowed context type.
 * @param plan Mutable plan whose stored task objects are executed in DFS order.
 * @param ctx Borrowed context exposed to each node through `from_ctx(...)`.
 * @return Final `step_result` of the whole execution.
 */
template <typename LayoutPolicy = slot_layout_one_to_one_policy, typename Plan, typename Ctx>
    requires detail::slot_layout_policy<LayoutPolicy> &&
             (Plan::node_count > 0) &&
             detail::plan_prev_source_valid_v<Plan> &&
             detail::plan_fanout_prev_valid_v<Plan>
[[nodiscard]] constexpr step_result run_plan(Plan& plan, Ctx& ctx) {
    plan_exec_slots<Plan, LayoutPolicy> slots;
    return detail::run_node<0>(plan, slots, ctx);
}

} // namespace yorch
