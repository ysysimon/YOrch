#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

#include "../../context.hpp"
#include "../../executor/concepts.hpp" // IWYU pragma: keep
#include "../../result.hpp"
#include "result.hpp"

namespace yorch::detail {

template <typename T>
struct exec_context_traits;

template <typename Ctx, typename Prev>
struct exec_context_traits<exec_context<Ctx, Prev>> {
    using ctx_type = Ctx;
    using prev_type = Prev;
};

template <typename Ctx>
[[nodiscard]] constexpr auto make_exec_context(Ctx& ctx, no_prev) noexcept
    -> exec_context<Ctx> {
    return {ctx};
}

template <typename Ctx, typename Prev>
[[nodiscard]] constexpr auto make_exec_context(Ctx& ctx, Prev prev) noexcept
    -> exec_context<Ctx, Prev> {
    return {ctx, prev};
}

[[nodiscard]] constexpr auto make_exec_context(no_prev) noexcept
    -> exec_context<void> {
    return {};
}

template <typename Prev>
[[nodiscard]] constexpr auto make_exec_context(Prev prev) noexcept
    -> exec_context<void, Prev> {
    return {prev};
}

template <typename Plan, std::size_t I, typename Slots>
[[nodiscard]] constexpr auto make_node_prev_view(Slots& slots) {
    if constexpr (Plan::template parent_index<I> == Plan::no_parent) {
        return no_prev {};
    } else {
        constexpr auto parent = Plan::template parent_index<I>;

        if constexpr (std::is_void_v<typename Plan::template output_type<parent>>) {
            return no_prev {};
        } else if constexpr (Plan::template child_count<parent> > 1) {
            return std::as_const(slots).template prev_view_for<parent>();
        } else {
            return slots.template prev_view_for<parent>();
        }
    }
}

template <typename Plan, typename Slots, std::size_t I>
using node_prev_view_t =
    decltype(make_node_prev_view<Plan, I>(std::declval<Slots&>()));

template <std::size_t I, std::size_t Ord = 0, typename Plan, typename Slots>
[[nodiscard]] constexpr step_result run_children(Plan& plan, Slots& slots);

template <std::size_t I, std::size_t Ord = 0, typename Plan, typename Slots, typename Ctx>
[[nodiscard]] constexpr step_result run_children(Plan& plan, Slots& slots, Ctx& ctx);

/**
 * @brief Executes node `I` itself after `run_node(...)` has already assembled
 * the correct execution context for that node.
 *
 * This function is the "node body" of the DFS executor:
 *
 * - it invokes the current task with the ready-to-use `exec_context`
 * - it extracts control-flow state from the raw return
 * - it materializes any successful payload into slot `I`
 * - it delegates child traversal to the injected `invoke_children` callback
 *
 * The callback is supplied by `run_node(...)` so this helper stays focused on
 * node-local execution rather than knowing how siblings are enumerated. The
 * slot guard keeps the current node's payload alive for the entire subtree and
 * destroys it automatically on every exit path, including early returns.
 *
 * @tparam I Node index being executed.
 * @tparam Plan Compiled plan type.
 * @tparam Slots Concrete slot storage type for the active plan run.
 * @tparam Ec Concrete `exec_context<...>` type prepared for node `I`.
 * @tparam ChildInvoker Callable that continues DFS into node `I`'s children.
 * @param plan Mutable compiled plan containing the node task object.
 * @param slots Per-run slot storage that owns subtree-visible payloads.
 * @param ec Borrowed execution context already tailored to node `I`.
 * @param invoke_children Continuation that traverses node `I`'s children.
 * @return The resulting `step_result` for node `I` and, when reached, its
 * child subtree.
 */
template <std::size_t I, typename Plan, typename Slots, typename Ec, typename ChildInvoker>
[[nodiscard]] constexpr step_result run_node_body(
    Plan& plan,
    Slots& slots,
    Ec& ec,
    // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    ChildInvoker&& invoke_children) {
    using task_t = typename Plan::template task_type<I>;
    using exec_traits_t = exec_context_traits<std::remove_cvref_t<Ec>>;
    using raw_result_t = typename Plan::template raw_result_type<I>;

    static_assert(
        executable_task<
            task_t&,
            typename exec_traits_t::ctx_type,
            typename exec_traits_t::prev_type> ||
        executable_direct_output_task<
            task_t&,
            typename exec_traits_t::ctx_type,
            typename exec_traits_t::prev_type>,
        "Plan nodes executed through run_plan(...) must expose a noexcept invoke_raw(exec_context<...>&) or invoke_into(exec_context<...>&, result_out<...>) surface");

    auto& task = plan.template entry<I>().task;
    node_slot_guard<Slots, I> guard {slots};

    if constexpr (executable_direct_output_task<
                      task_t&,
                      typename exec_traits_t::ctx_type,
                      typename exec_traits_t::prev_type>) {
        const auto step = finalize_direct_output_step<I>(
            slots,
            task.invoke_into(ec, slots.template out<I>()));

        if (!step.ok()) {
            return step;
        }

        guard.arm();
    } else if constexpr (std::is_void_v<raw_result_t>) {
        task.invoke_raw(ec);
    } else {
        const auto step = [&]() constexpr -> step_result {
            auto raw = task.invoke_raw(ec);
            const auto current = extract_step_result(raw);

            if (current.ok()) {
                if (store_node_output<I>(slots, raw)) {
                    guard.arm();
                }
            }

            return current;
        }();

        if (!step.ok()) {
            return step;
        }
    }

    return invoke_children();
}

/**
 * @brief Enters node `I` in the serial DFS executor without an external
 * context object.
 *
 * `run_node(...)` is the layer that bridges static plan structure into a
 * concrete `exec_context` for one node. It computes the direct-parent view for
 * node `I`, packages that view into an `exec_context`, and then hands control
 * to `run_node_body(...)`.
 *
 * Depth-first descent happens because the node body later invokes
 * `run_children<I, 0>(...)`, which selects a child and recursively calls that
 * child's own `run_node<child>(...)`.
 *
 * @tparam I Node index being entered.
 * @tparam Plan Compiled plan type.
 * @tparam Slots Concrete slot storage type for the active plan run.
 * @param plan Mutable compiled plan containing the node task object.
 * @param slots Per-run slot storage that owns subtree-visible payloads.
 * @return The resulting `step_result` for node `I` and its subtree.
 */
template <std::size_t I, typename Plan, typename Slots>
[[nodiscard]] constexpr step_result run_node(Plan& plan, Slots& slots) {
    auto prev = make_node_prev_view<Plan, I>(slots);
    auto ec = make_exec_context(prev);

    return run_node_body<I>(
        plan,
        slots,
        ec,
        [&]() constexpr -> step_result {
            return run_children<I>(plan, slots);
        });
}

/**
 * @brief Enters node `I` in the serial DFS executor with an external typed
 * context object.
 *
 * This overload performs the same DFS entry work as the context-free version,
 * but also threads the caller-provided execution context through the node's
 * `exec_context<Ctx, Prev>`.
 *
 * @tparam I Node index being entered.
 * @tparam Plan Compiled plan type.
 * @tparam Slots Concrete slot storage type for the active plan run.
 * @tparam Ctx Borrowed context type exposed through `from_ctx(...)`.
 * @param plan Mutable compiled plan containing the node task object.
 * @param slots Per-run slot storage that owns subtree-visible payloads.
 * @param ctx Borrowed typed context object shared by this plan execution.
 * @return The resulting `step_result` for node `I` and its subtree.
 */
template <std::size_t I, typename Plan, typename Slots, typename Ctx>
[[nodiscard]] constexpr step_result run_node(Plan& plan, Slots& slots, Ctx& ctx) {
    auto prev = make_node_prev_view<Plan, I>(slots);
    auto ec = make_exec_context(ctx, prev);

    return run_node_body<I>(
        plan,
        slots,
        ec,
        [&]() constexpr -> step_result {
            return run_children<I>(plan, slots, ctx);
        });
}

/**
 * @brief Executes the direct children of node `I` one by one in DFS order,
 * starting at child ordinal `Ord`, without an external context object.
 *
 * This helper is the sibling-enumeration half of the DFS recursion. It does
 * not itself "go deeper" by incrementing `Ord`; the actual descent into a
 * child subtree happens at `run_node<child>(...)`. Advancing to
 * `run_children<I, Ord + 1>(...)` means "the current child has finished, move
 * on to the next sibling of the same parent."
 *
 * From the parent's point of view, both `success` and `abort_branch` mean the
 * current child subtree has ended and sibling traversal may continue. Other
 * non-success states stop the remaining DFS and propagate upward unchanged.
 *
 * @tparam I Parent node index whose child list is being traversed.
 * @tparam Ord Current child ordinal within node `I`'s direct-child list.
 * @tparam Plan Compiled plan type.
 * @tparam Slots Concrete slot storage type for the active plan run.
 * @param plan Mutable compiled plan containing the node task objects.
 * @param slots Per-run slot storage that owns subtree-visible payloads.
 * @return The aggregate `step_result` for the remaining children of node `I`.
 */
template <std::size_t I, std::size_t Ord, typename Plan, typename Slots>
[[nodiscard]] constexpr step_result run_children(Plan& plan, Slots& slots) {
    static_cast<void>(plan);
    static_cast<void>(slots);

    if constexpr (Ord == Plan::template child_count<I>) {
        return step_result::success();
    } else {
        constexpr auto child = Plan::template child_index<I, Ord>;

        const auto child_step = run_node<child>(plan, slots);

        if (child_step.status == step_status::abort_branch) {
            return run_children<I, Ord + 1>(plan, slots);
        }

        if (!child_step.ok()) {
            return child_step;
        }

        return run_children<I, Ord + 1>(plan, slots);
    }
}

/**
 * @brief Executes the direct children of node `I` one by one in DFS order,
 * starting at child ordinal `Ord`, with an external typed context object.
 *
 * This overload is identical in traversal semantics to the context-free
 * version and exists only to thread `ctx` through recursive child entry.
 *
 * @tparam I Parent node index whose child list is being traversed.
 * @tparam Ord Current child ordinal within node `I`'s direct-child list.
 * @tparam Plan Compiled plan type.
 * @tparam Slots Concrete slot storage type for the active plan run.
 * @tparam Ctx Borrowed context type exposed through `from_ctx(...)`.
 * @param plan Mutable compiled plan containing the node task objects.
 * @param slots Per-run slot storage that owns subtree-visible payloads.
 * @param ctx Borrowed typed context object shared by this plan execution.
 * @return The aggregate `step_result` for the remaining children of node `I`.
 */
template <std::size_t I, std::size_t Ord, typename Plan, typename Slots, typename Ctx>
[[nodiscard]] constexpr step_result run_children(Plan& plan, Slots& slots, Ctx& ctx) {
    static_cast<void>(plan);
    static_cast<void>(slots);
    static_cast<void>(ctx);

    if constexpr (Ord == Plan::template child_count<I>) {
        return step_result::success();
    } else {
        constexpr auto child = Plan::template child_index<I, Ord>;

        const auto child_step = run_node<child>(plan, slots, ctx);

        if (child_step.status == step_status::abort_branch) {
            return run_children<I, Ord + 1>(plan, slots, ctx);
        }

        if (!child_step.ok()) {
            return child_step;
        }

        return run_children<I, Ord + 1>(plan, slots, ctx);
    }
}

} // namespace yorch::detail
