#pragma once

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include "../../result.hpp"
#include "node_entry.hpp"

namespace yorch::detail {

/**
 * @brief Runtime frame used by the explicit-heap-stack DFS executor.
 *
 * Recursive DFS keeps this state implicitly on the C++ call stack. The
 * explicit-stack executor instead materializes one frame per active node on a
 * heap-backed `std::vector`, so deep plans no longer depend on call-stack
 * depth.
 */
struct dfs_stack_frame {
    std::size_t node_index = 0;
    std::size_t next_child_ordinal = 0;
    bool entered = false;
    bool payload_live = false;
    bool requires_fanout_staging = false;
    bool fanout_prepared = false;
};

/**
 * @brief Per-node runtime thunk for entering node `I`.
 *
 * The explicit-stack executor discovers the current node only through the
 * runtime `frame.node_index`, but the real node execution helper is still the
 * compile-time template `enter_node<I>(...)`. This thunk fixes `I` and exposes
 * a uniform function-pointer shape so later code can dispatch by index.
 */
template <std::size_t I, typename Plan, typename Slots>
[[nodiscard]] constexpr node_enter_result enter_node_runtime(
    Plan& plan,
    Slots& slots,
    plan_fanout_state<Plan>& fanout) {
    return enter_node<I>(plan, slots, fanout);
}

/**
 * @brief Context-threading variant of the per-node enter thunk.
 */
template <std::size_t I, typename Plan, typename Slots, typename Ctx>
[[nodiscard]] constexpr node_enter_result enter_node_runtime(
    Plan& plan,
    Slots& slots,
    plan_fanout_state<Plan>& fanout,
    Ctx& ctx) {
    return enter_node<I>(plan, slots, fanout, ctx);
}

/**
 * @brief Per-node runtime thunk for destroying node `I`'s payload slot.
 *
 * Payload destruction is also a compile-time operation (`slots.destroy<I>()`),
 * so the explicit-stack executor needs the same "fix `I`, erase the rest"
 * adapter layer as node entry.
 */
template <std::size_t I, typename Slots>
constexpr void destroy_node_payload_runtime(Slots& slots) noexcept {
    slots.template destroy<I>();
}

template <std::size_t I, typename Plan, typename Slots>
constexpr void prepare_fanout_runtime(Slots& slots, plan_fanout_state<Plan>& fanout) {
    fanout.template prepare_fanout_staging<I>(slots);
}

template <std::size_t I, typename Plan>
constexpr void destroy_fanout_runtime(plan_fanout_state<Plan>& fanout) noexcept {
    fanout.template destroy_fanout_staging<I>();
}

/**
 * @brief Builds the runtime dispatch table for node entry without external
 * context.
 *
 * After expansion this returns an array equivalent to:
 *
 * `{
 *   &enter_node_runtime<0, Plan, Slots>,
 *   &enter_node_runtime<1, Plan, Slots>,
 *   ...
 * }`
 *
 * The explicit-stack loop can then execute `dispatch[node_index](plan, slots)`
 * to bridge from runtime node indices back into compile-time node-specific
 * code.
 */
template <typename Plan, typename Slots, std::size_t... I>
[[nodiscard]] consteval auto make_enter_node_dispatch_table(std::index_sequence<I...>) {
    using fn_t = node_enter_result (*)(Plan&, Slots&, plan_fanout_state<Plan>&);
    return std::array<fn_t, sizeof...(I)> {
        &enter_node_runtime<I, Plan, Slots>...
    };
}

/**
 * @brief Builds the runtime dispatch table for node entry with external
 * context.
 */
template <typename Plan, typename Slots, typename Ctx, std::size_t... I>
[[nodiscard]] consteval auto make_enter_node_dispatch_table(std::index_sequence<I...>) {
    using fn_t = node_enter_result (*)(Plan&, Slots&, plan_fanout_state<Plan>&, Ctx&);
    return std::array<fn_t, sizeof...(I)> {
        &enter_node_runtime<I, Plan, Slots, Ctx>...
    };
}

/**
 * @brief Builds the runtime dispatch table for payload destruction.
 *
 * This is the destruction-side counterpart to `make_enter_node_dispatch_table(...)`.
 * Once a frame leaves the active DFS path, the loop uses
 * `dispatch[node_index](slots)` to call the matching `slots.destroy<I>()`
 * instantiation.
 */
template <typename Plan, typename Slots, std::size_t... I>
[[nodiscard]] consteval auto make_destroy_node_dispatch_table(std::index_sequence<I...>) {
    using fn_t = void (*)(Slots&) noexcept;
    return std::array<fn_t, sizeof...(I)> {
        &destroy_node_payload_runtime<I, Slots>...
    };
}

template <typename Plan, typename Slots, std::size_t... I>
[[nodiscard]] consteval auto make_prepare_fanout_dispatch_table(std::index_sequence<I...>) {
    using fn_t = void (*)(Slots&, plan_fanout_state<Plan>&);
    return std::array<fn_t, sizeof...(I)> {
        &prepare_fanout_runtime<I, Plan, Slots>...
    };
}

template <typename Plan, typename Slots, std::size_t... I>
[[nodiscard]] consteval auto make_destroy_fanout_dispatch_table(std::index_sequence<I...>) {
    using fn_t = void (*)(plan_fanout_state<Plan>&) noexcept;
    return std::array<fn_t, sizeof...(I)> {
        &destroy_fanout_runtime<I, Plan>...
    };
}

template <typename Plan, std::size_t... I>
[[nodiscard]] consteval auto make_requires_fanout_staging_table(std::index_sequence<I...>) {
    return std::array<bool, sizeof...(I)> {
        parent_requires_fanout_staging_v<Plan, I>...
    };
}

/**
 * @brief Executes serial DFS using a heap-backed explicit stack.
 *
 * The loop itself only knows runtime node indices stored in `frames`, so all
 * type-specific work is delegated to the enter/destroy dispatch callables.
 * Together they form the bridge between:
 *
 * - runtime traversal state: `frame.node_index`
 * - compile-time node logic: `enter_node<I>(...)` and `slots.destroy<I>()`
 */
template <typename Plan, typename EnterInvoker, typename DestroyInvoker, typename PrepareFanoutInvoker, typename DestroyFanoutInvoker>
[[nodiscard]] constexpr step_result run_explicit_heap_stack_loop(
    const auto& requires_fanout_staging_by_index,
    EnterInvoker enter_node_by_index,
    DestroyInvoker destroy_node_payload_by_index,
    PrepareFanoutInvoker prepare_fanout_by_index,
    DestroyFanoutInvoker destroy_fanout_by_index) {
    std::vector<dfs_stack_frame> frames;
    frames.reserve(Plan::max_level + 1);
    frames.push_back(dfs_stack_frame {
        .node_index = 0,
        .requires_fanout_staging = requires_fanout_staging_by_index[0]});

    // `current_step` always represents the result of the subtree that most
    // recently finished. `subtree_complete == false` means we are still
    // descending into the current top frame. `subtree_complete == true` means
    // the top subtree has ended and its result now needs to be consumed by the
    // parent frame.
    auto current_step = step_result::success();
    bool subtree_complete = false;

    while (!frames.empty()) {
        auto& frame = frames.back();

        if (!subtree_complete) {
            // First time we see this frame: execute the node itself, materialize
            // its payload if needed, and record whether that payload must stay
            // alive until the subtree later unwinds.
            if (!frame.entered) {
                const auto entered = enter_node_by_index(frame.node_index);
                frame.entered = true;
                frame.payload_live = entered.payload_live;

                // Non-success from the node body ends this subtree immediately.
                // Child traversal is skipped and the result is handed to the
                // parent during the unwind phase below.
                if (!entered.step.ok()) {
                    current_step = entered.step;
                    subtree_complete = true;
                    continue;
                }
            }

            if (frame.requires_fanout_staging && !frame.fanout_prepared) {
                prepare_fanout_by_index(frame.node_index);
                frame.fanout_prepared = true;
            }

            // The node body already succeeded and there are no more children to
            // visit, so this whole subtree completes with `success`.
            if (frame.next_child_ordinal == Plan::child_counts[frame.node_index]) {
                current_step = step_result::success();
                subtree_complete = true;
                continue;
            }

            // Descend into the next direct child of the current node. Children
            // are stored in one flattened adjacency array, so this uses the
            // parent's child-list start offset plus the next child ordinal to
            // recover the concrete child node index.
            const auto child_index = Plan::child_indices[
                Plan::child_offsets[frame.node_index] + frame.next_child_ordinal];
            frames.push_back(dfs_stack_frame {
                .node_index = child_index,
                .requires_fanout_staging = requires_fanout_staging_by_index[child_index]});
            continue;
        }

        // We are leaving the current subtree. If the node produced a live
        // payload, destroy it now because the node is about to leave the active
        // DFS path.
        // Take a by-value snapshot before `pop_back()` so later stack mutation
        // cannot leave us with a dangling reference to the finished frame.
        const auto finished = frames.back();
        if (finished.requires_fanout_staging && finished.fanout_prepared) {
            destroy_fanout_by_index(finished.node_index);
        }
        if (finished.payload_live) {
            destroy_node_payload_by_index(finished.node_index);
        }

        frames.pop_back();
        if (frames.empty()) {
            // Popping the last frame means the root subtree just finished, so
            // `current_step` is the final plan result.
            return current_step;
        }

        auto& parent = frames.back();
        if (current_step.ok() || current_step.status == step_status::abort_branch) {
            // From the parent's perspective both `success` and `abort_branch`
            // mean "the current child subtree has ended, move to the next
            // sibling if there is one".
            ++parent.next_child_ordinal;

            if (parent.next_child_ordinal == Plan::child_counts[parent.node_index]) {
                // This parent has now exhausted all direct children, so its own
                // subtree finishes successfully and the unwind continues upward.
                current_step = step_result::success();
                subtree_complete = true;
            } else {
                // There is another sibling to execute, so switch back to the
                // descend phase and keep the parent frame active.
                subtree_complete = false;
            }

            continue;
        }

        // `failure`, `retry`, and `abort_execution` are non-local results. They
        // stop sibling traversal at every level, so we keep unwinding with the
        // same `current_step` until the stack becomes empty.
        subtree_complete = true;
    }

    return current_step;
}

template <typename Plan, typename Slots>
struct explicit_heap_stack_dispatch {
    static constexpr auto enter =
        make_enter_node_dispatch_table<Plan, Slots>(
            std::make_index_sequence<Plan::node_count> {});
    static constexpr auto destroy =
        make_destroy_node_dispatch_table<Plan, Slots>(
            std::make_index_sequence<Plan::node_count> {});
    static constexpr auto prepare_fanout =
        make_prepare_fanout_dispatch_table<Plan, Slots>(
            std::make_index_sequence<Plan::node_count> {});
    static constexpr auto destroy_fanout =
        make_destroy_fanout_dispatch_table<Plan, Slots>(
            std::make_index_sequence<Plan::node_count> {});
    static constexpr auto requires_fanout_staging =
        make_requires_fanout_staging_table<Plan>(
            std::make_index_sequence<Plan::node_count> {});
};

template <typename Plan, typename Slots, typename Ctx>
struct explicit_heap_stack_dispatch_with_ctx {
    static constexpr auto enter =
        make_enter_node_dispatch_table<Plan, Slots, Ctx>(
            std::make_index_sequence<Plan::node_count> {});
    static constexpr auto destroy =
        make_destroy_node_dispatch_table<Plan, Slots>(
            std::make_index_sequence<Plan::node_count> {});
    static constexpr auto prepare_fanout =
        make_prepare_fanout_dispatch_table<Plan, Slots>(
            std::make_index_sequence<Plan::node_count> {});
    static constexpr auto destroy_fanout =
        make_destroy_fanout_dispatch_table<Plan, Slots>(
            std::make_index_sequence<Plan::node_count> {});
    static constexpr auto requires_fanout_staging =
        make_requires_fanout_staging_table<Plan>(
            std::make_index_sequence<Plan::node_count> {});
};

template <typename Plan, typename Slots>
[[nodiscard]] constexpr step_result run_explicit_heap_stack(
    Plan& plan,
    Slots& slots,
    plan_fanout_state<Plan>& fanout) {
    return run_explicit_heap_stack_loop<Plan>(
        explicit_heap_stack_dispatch<Plan, Slots>::requires_fanout_staging,
        [&](std::size_t node_index) constexpr -> node_enter_result {
            return explicit_heap_stack_dispatch<Plan, Slots>::enter[node_index](plan, slots, fanout);
        },
        [&](std::size_t node_index) constexpr {
            explicit_heap_stack_dispatch<Plan, Slots>::destroy[node_index](slots);
        },
        [&](std::size_t node_index) constexpr {
            explicit_heap_stack_dispatch<Plan, Slots>::prepare_fanout[node_index](slots, fanout);
        },
        [&](std::size_t node_index) constexpr {
            explicit_heap_stack_dispatch<Plan, Slots>::destroy_fanout[node_index](fanout);
        });
}

template <typename Plan, typename Slots, typename Ctx>
[[nodiscard]] constexpr step_result run_explicit_heap_stack(
    Plan& plan,
    Slots& slots,
    plan_fanout_state<Plan>& fanout,
    Ctx& ctx) {
    return run_explicit_heap_stack_loop<Plan>(
        explicit_heap_stack_dispatch_with_ctx<Plan, Slots, Ctx>::requires_fanout_staging,
        [&](std::size_t node_index) constexpr -> node_enter_result {
            return explicit_heap_stack_dispatch_with_ctx<Plan, Slots, Ctx>::enter[node_index](
                plan,
                slots,
                fanout,
                ctx);
        },
        [&](std::size_t node_index) constexpr {
            explicit_heap_stack_dispatch_with_ctx<Plan, Slots, Ctx>::destroy[node_index](slots);
        },
        [&](std::size_t node_index) constexpr {
            explicit_heap_stack_dispatch_with_ctx<Plan, Slots, Ctx>::prepare_fanout[node_index](
                slots,
                fanout);
        },
        [&](std::size_t node_index) constexpr {
            explicit_heap_stack_dispatch_with_ctx<Plan, Slots, Ctx>::destroy_fanout[node_index](
                fanout);
        });
}

} // namespace yorch::detail
