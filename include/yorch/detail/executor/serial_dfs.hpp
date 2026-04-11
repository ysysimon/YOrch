#pragma once

#include <cstddef>

#include "../../result.hpp"
#include "node_entry.hpp"

namespace yorch::detail {

template <typename Plan, std::size_t I>
struct node_fanout_guard {
    plan_fanout_state<Plan>* fanout = nullptr;
    bool armed = false;

    explicit constexpr node_fanout_guard(plan_fanout_state<Plan>* in_fanout) noexcept
        : fanout(in_fanout) {}

    node_fanout_guard(const node_fanout_guard&) = default;
    node_fanout_guard& operator=(const node_fanout_guard&) = default;
    ~node_fanout_guard() {
        if (armed) {
            fanout->template destroy_fanout_staging<I>();
        }
    }

    constexpr void arm() noexcept {
        armed = true;
    }
};

template <std::size_t I, std::size_t Ord = 0, typename Plan, typename Slots>
[[nodiscard]] constexpr step_result run_children(
    Plan& plan,
    Slots& slots,
    plan_fanout_state<Plan>& fanout);

template <std::size_t I, std::size_t Ord = 0, typename Plan, typename Slots, typename Ctx>
[[nodiscard]] constexpr step_result run_children(
    Plan& plan,
    Slots& slots,
    plan_fanout_state<Plan>& fanout,
    Ctx& ctx);

template <std::size_t I, typename Plan, typename Slots>
[[nodiscard]] constexpr step_result run_node(
    Plan& plan,
    Slots& slots,
    plan_fanout_state<Plan>& fanout) {
    const auto entered = enter_node<I>(plan, slots, fanout);

    if (!entered.step.ok()) {
        return entered.step;
    }

    node_slot_guard<Slots, I> guard {slots};
    if (entered.payload_live) {
        guard.arm();
    }

    return run_children<I>(plan, slots, fanout);
}

template <std::size_t I, typename Plan, typename Slots, typename Ctx>
[[nodiscard]] constexpr step_result run_node(
    Plan& plan,
    Slots& slots,
    plan_fanout_state<Plan>& fanout,
    Ctx& ctx) {
    const auto entered = enter_node<I>(plan, slots, fanout, ctx);

    if (!entered.step.ok()) {
        return entered.step;
    }

    node_slot_guard<Slots, I> guard {slots};
    if (entered.payload_live) {
        guard.arm();
    }

    return run_children<I>(plan, slots, fanout, ctx);
}

template <std::size_t I, std::size_t Ord, typename Plan, typename Slots>
[[nodiscard]] constexpr step_result run_children(
    Plan& plan,
    Slots& slots,
    plan_fanout_state<Plan>& fanout) {
    static_cast<void>(plan);
    static_cast<void>(slots);

    node_fanout_guard<Plan, I> guard {&fanout};
    if constexpr (Ord == 0) {
        fanout.template prepare_fanout_staging<I>(slots);
        guard.arm();
    }

    if constexpr (Ord == Plan::template child_count<I>) {
        return step_result::success();
    } else {
        constexpr auto child = Plan::template child_index<I, Ord>;

        const auto child_step = run_node<child>(plan, slots, fanout);

        if (child_step.status == step_status::abort_branch) {
            return run_children<I, Ord + 1>(plan, slots, fanout);
        }

        if (!child_step.ok()) {
            return child_step;
        }

        return run_children<I, Ord + 1>(plan, slots, fanout);
    }
}

template <std::size_t I, std::size_t Ord, typename Plan, typename Slots, typename Ctx>
[[nodiscard]] constexpr step_result run_children(
    Plan& plan,
    Slots& slots,
    plan_fanout_state<Plan>& fanout,
    Ctx& ctx) {
    static_cast<void>(plan);
    static_cast<void>(slots);
    static_cast<void>(ctx);

    node_fanout_guard<Plan, I> guard {&fanout};
    if constexpr (Ord == 0) {
        fanout.template prepare_fanout_staging<I>(slots);
        guard.arm();
    }

    if constexpr (Ord == Plan::template child_count<I>) {
        return step_result::success();
    } else {
        constexpr auto child = Plan::template child_index<I, Ord>;

        const auto child_step = run_node<child>(plan, slots, fanout, ctx);

        if (child_step.status == step_status::abort_branch) {
            return run_children<I, Ord + 1>(plan, slots, fanout, ctx);
        }

        if (!child_step.ok()) {
            return child_step;
        }

        return run_children<I, Ord + 1>(plan, slots, fanout, ctx);
    }
}

} // namespace yorch::detail
