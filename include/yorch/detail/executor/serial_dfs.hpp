#pragma once

#include <cstddef>

#include "../../result.hpp"
#include "node_entry.hpp"

namespace yorch::detail {

template <std::size_t I, std::size_t Ord = 0, typename Plan, typename Slots>
[[nodiscard]] constexpr step_result run_children(Plan& plan, Slots& slots);

template <std::size_t I, std::size_t Ord = 0, typename Plan, typename Slots, typename Ctx>
[[nodiscard]] constexpr step_result run_children(Plan& plan, Slots& slots, Ctx& ctx);

template <std::size_t I, typename Plan, typename Slots>
[[nodiscard]] constexpr step_result run_node(Plan& plan, Slots& slots) {
    const auto entered = enter_node<I>(plan, slots);

    if (!entered.step.ok()) {
        return entered.step;
    }

    node_slot_guard<Slots, I> guard {slots};
    if (entered.payload_live) {
        guard.arm();
    }

    return run_children<I>(plan, slots);
}

template <std::size_t I, typename Plan, typename Slots, typename Ctx>
[[nodiscard]] constexpr step_result run_node(Plan& plan, Slots& slots, Ctx& ctx) {
    const auto entered = enter_node<I>(plan, slots, ctx);

    if (!entered.step.ok()) {
        return entered.step;
    }

    node_slot_guard<Slots, I> guard {slots};
    if (entered.payload_live) {
        guard.arm();
    }

    return run_children<I>(plan, slots, ctx);
}

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
