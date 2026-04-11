#pragma once

#include "yorch/executor.hpp"

#include "yorch/bind.hpp" // IWYU pragma: keep
#include "yorch/plan.hpp" // IWYU pragma: keep

#include <gtest/gtest.h>

#include <stdexcept> // IWYU pragma: keep
#include <string> // IWYU pragma: keep
#include <vector> // IWYU pragma: keep

namespace executor_test_support {

struct lifetime_tracker {
    int live_count = 0;
    int destroyed_count = 0;
};

struct lifetime_probe {
    lifetime_tracker* tracker = nullptr;

    int value = 0;

    explicit lifetime_probe(lifetime_tracker& tracker_ref)
        : tracker(&tracker_ref) {
        ++tracker->live_count;
    }

    lifetime_probe(lifetime_tracker& tracker_ref, int in)
        : tracker(&tracker_ref),
          value(in) {
        ++tracker->live_count;
    }

    lifetime_probe(const lifetime_probe& other) noexcept
        : tracker(other.tracker),
          value(other.value) {
        ++tracker->live_count;
    }

    lifetime_probe(lifetime_probe&& other) noexcept
        : tracker(other.tracker),
          value(other.value) {
        ++tracker->live_count;
        other.value = -1;
    }

    lifetime_probe& operator=(const lifetime_probe&) = default;
    lifetime_probe& operator=(lifetime_probe&&) noexcept = default;

    ~lifetime_probe() {
        --tracker->live_count;
        ++tracker->destroyed_count;
    }
};

struct immovable_payload {
    explicit immovable_payload(int in) : value(in) {}

    immovable_payload(const immovable_payload&) = delete;
    immovable_payload& operator=(const immovable_payload&) = delete;
    immovable_payload(immovable_payload&&) = delete;
    immovable_payload& operator=(immovable_payload&&) = delete;
    ~immovable_payload() = default;

    int value = 0;
};

struct move_only_payload {
    explicit move_only_payload(int in) : value(in) {}

    move_only_payload(const move_only_payload&) = delete;
    move_only_payload& operator=(const move_only_payload&) = delete;
    move_only_payload(move_only_payload&&) noexcept = default;
    move_only_payload& operator=(move_only_payload&&) noexcept = default;
    ~move_only_payload() = default;

    int value = 0;
};

template <typename Plan>
concept can_run_plan =
    requires(Plan& plan) {
        yorch::run_plan(plan);
    };

template <typename LayoutPolicy, typename Plan>
concept can_run_plan_with_layout =
    requires(Plan& plan) {
        yorch::run_plan<LayoutPolicy>(plan);
    };

template <typename LayoutPolicy, typename ExecPolicy, typename Plan>
concept can_run_plan_with_layout_and_exec =
    requires(Plan& plan) {
        yorch::run_plan<LayoutPolicy, ExecPolicy>(plan);
    };

template <typename Plan, typename Ctx>
concept can_run_plan_with_ctx =
    requires(Plan& plan, Ctx& ctx) {
        yorch::run_plan(plan, ctx);
    };

template <typename LayoutPolicy, typename Plan, typename Ctx>
concept can_run_plan_with_ctx_and_layout =
    requires(Plan& plan, Ctx& ctx) {
        yorch::run_plan<LayoutPolicy>(plan, ctx);
    };

template <typename LayoutPolicy, typename ExecPolicy, typename Plan, typename Ctx>
concept can_run_plan_with_ctx_and_layout_and_exec =
    requires(Plan& plan, Ctx& ctx) {
        yorch::run_plan<LayoutPolicy, ExecPolicy>(plan, ctx);
    };

}  // namespace executor_test_support
