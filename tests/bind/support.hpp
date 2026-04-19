#pragma once

#include "yorch/bind.hpp" // IWYU pragma: keep
#include "yorch/executor.hpp" // IWYU pragma: keep
#include "yorch/result.hpp"

#include <gtest/gtest.h>

#include <string>
#include <type_traits>
#include <utility>

namespace bind_test_support {

inline bool add_and_check_positive(int& value, int delta) {
    value += delta;
    return value > 0;
}

struct recorder {
    int seen_value = 0;
    const std::string* seen_label = nullptr;

    yorch::step_result operator()(int& value, const std::string& label, long delta) {
        seen_value = value;
        seen_label = &label;
        value += static_cast<int>(delta);
        return label == "job"
                 ? yorch::step_result::retry()
                 : yorch::step_result::failure();
    }
};

struct ref_recorder {
    int* seen_value = nullptr;
    const std::string* seen_label = nullptr;

    yorch::step_result operator()(int& value, const std::string& label) {
        seen_value = &value;
        seen_label = &label;
        value += 2;
        return yorch::step_result::success();
    }
};

struct member_worker {
    int seen_value = 0;
    const std::string* seen_label = nullptr;
    int state = 0;

    yorch::step_result accumulate(int& value, const std::string& label, long delta) {
        seen_value = value;
        seen_label = &label;
        state += static_cast<int>(delta);
        value += static_cast<int>(delta);
        return label == "job"
                 ? yorch::step_result::success()
                 : yorch::step_result::failure();
    }

    bool note(int value) noexcept {
        seen_value = value;
        state += value;
        return true;
    }

    [[nodiscard]] int read_state(int delta) const noexcept {
        return state + delta;
    }

    yorch::step_result mutate_state(int delta) noexcept {
        state += delta;
        return yorch::step_result::success();
    }

    yorch::step_result advance_and_emit(int delta, yorch::direct_out<int> out) noexcept {
        state += delta;
        return out.success(state);
    }

    [[nodiscard]] yorch::step_result emit_state(yorch::direct_out<int> out) const noexcept {
        return out.success(state);
    }

    yorch::step_result emit(int& value, yorch::direct_out<std::string> out) noexcept {
        seen_value = value;
        value += state;
        return out.success(std::to_string(value));
    }
};

struct move_only_member_worker {
    explicit move_only_member_worker(int initial) : state(initial) {}

    move_only_member_worker(const move_only_member_worker&) = delete;
    move_only_member_worker& operator=(const move_only_member_worker&) = delete;
    move_only_member_worker(move_only_member_worker&&) noexcept = default;
    move_only_member_worker& operator=(move_only_member_worker&&) noexcept = default;
    ~move_only_member_worker() = default;

    int state = 0;

    int bump(int delta) noexcept {
        state += delta;
        return state;
    }

    [[nodiscard]] yorch::step_result emit(yorch::direct_out<int> out) const noexcept {
        return out.success(state);
    }
};

template <typename F>
concept can_make_task_member_without_receiver =
    requires(F&& func) {
        requires (!std::is_void_v<decltype(yorch::task_member(std::forward<F>(func)))>);
    };

template <typename F>
concept can_make_task_member_from_receiver_first =
    requires(F&& func) {
        yorch::task_member(
            std::forward<F>(func),
            yorch::value(member_worker {}))(
            yorch::value(1));
    };

template <typename F>
concept can_make_task_into_member_without_receiver =
    requires(F&& func) {
        requires (!std::is_void_v<decltype(yorch::task_into_member(std::forward<F>(func)))>);
    };

template <typename F>
concept can_make_task_into_member_from_receiver_first =
    requires(F&& func) {
        yorch::task_into_member(
            std::forward<F>(func),
            yorch::value(member_worker {}))(
            yorch::value(1));
    };

struct forward_prev_probe {
    int value = 0;
};

}  // namespace bind_test_support
