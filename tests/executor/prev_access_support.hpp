#pragma once

#include "support.hpp" // IWYU pragma: keep

namespace executor_test_support {

struct member_prev_worker {
    int value = 0;

    int mutate(int delta) noexcept {
        value += delta;
        return value;
    }

    int mutate_from_view(const member_prev_worker& other) noexcept {
        value += other.value;
        return value;
    }

    int mutate_with_copy(member_prev_worker other) noexcept {
        value += other.value;
        return value;
    }

    [[nodiscard]] int read_plus(int delta) const noexcept {
        return value + delta;
    }

    [[nodiscard]] int read_with_copy(member_prev_worker other) const noexcept {
        return value + other.value;
    }
};

struct forwarded_payload {
    int value = 0;
};

struct move_only_forwarded_payload {
    explicit move_only_forwarded_payload(int initial) : value(initial) {}

    move_only_forwarded_payload(const move_only_forwarded_payload&) = delete;
    move_only_forwarded_payload& operator=(const move_only_forwarded_payload&) = delete;
    move_only_forwarded_payload(move_only_forwarded_payload&&) noexcept = default;
    move_only_forwarded_payload& operator=(move_only_forwarded_payload&&) noexcept = default;
    ~move_only_forwarded_payload() = default;

    int value = 0;
};

struct member_forward_prev_worker {
    int state = 0;

    yorch::step_result mutate_self(int delta) noexcept {
        state += delta;
        return yorch::step_result::success();
    }
};

struct member_forward_prev_service {
    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    yorch::step_result bump(forwarded_payload& payload, int delta) noexcept {
        payload.value += delta;
        return yorch::step_result::success();
    }

    // NOLINTNEXTLINE(readability-convert-member-functions-to-static, cppcoreguidelines-rvalue-reference-param-not-moved)
    yorch::step_result consume_and_bump(forwarded_payload&& payload, int delta) noexcept {
        payload.value += delta;
        return yorch::step_result::success();
    }

    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    yorch::step_result bump_int(int& payload, int delta) noexcept {
        payload += delta;
        return yorch::step_result::success();
    }
};

}  // namespace executor_test_support
