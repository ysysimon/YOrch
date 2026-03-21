#pragma once
#include <type_traits>

namespace yorch {

enum class step_status : unsigned char {
    success,
    failure,
    skip,
    retry,
    abort_chain,
    abort_schedule,
};

struct step_result {
    step_status status = step_status::success;

    [[nodiscard]] static constexpr step_result success() noexcept {
        return {step_status::success};
    }

    [[nodiscard]] static constexpr step_result failure() noexcept {
        return {step_status::failure};
    }

    [[nodiscard]] static constexpr step_result skip() noexcept {
        return {step_status::skip};
    }

    [[nodiscard]] static constexpr step_result retry() noexcept {
        return {step_status::retry};
    }

    [[nodiscard]] static constexpr step_result abort_chain() noexcept {
        return {step_status::abort_chain};
    }

    [[nodiscard]] static constexpr step_result abort_schedule() noexcept {
        return {step_status::abort_schedule};
    }

    [[nodiscard]] constexpr bool ok() const noexcept {
        return status == step_status::success;
    }

    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return ok();
    }
};

template <typename T>
struct task_result {
    static_assert(!std::is_reference_v<T>, "yorch::task_result<T> does not support reference types");

    step_result step {};
    T value;
};

template <>
struct task_result<void> {
    step_result step {};
};

}  // namespace yorch
