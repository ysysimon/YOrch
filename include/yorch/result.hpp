#pragma once
#include <type_traits>

namespace yorch {

/**
 * @brief Describes the execution status of a single step.
 *
 * This enum only represents control-flow state and does not carry a task value.
 */
enum class step_status : unsigned char {
    success,
    failure,
    skip,
    retry,
    abort_chain,
    abort_schedule,
};

/**
 * @brief Represents the basic outcome of a task step.
 *
 * `step_result` describes whether the current step succeeded, was skipped, should
 * be retried, or should stop later execution. It only models status; use
 * `task_result<T>` when a step also needs to produce a value for downstream steps.
 */
struct step_result {
    step_status status = step_status::success;

    /** @brief Creates a successful result. */
    [[nodiscard]] static constexpr step_result success() noexcept {
        return {step_status::success};
    }

    /** @brief Creates a failed result. */
    [[nodiscard]] static constexpr step_result failure() noexcept {
        return {step_status::failure};
    }

    /** @brief Creates a skipped result. */
    [[nodiscard]] static constexpr step_result skip() noexcept {
        return {step_status::skip};
    }

    /** @brief Creates a retry result. */
    [[nodiscard]] static constexpr step_result retry() noexcept {
        return {step_status::retry};
    }

    /** @brief Creates a result that aborts the current chain. */
    [[nodiscard]] static constexpr step_result abort_chain() noexcept {
        return {step_status::abort_chain};
    }

    /** @brief Creates a result that aborts the whole schedule. */
    [[nodiscard]] static constexpr step_result abort_schedule() noexcept {
        return {step_status::abort_schedule};
    }

    /** @brief Returns whether the status is `success`. */
    [[nodiscard]] constexpr bool ok() const noexcept {
        return status == step_status::success;
    }

    /** @brief Allows the result to be tested as a success flag. */
    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return ok();
    }
};

/**
 * @brief Represents a step result that also carries a produced value.
 *
 * `task_result<T>` models a value produced by a task and intended to flow through
 * the execution pipeline. `T` must be a non-reference type so the result keeps
 * clear value semantics instead of exposing an alias to external storage.
 *
 * @tparam T Value type produced by the task. Must not be a reference type.
 */
template <typename T>
struct task_result {
    static_assert(!std::is_reference_v<T>, "yorch::task_result<T> does not support reference types");

    /** @brief Execution status of the current step. */
    step_result step {};
    /** @brief Value produced by the current task. */
    T value;
};

/**
 * @brief Specialization for tasks that do not produce an extra value.
 *
 * Use `task_result<void>` when a task only needs to report execution status.
 */
template <>
struct task_result<void> {
    /** @brief Execution status of the current step. */
    step_result step {};
};

}  // namespace yorch
