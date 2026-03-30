#pragma once
#include <utility>
#include <type_traits>

#include "yorch/assert.hpp"
#include "yorch/detail/maybe_storage.hpp"

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
    abort_branch,
    abort_execution,
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

    /** @brief Creates a result that aborts the current branch. */
    [[nodiscard]] static constexpr step_result abort_branch() noexcept {
        return {step_status::abort_branch};
    }

    /** @brief Creates a result that aborts the whole execution. */
    [[nodiscard]] static constexpr step_result abort_execution() noexcept {
        return {step_status::abort_execution};
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
 * @brief Represents a step result that may carry a produced value.
 *
 * `task_result<T>` models a value produced by a task and intended to flow through
 * the execution pipeline. `T` must be a non-reference type so the result keeps
 * clear value semantics instead of exposing an alias to external storage.
 *
 * In the current contract, only `success` results may carry a payload.
 * All non-success states are value-less and therefore do not require
 * constructing a placeholder `T`.
 *
 * @tparam T Value type produced by the task. Must not be a reference type.
 */
template <typename T>
struct task_result {
    static_assert(!std::is_reference_v<T>,
                  "yorch::task_result<T> does not support reference types");

public:
    /// Value type carried by this result wrapper.
    using value_type = T;

    task_result() = delete;
    task_result(const task_result&) = default;
    task_result(task_result&&) = default;
    task_result& operator=(const task_result&) = default;
    task_result& operator=(task_result&&) = default;
    ~task_result() = default;

    /** @brief Execution status of the current step. */
    step_result step {};

    template <typename U>
    [[nodiscard]] static constexpr task_result success(U&& value)
        noexcept(std::is_nothrow_constructible_v<T, U&&>) {
        return task_result(step_result::success(), std::forward<U>(value));
    }

    [[nodiscard]] static constexpr task_result failure() noexcept {
        return from_step(step_result::failure());
    }

    [[nodiscard]] static constexpr task_result skip() noexcept {
        return from_step(step_result::skip());
    }

    [[nodiscard]] static constexpr task_result retry() noexcept {
        return from_step(step_result::retry());
    }

    [[nodiscard]] static constexpr task_result abort_branch() noexcept {
        return from_step(step_result::abort_branch());
    }

    [[nodiscard]] static constexpr task_result abort_execution() noexcept {
        return from_step(step_result::abort_execution());
    }

    [[nodiscard]] static constexpr task_result from_step(step_result s) noexcept {
        YORCH_ASSERT(!s.ok() &&
                     "yorch::task_result<T>::from_step(...) requires a non-success status");
        return task_result(s);
    }

    [[nodiscard]] constexpr bool has_value() const noexcept {
        return value_.has_value();
    }

    [[nodiscard]] constexpr T& value() & noexcept {
        return value_.get();
    }

    [[nodiscard]] constexpr const T& value() const& noexcept {
        return value_.get();
    }

    [[nodiscard]] constexpr T&& value() && noexcept {
        return std::move(value_).get();
    }

private:
    constexpr explicit task_result(step_result s) noexcept
        : step(s) {}

    template <typename U>
    constexpr task_result(step_result s, U&& value)
        noexcept(std::is_nothrow_constructible_v<T, U&&>)
        : step(s) {
        value_.emplace(std::forward<U>(value));
    }

    detail::maybe_storage<T> value_ {};
};

/**
 * @brief Specialization for tasks that do not produce an extra value.
 *
 * Use `task_result<void>` when a task only needs to report execution status.
 */
template <>
struct task_result<void> {
    /// Value type carried by this result wrapper.
    using value_type = void;

    /** @brief Execution status of the current step. */
    step_result step {};

    [[nodiscard]] static constexpr task_result success() noexcept {
        return {step_result::success()};
    }

    [[nodiscard]] static constexpr task_result failure() noexcept {
        return {step_result::failure()};
    }

    [[nodiscard]] static constexpr task_result skip() noexcept {
        return {step_result::skip()};
    }

    [[nodiscard]] static constexpr task_result retry() noexcept {
        return {step_result::retry()};
    }

    [[nodiscard]] static constexpr task_result abort_branch() noexcept {
        return {step_result::abort_branch()};
    }

    [[nodiscard]] static constexpr task_result abort_execution() noexcept {
        return {step_result::abort_execution()};
    }

    [[nodiscard]] static constexpr task_result from_step(step_result s) noexcept {
        return {s};
    }
};

/**
 * @brief Type trait that detects `task_result<T>`.
 *
 * The primary template reports `false`. The specialization for
 * `task_result<T>` reports `true`, and the variable template strips cv-ref
 * qualifiers before checking.
 *
 * @tparam T Candidate type to inspect.
 */
template <typename T>
struct is_task_result : std::false_type {};

template <typename T>
struct is_task_result<task_result<T>> : std::true_type {};

template <typename T>
inline constexpr bool is_task_result_v =
    is_task_result<std::remove_cvref_t<T>>::value;

/**
 * @brief Extracts the payload type from `task_result<T>`.
 *
 * @tparam T `task_result` wrapper type.
 */
template <typename T>
struct task_result_value;

template <typename T>
struct task_result_value<task_result<T>> {
    using type = T;
};

template <typename T>
using task_result_value_t =
    typename task_result_value<std::remove_cvref_t<T>>::type;

}  // namespace yorch
