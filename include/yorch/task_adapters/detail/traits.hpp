#pragma once

#include <concepts> // IWYU pragma: keep
#include <exception>
#include <type_traits>
#include <utility>

#include "../../context.hpp"
#include "../../result.hpp"

namespace yorch::detail {

template <typename Task, typename Ctx, typename Prev>
using raw_task_result_t =
    decltype(std::declval<Task>().invoke_raw(std::declval<exec_context<Ctx, Prev>&>()));

template <typename Task, typename = void>
struct declared_task_output;

template <typename Task>
struct declared_task_output<
    Task,
    std::void_t<typename std::remove_cvref_t<Task>::output_type>
> {
    using type = typename std::remove_cvref_t<Task>::output_type;
};

template <typename Task>
using declared_task_output_t = typename declared_task_output<Task>::type;

template <typename Task, typename = void>
struct has_declared_task_output : std::false_type {};

template <typename Task>
struct has_declared_task_output<
    Task,
    std::void_t<declared_task_output_t<Task>>
> : std::true_type {};

template <typename Task>
inline constexpr bool has_declared_task_output_v =
    has_declared_task_output<Task>::value;

/**
 * @brief Reports whether the default `catch_as_failure(task)` adapter can
 * synthesize a failure result for raw return type `R`.
 *
 * The no-policy overload intentionally stays conservative: it only supports
 * return categories where the library can manufacture a failure result without
 * inventing a payload object.
 *
 * @tparam R Raw task return type.
 */
template <typename R>
inline constexpr bool default_catch_supported_v =
    std::is_void_v<R> ||
    std::is_same_v<R, step_result>;

/**
 * @brief Extracts the result type returned by an exception fallback policy.
 *
 * The policy contract is a `noexcept` call that accepts a captured exception
 * either by value (`std::exception_ptr`) or by const lvalue reference
 * (`const std::exception_ptr&`).
 *
 * @tparam Policy Policy type stored inside `catch_as_failure(task, policy)`.
 * @tparam Enable SFINAE hook used to detect whether the policy is invocable.
 */
template <typename Policy, typename = void>
struct policy_result;

template <typename Policy>
struct policy_result<Policy, std::void_t<std::invoke_result_t<Policy&, const std::exception_ptr&>>> {
    using type = std::invoke_result_t<Policy&, const std::exception_ptr&>;
};

template <typename Policy>
using policy_result_t = typename policy_result<Policy>::type;

template <typename Task, typename = void>
struct declared_task_raw_result;

template <typename Task>
struct declared_task_raw_result<
    Task,
    std::void_t<typename std::remove_cvref_t<Task>::raw_result_type>
> {
    using type = typename std::remove_cvref_t<Task>::raw_result_type;
};

template <typename Task>
using declared_task_raw_result_t = typename declared_task_raw_result<Task>::type;

/**
 * @brief Reports whether a task's declared `raw_result_type` agrees with the
 * concrete return type of `invoke_raw(exec_context<Ctx, Prev>&)`.
 *
 * This trait is intentionally permissive for tasks that do not declare
 * `raw_result_type`: in that case it yields `true`, so runtime-only task
 * shapes can still participate in lower-level execution checks. When a task
 * does opt into the declared raw-result contract, however, this trait enforces
 * that the declaration remains consistent with the actual `invoke_raw(...)`
 * expression type observed for the concrete execution context.
 *
 * The executor and task adapters use this as a protocol guard so plan-facing
 * metadata and execution-time behavior cannot silently drift apart.
 *
 * @tparam Task Task type being checked.
 * @tparam Ctx Context schema for the concrete invocation.
 * @tparam Prev Direct-parent slot view type for the concrete invocation.
 * @tparam Enable SFINAE hook that activates strict checking only when
 * `Task::raw_result_type` is declared.
 */
template <typename Task, typename Ctx, typename Prev, typename = void>
struct declared_task_raw_result_matches_invoke_raw : std::true_type {};

template <typename Task, typename Ctx, typename Prev>
struct declared_task_raw_result_matches_invoke_raw<
    Task,
    Ctx,
    Prev,
    std::void_t<declared_task_raw_result_t<Task>>
> : std::bool_constant<
        std::same_as<
            raw_task_result_t<Task&, Ctx, Prev>,
            declared_task_raw_result_t<Task>>> {};

template <typename Task, typename Ctx, typename Prev>
inline constexpr bool declared_task_raw_result_matches_invoke_raw_v =
    declared_task_raw_result_matches_invoke_raw<Task, Ctx, Prev>::value;

template <typename Task, typename = void>
struct catch_failure_task_raw_result_base {};

template <typename Task>
struct catch_failure_task_raw_result_base<
    Task,
    std::void_t<declared_task_raw_result_t<Task>>
> {
    using inner_type = declared_task_raw_result_t<Task>;
    using raw_result_type = std::conditional_t<std::is_void_v<inner_type>, step_result, inner_type>;
};

template <typename Task, typename Policy, typename = void>
struct catch_failure_with_policy_task_raw_result_base {};

template <typename Task, typename Policy>
struct catch_failure_with_policy_task_raw_result_base<
    Task,
    Policy,
    std::void_t<
        declared_task_raw_result_t<Task>,
        policy_result_t<std::remove_cvref_t<Policy>>
    >
> {
    using inner_type = declared_task_raw_result_t<Task>;
    using policy_type = policy_result_t<std::remove_cvref_t<Policy>>;
    using raw_result_type = std::conditional_t<std::is_void_v<inner_type>, policy_type, inner_type>;
};

template <typename Task, typename = void>
struct retry_task_raw_result_base {};

template <typename Task>
struct retry_task_raw_result_base<
    Task,
    std::void_t<declared_task_raw_result_t<Task>>
> {
    using raw_result_type = declared_task_raw_result_t<Task>;
};

template <typename Task, typename = void>
struct forwarded_task_output_base {};

template <typename Task>
struct forwarded_task_output_base<
    Task,
    std::void_t<declared_task_output_t<Task>>
> {
    using output_type = declared_task_output_t<Task>;
};

/**
 * @brief Trait that reports whether a fallback policy is compatible with raw
 * task result type `R`.
 *
 * Compatibility means:
 * - the policy is invocable as `policy(const std::exception_ptr&)` and is
 *   `noexcept`
 * - for `void` tasks, the policy returns `step_result`
 * - for non-void tasks, the policy result is convertible to the raw result
 *
 * @tparam R Raw task return type.
 * @tparam Policy Policy type.
 * @tparam Enable SFINAE hook used to suppress diagnostics for non-invocable
 * policies.
 */
template <typename R, typename Policy, typename = void>
struct catch_policy_supported : std::false_type {};

template <typename R, typename Policy>
struct catch_policy_supported<
    R,
    Policy,
    std::void_t<std::invoke_result_t<Policy&, const std::exception_ptr&>>>
    : std::bool_constant<
          std::is_nothrow_invocable_v<Policy&, const std::exception_ptr&> &&
          (std::is_void_v<R>
               ? std::is_same_v<policy_result_t<Policy>, step_result>
               : (!std::is_reference_v<R> &&
                  std::is_convertible_v<policy_result_t<Policy>, R>))> {};

template <typename R, typename Policy>
inline constexpr bool catch_policy_supported_v =
    catch_policy_supported<R, Policy>::value;

}  // namespace yorch::detail
