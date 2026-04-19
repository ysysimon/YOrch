#pragma once

#include <type_traits>

#include "../../bind/adapters.hpp" // IWYU pragma: keep
#include "../../detail/bind/traits.hpp" // IWYU pragma: keep
#include "../task/traits.hpp" // IWYU pragma: keep
#include "../../plan/traits.hpp" // IWYU pragma: keep
#include "../../task_tree/policies.hpp" // IWYU pragma: keep

namespace yorch::detail {

template <typename Task>
concept task_object_argument =
    (!bind_callable<Task>) || plannable_task<std::remove_cvref_t<Task>>;

template <typename F>
concept callable_task_argument =
    bind_callable<F> && !plannable_task<std::remove_cvref_t<F>>;

template <typename Task>
concept direct_output_task_object_argument =
    task_object_argument<Task> &&
    task_uses_direct_output_protocol_v<Task>;

template <typename Task>
concept ordinary_task_object_argument =
    task_object_argument<Task> &&
    !direct_output_task_object_argument<Task>;

template <typename F>
concept ordinary_callable_task_argument =
    callable_task_argument<F> &&
    ordinary_bind_callable<F>;

template <typename F>
concept direct_output_callable_task_argument =
    callable_task_argument<F> &&
    inferable_direct_output_callable<F>;

template <typename F>
concept ordinary_member_callable_task_argument =
    callable_task_argument<F> &&
    ordinary_member_bind_callable<F>;

template <typename F>
concept direct_output_member_callable_task_argument =
    callable_task_argument<F> &&
    inferable_direct_output_member_callable<F>;

template <typename T>
concept fanout_policy_or_chain =
    fanout_policy<T> || adapter_chain_like<T>;

} // namespace yorch::detail
