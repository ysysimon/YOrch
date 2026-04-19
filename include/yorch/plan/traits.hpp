#pragma once

#include <type_traits>

#include "../detail/slots/policy.hpp"
#include "../detail/task/traits.hpp"
#include "../result.hpp"

namespace yorch::detail {

/**
 * @brief Compile-time trait that extracts a task's raw return type.
 *
 * This is a low-level `plan` trait whose job is to answer "what raw type does
 * this task expose through `invoke_raw(...)`?" before any later payload
 * normalization is applied.
 *
 * The primary template intentionally provides no `type`. Concrete behavior is
 * supplied by opt-in specializations or, in the common case, by tasks that
 * declare `raw_result_type` directly.
 *
 * @tparam Task Task type being analyzed.
 * @tparam SFINAE hook used for detection-based specialization.
 */
template <typename Task, typename = void>
struct task_raw_result;

/**
 * @brief Uses a task's own `raw_result_type` declaration as the raw return.
 *
 * This is the base case for `task_raw_result`. Tasks such as `bound_task`
 * already expose a canonical `raw_result_type`, so no wrapper unwrapping is
 * required and that declared type becomes the answer directly.
 *
 * `std::remove_cvref_t` strips const/reference qualifiers so detection always
 * targets the normalized task object type.
 *
 * @tparam Task Task type being analyzed.
 */
template <typename Task>
struct task_raw_result<Task, std::void_t<typename std::remove_cvref_t<Task>::raw_result_type>> {
    using type = typename std::remove_cvref_t<Task>::raw_result_type;
};

/**
 * @brief Convenience alias for a task's raw return type.
 *
 * This avoids repeatedly spelling `typename task_raw_result<T>::type`.
 *
 * @tparam Task Task type being analyzed.
 */
template <typename Task>
using task_raw_result_t = typename task_raw_result<Task>::type;

template <typename Task>
concept plannable_task =
    requires {
        typename task_raw_result<std::remove_cvref_t<Task>>::type;
    };

template <typename Task, typename = void>
struct plan_declared_task_output;

template <typename Task>
struct plan_declared_task_output<
    Task,
    std::void_t<typename std::remove_cvref_t<Task>::output_type>
> {
    using type = typename std::remove_cvref_t<Task>::output_type;
};

template <typename Task, typename = void>
struct has_plan_declared_task_output : std::false_type {};

template <typename Task>
struct has_plan_declared_task_output<
    Task,
    std::void_t<typename std::remove_cvref_t<Task>::output_type>
> : std::true_type {};

template <typename R>
struct task_output_type_impl {
    using type = R;
};

template <>
struct task_output_type_impl<step_result> {
    using type = void;
};

template <typename T>
struct task_output_type_impl<task_result<T>> {
    using type = T;
};

template <typename R>
struct task_output_type {
    static_assert(!std::is_reference_v<R>,
                  "Task raw return references are not supported as parent payloads");

    using type = typename task_output_type_impl<std::remove_cv_t<R>>::type;
};

template <>
struct task_output_type<void> {
    using type = void;
};

template <typename R>
using task_output_t = typename task_output_type<R>::type;

template <typename Task, typename = void>
struct task_output_for {
    using type = task_output_t<task_raw_result_t<Task>>;
};

template <typename Task>
struct task_output_for<
    Task,
    std::void_t<typename plan_declared_task_output<Task>::type>
> {
    using type = typename plan_declared_task_output<Task>::type;
};

template <typename Task>
using task_output_for_t = typename task_output_for<Task>::type;

template <typename Task, typename = void>
struct task_output_storage_mode {
private:
    using raw_t = task_raw_result_t<Task>;

public:
    static constexpr detail::output_storage_mode value =
        std::is_void_v<raw_t> ||
                std::is_same_v<raw_t, step_result>
            ? detail::output_storage_mode::none
            : detail::output_storage_mode::owned;
};

template <typename Task>
struct task_output_storage_mode<
    Task,
    std::void_t<typename plan_declared_task_output<Task>::type>
> {
    static constexpr detail::output_storage_mode value =
        task_uses_forward_prev_output_protocol_v<Task>
            ? detail::output_storage_mode::forwarded_prev
            : detail::output_storage_mode::owned;
};

template <typename Task>
inline constexpr detail::output_storage_mode task_output_storage_mode_v =
    task_output_storage_mode<Task>::value;

template <typename Task, typename = void>
struct task_slot_logical_policy {
private:
    using raw_t = task_raw_result_t<Task>;

public:
    static constexpr slot_logical_policy value =
        std::is_void_v<raw_t> ||
                std::is_same_v<raw_t, step_result>
            ? slot_logical_policy::none
        : is_task_result_v<raw_t>
            ? slot_logical_policy::maybe_payload
            : slot_logical_policy::must_payload;
};

/**
 * @brief Direct-output tasks always use maybe-payload slot semantics.
 *
 * A declared `output_type` means the payload is written through a sink such as
 * `direct_out<T>` instead of being returned as the raw result value. Receiving
 * that sink does not prove the task will call `emplace(...)`; the task may
 * return `failure`, `retry`, `abort_execution`, or simply return success without
 * materializing a value. The slot therefore needs an engagement bit even though
 * the declared payload type itself is non-void.
 */
template <typename Task>
struct task_slot_logical_policy<
    Task,
    std::void_t<typename plan_declared_task_output<Task>::type>
> {
    static constexpr slot_logical_policy value =
        task_uses_forward_prev_output_protocol_v<Task>
            ? slot_logical_policy::none
            : slot_logical_policy::maybe_payload;
};

template <typename Task>
inline constexpr slot_logical_policy task_slot_logical_policy_v =
    task_slot_logical_policy<Task>::value;

template <typename Node>
concept plannable_plan_node =
    requires {
        typename Node::task_type;
    } &&
    plannable_task<typename Node::task_type>;

template <typename... Nodes>
concept plannable_plan_nodes =
    (plannable_plan_node<Nodes> && ...);

} // namespace yorch::detail
