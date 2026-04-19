#pragma once

#include <type_traits>

namespace yorch::detail {

struct no_declared_output_protocol_tag {};
struct direct_output_protocol_tag {};
struct forward_prev_output_protocol_tag {};

/**
 * @brief Detects a task's declared logical output type.
 *
 * This trait answers only whether the task exposes an `output_type` contract,
 * meaning the task participates in an explicit output protocol instead of
 * deriving its payload type solely from `raw_result_type`.
 *
 * Important: a declared logical output type no longer implies that the task
 * owns physical slot storage. For example, `forward_prev` tasks declare an
 * `output_type` so downstream nodes can see the logical payload type, but
 * their payload is statically forwarded from the direct parent rather than
 * materialized into a new slot. Physical storage decisions therefore belong to
 * separate traits such as plan/output storage mode, not to
 * `declared_task_output` itself.
 *
 * @tparam Task Task type being inspected.
 * @tparam SFINAE hook used for detection-based specialization.
 */
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

template <typename Task, typename = void>
struct declared_task_output_protocol {
    using type = no_declared_output_protocol_tag;
};

template <typename Task>
struct declared_task_output_protocol<
    Task,
    std::void_t<typename std::remove_cvref_t<Task>::output_protocol>
> {
    using type = typename std::remove_cvref_t<Task>::output_protocol;
};

template <typename Task>
using declared_task_output_protocol_t =
    typename declared_task_output_protocol<Task>::type;

template <typename Task>
inline constexpr bool task_uses_direct_output_protocol_v =
    std::is_same_v<
        declared_task_output_protocol_t<Task>,
        direct_output_protocol_tag>;

template <typename Task>
inline constexpr bool task_uses_forward_prev_output_protocol_v =
    std::is_same_v<
        declared_task_output_protocol_t<Task>,
        forward_prev_output_protocol_tag>;

template <typename Task, typename = void>
struct forwarded_task_output_base {};

template <typename Task>
struct forwarded_task_output_base<
    Task,
    std::void_t<declared_task_output_t<Task>>
> {
    using output_type = declared_task_output_t<Task>;
};

template <typename Task, typename = void>
struct forwarded_task_output_protocol_base {};

template <typename Task>
struct forwarded_task_output_protocol_base<
    Task,
    std::void_t<typename std::remove_cvref_t<Task>::output_protocol>
> {
    using output_protocol = declared_task_output_protocol_t<Task>;
};

}  // namespace yorch::detail
