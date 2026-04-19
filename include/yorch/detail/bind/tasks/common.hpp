#pragma once

#include <type_traits>

namespace yorch::detail {

template <typename>
inline constexpr bool bind_tasks_always_false_v = false;

template <typename Task>
concept bind_task_object =
    requires {
        typename std::remove_cvref_t<Task>::raw_result_type;
    };

} // namespace yorch::detail
