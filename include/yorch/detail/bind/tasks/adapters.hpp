#pragma once

#include <exception>
#include <type_traits>
#include <utility>

#include "../../../task_adapters.hpp"

namespace yorch::detail {

template <typename Policy>
concept catch_policy_like =
    requires(Policy& policy, const std::exception_ptr& ep) {
        { policy(ep) } noexcept;
    };

struct default_catch_adapter_policy_tag {};

template <typename Task, typename PolicyLike>
constexpr auto apply_catch_adapter(Task&& task, PolicyLike&& policy_like) {
    if constexpr (std::is_same_v<std::remove_cvref_t<PolicyLike>, default_catch_adapter_policy_tag>) {
        return catch_as_failure(std::forward<Task>(task));
    } else {
        return catch_as_failure(std::forward<Task>(task), std::forward<PolicyLike>(policy_like));
    }
}

} // namespace yorch::detail
