#pragma once

#include <tuple>
#include <type_traits>
#include <utility>

#include "tasks.hpp"

namespace yorch {

struct catch_as_failure_adapter_desc {};

template <typename Policy>
struct catch_as_failure_with_policy_adapter_desc {
    Policy policy;
};

template <typename Policy>
struct retry_adapter_desc {
    Policy policy;
};

template <typename... Descs>
struct adapter_chain {
    std::tuple<Descs...> descriptors;
};

template <typename Policy>
constexpr auto adapt_retry(Policy&& policy)
    requires retry_policy<std::decay_t<Policy>> {
    return retry_adapter_desc<std::decay_t<Policy>> {
        std::forward<Policy>(policy)
    };
}

[[nodiscard]] constexpr auto adapt_catch_as_failure() noexcept {
    return catch_as_failure_adapter_desc {};
}

template <typename Policy>
constexpr auto adapt_catch_as_failure(Policy&& policy)
    requires detail::catch_policy_like<std::remove_cvref_t<Policy>> {
    return catch_as_failure_with_policy_adapter_desc<std::decay_t<Policy>> {
        std::forward<Policy>(policy)
    };
}

namespace detail {

template <typename Desc>
struct is_adapter_descriptor : std::false_type {};

template <>
struct is_adapter_descriptor<catch_as_failure_adapter_desc> : std::true_type {};

template <typename Policy>
struct is_adapter_descriptor<catch_as_failure_with_policy_adapter_desc<Policy>> : std::true_type {};

template <typename Policy>
struct is_adapter_descriptor<retry_adapter_desc<Policy>> : std::true_type {};

template <typename Desc>
inline constexpr bool is_adapter_descriptor_v =
    is_adapter_descriptor<std::remove_cvref_t<Desc>>::value;

template <typename Desc>
concept adapter_descriptor =
    is_adapter_descriptor_v<Desc>;

template <typename T>
struct is_adapter_chain : std::false_type {};

template <typename... Descs>
struct is_adapter_chain<adapter_chain<Descs...>> : std::true_type {};

template <typename T>
inline constexpr bool is_adapter_chain_v =
    is_adapter_chain<std::remove_cvref_t<T>>::value;

template <typename T>
concept adapter_chain_like =
    is_adapter_chain_v<T>;

template <std::size_t I, typename Task, typename Tuple>
constexpr auto apply_adapters_from_const(Task&& task, const Tuple& descriptors) {
    if constexpr (I == 0) {
        return std::forward<Task>(task);
    } else {
        return apply_adapter(
            std::get<I - 1>(descriptors),
            apply_adapters_from_const<I - 1>(std::forward<Task>(task), descriptors));
    }
}

template <std::size_t I, typename Task, typename Tuple>
constexpr auto apply_adapters_from_mut(Task&& task, Tuple&& descriptors) {
    if constexpr (I == 0) {
        return std::forward<Task>(task);
    } else {
        return apply_adapter(
            std::move(std::get<I - 1>(descriptors)),
            apply_adapters_from_mut<I - 1>(
                std::forward<Task>(task),
                std::forward<Tuple>(descriptors)));
    }
}

} // namespace detail

template <typename... Descs>
    requires (detail::adapter_descriptor<Descs> && ...)
constexpr auto adapters(Descs&&... descs) {
    return adapter_chain<std::decay_t<Descs>...> {
        std::tuple<std::decay_t<Descs>...> {
            std::forward<Descs>(descs)...
        }
    };
}

template <typename Task>
constexpr auto apply_adapters(Task&& task, const adapter_chain<>&) {
    return std::forward<Task>(task);
}

template <typename Task, typename... Descs>
constexpr auto apply_adapters(Task&& task, const adapter_chain<Descs...>& chain) {
    return detail::apply_adapters_from_const<sizeof...(Descs)>(
        std::forward<Task>(task),
        chain.descriptors);
}

template <typename Task, typename... Descs>
constexpr auto apply_adapters(Task&& task, adapter_chain<Descs...>&& chain) {
    return detail::apply_adapters_from_mut<sizeof...(Descs)>(
        std::forward<Task>(task),
        std::move(chain).descriptors);
}

template <typename Task, typename Policy>
constexpr auto apply_adapter(retry_adapter_desc<Policy>& desc, Task&& task) {
    return with_retry(std::forward<Task>(task), desc.policy);
}

template <typename Task, typename Policy>
constexpr auto apply_adapter(const retry_adapter_desc<Policy>& desc, Task&& task) {
    return with_retry(std::forward<Task>(task), desc.policy);
}

template <typename Task, typename Policy>
constexpr auto apply_adapter(retry_adapter_desc<Policy>&& desc, Task&& task) {
    return with_retry(std::forward<Task>(task), std::move(desc.policy));
}

template <typename Task>
constexpr auto apply_adapter(catch_as_failure_adapter_desc, Task&& task) {
    return detail::apply_catch_adapter(
        std::forward<Task>(task),
        detail::default_catch_adapter_policy_tag {});
}

template <typename Task, typename Policy>
constexpr auto apply_adapter(catch_as_failure_with_policy_adapter_desc<Policy>& desc, Task&& task) {
    return detail::apply_catch_adapter(std::forward<Task>(task), desc.policy);
}

template <typename Task, typename Policy>
constexpr auto apply_adapter(
    const catch_as_failure_with_policy_adapter_desc<Policy>& desc,
    Task&& task) {
    return detail::apply_catch_adapter(std::forward<Task>(task), desc.policy);
}

template <typename Task, typename Policy>
constexpr auto apply_adapter(
    catch_as_failure_with_policy_adapter_desc<Policy>&& desc,
    Task&& task) {
    return detail::apply_catch_adapter(std::forward<Task>(task), std::move(desc.policy));
}

} // namespace yorch
