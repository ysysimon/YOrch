#pragma once

#include <cstddef>
#include <exception>
#include <type_traits>
#include <utility>

#include "../../task_adapters.hpp"
#include "../../detail/executor/prev_access_specs.hpp"
#include "../../detail/bind/traits.hpp"

namespace yorch::detail {

template <typename Task>
concept bind_task_object =
    requires {
        typename std::remove_cvref_t<Task>::raw_result_type;
    };

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

template <typename... Specs>
inline constexpr std::size_t forward_prev_prev_access_count_v =
    (0U + ... + (is_prev_access_spec_v<std::remove_cvref_t<Specs>> ? 1U : 0U));

template <typename... Specs>
struct forward_prev_unique_prev_payload {
    using type = void;
};

template <typename Spec, typename... Rest>
struct forward_prev_unique_prev_payload<Spec, Rest...> {
private:
    using spec_t = std::remove_cvref_t<Spec>;
    using rest_t = typename forward_prev_unique_prev_payload<Rest...>::type;

public:
    using type = std::conditional_t<
        is_prev_access_spec_v<spec_t>,
        std::conditional_t<
            std::is_void_v<rest_t>,
            typename normalized_prev_access_spec_traits<spec_t>::payload_type,
            void>,
        rest_t>;
};

template <typename... Specs>
using forward_prev_unique_prev_payload_t =
    typename forward_prev_unique_prev_payload<Specs...>::type;

template <typename T, typename Arg, typename Spec>
inline constexpr bool forward_prev_spec_matches_binding_v =
    (is_borrow_prev_mut_spec_v<Spec> &&
     std::is_same_v<typename normalized_prev_access_spec_traits<Spec>::payload_type, T> &&
     std::is_same_v<Arg, T&>) ||
    (is_consume_prev_spec_v<Spec> &&
     std::is_same_v<typename normalized_prev_access_spec_traits<Spec>::payload_type, T> &&
     std::is_same_v<Arg, T&&>);

template <typename T, typename Arg, typename Spec>
inline constexpr bool forward_prev_consume_bound_to_value_v =
    is_consume_prev_spec_v<Spec> &&
    std::is_same_v<typename normalized_prev_access_spec_traits<Spec>::payload_type, T> &&
    std::is_same_v<Arg, T>;

template <typename T, typename F, typename... Specs, std::size_t... I>
[[nodiscard]] consteval bool forward_prev_bindings_supported_impl(std::index_sequence<I...>) {
    return (((!is_prev_access_spec_v<std::remove_cvref_t<Specs>>) ||
             forward_prev_spec_matches_binding_v<
                 T,
                 nth_arg_t<I, std::remove_cvref_t<F>>,
                 std::remove_cvref_t<Specs>>) &&
            ...);
}

template <typename T, typename F, typename... Specs, std::size_t... I>
[[nodiscard]] consteval bool forward_prev_consume_by_value_requested_impl(std::index_sequence<I...>) {
    return ((is_prev_access_spec_v<std::remove_cvref_t<Specs>> &&
             forward_prev_consume_bound_to_value_v<
                 T,
                 nth_arg_t<I, std::remove_cvref_t<F>>,
                 std::remove_cvref_t<Specs>>) ||
            ...);
}

template <typename T, typename F, typename... Specs>
inline constexpr bool bind_forward_prev_payload_matches_v =
    std::is_same_v<T, forward_prev_unique_prev_payload_t<Specs...>>;

template <typename T, typename F, typename... Specs>
inline constexpr bool bind_forward_prev_bindings_supported_v =
    forward_prev_bindings_supported_impl<T, F, Specs...>(
        std::index_sequence_for<Specs...> {});

template <typename T, typename F, typename... Specs>
inline constexpr bool bind_forward_prev_consume_by_value_requested_v =
    forward_prev_consume_by_value_requested_impl<T, F, Specs...>(
        std::index_sequence_for<Specs...> {});

} // namespace yorch::detail
