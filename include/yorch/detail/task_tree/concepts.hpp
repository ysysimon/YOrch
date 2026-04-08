#pragma once

#include <concepts> // IWYU pragma: keep
#include <cstddef>
#include <exception>
#include <type_traits>

#include "../../bind.hpp"
#include "../../task_adapters.hpp" // IWYU pragma: keep

namespace yorch::detail {

template <typename F>
concept bind_callable =
    requires {
        { function_traits<std::remove_cvref_t<F>>::arity } -> std::convertible_to<std::size_t>;
    };

template <typename F, typename... Specs>
concept bind_signature_matches =
    bind_callable<F> &&
    function_traits<std::remove_cvref_t<F>>::arity == sizeof...(Specs);

template <typename T, typename F, typename... Specs>
concept bind_into_signature_matches =
    bind_callable<F> &&
    function_traits<std::remove_cvref_t<F>>::arity > 0 &&
    function_traits<std::remove_cvref_t<F>>::arity == sizeof...(Specs) + 1 &&
    std::is_same_v<
        std::remove_cvref_t<last_arg_t<std::remove_cvref_t<F>>>,
        result_out<T>>;

template <typename Policy>
concept catch_policy_like =
    requires(Policy& policy, const std::exception_ptr& ep) {
        { policy(ep) } noexcept;
    };

} // namespace yorch::detail
