#pragma once

#include <concepts> // IWYU pragma: keep
#include <cstddef>
#include <tuple>
#include <type_traits>

#include "../../slots/direct_out.hpp"

namespace yorch::detail {

/**
 * @brief Extracts the canonical function signature information of a callable.
 *
 * The primary implementation is based on the normalized form `R(Args...)`.
 * Other specializations strip wrappers such as pointers, references, or
 * member-function qualifiers so `bind(...)` can reason about parameter types
 * in a uniform way.
 *
 * @tparam T Callable or function type to inspect.
 */
template <typename T, typename = void>
struct function_traits;

/**
 * @brief Core traits specialization for a plain function type.
 *
 * @tparam R Return type.
 * @tparam Args Parameter type list.
 */
template <typename R, typename... Args>
struct function_traits<R(Args...)> {
    /// Function return type.
    using result_type = R;

    /// Number of declared parameters.
    static constexpr std::size_t arity = sizeof...(Args);

    /// @brief Alias to the `I`-th parameter type.
    template <std::size_t I>
    using arg = std::tuple_element_t<I, std::tuple<Args...>>;
};

template <typename R, typename... Args>
struct function_traits<R (*)(Args...)> : function_traits<R(Args...)> {};

template <typename R, typename... Args>
struct function_traits<R (*)(Args...) noexcept> : function_traits<R(Args...)> {};

template <typename R, typename... Args>
struct function_traits<R (&)(Args...)> : function_traits<R(Args...)> {};

template <typename R, typename... Args>
struct function_traits<R (&)(Args...) noexcept> : function_traits<R(Args...)> {};

template <typename C, typename R, typename... Args>
struct function_traits<R (C::*)(Args...)> : function_traits<R(Args...)> {};

template <typename C, typename R, typename... Args>
struct function_traits<R (C::*)(Args...) noexcept> : function_traits<R(Args...)> {};

template <typename C, typename R, typename... Args>
struct function_traits<R (C::*)(Args...) const> : function_traits<R(Args...)> {};

template <typename C, typename R, typename... Args>
struct function_traits<R (C::*)(Args...) const noexcept> : function_traits<R(Args...)> {};

template <typename F>
struct function_traits<
    F,
    std::void_t<decltype(&std::remove_reference_t<F>::operator())>>
    : function_traits<decltype(&std::remove_reference_t<F>::operator())> {};

template <typename T>
struct is_member_function_pointer_callable
    : std::bool_constant<std::is_member_function_pointer_v<std::remove_cvref_t<T>>> {};

template <typename T>
inline constexpr bool is_member_function_pointer_callable_v =
    is_member_function_pointer_callable<T>::value;

template <typename T, typename = void>
struct member_function_traits;

template <typename C, typename R, typename... Args>
struct member_function_traits<R (C::*)(Args...)> {
    using class_type = C;
    using receiver_arg_type = C&;
    using result_type = R;

    static constexpr std::size_t arity = sizeof...(Args);

    template <std::size_t I>
    using arg = std::tuple_element_t<I, std::tuple<Args...>>;
};

template <typename C, typename R, typename... Args>
struct member_function_traits<R (C::*)(Args...) noexcept>
    : member_function_traits<R (C::*)(Args...)> {};

template <typename C, typename R, typename... Args>
struct member_function_traits<R (C::*)(Args...) const> {
    using class_type = C;
    using receiver_arg_type = const C&;
    using result_type = R;

    static constexpr std::size_t arity = sizeof...(Args);

    template <std::size_t I>
    using arg = std::tuple_element_t<I, std::tuple<Args...>>;
};

template <typename C, typename R, typename... Args>
struct member_function_traits<R (C::*)(Args...) const noexcept>
    : member_function_traits<R (C::*)(Args...) const> {};

template <std::size_t I, typename F>
using nth_arg_t = typename function_traits<std::remove_cvref_t<F>>::template arg<I>;

template <typename F>
using result_t = typename function_traits<std::remove_cvref_t<F>>::result_type;

template <typename F>
using member_result_t =
    typename member_function_traits<std::remove_cvref_t<F>>::result_type;

template <typename F>
using member_class_t =
    typename member_function_traits<std::remove_cvref_t<F>>::class_type;

template <typename F>
using member_receiver_arg_t =
    typename member_function_traits<std::remove_cvref_t<F>>::receiver_arg_type;

template <typename F>
using member_borrowed_receiver_t =
    std::remove_reference_t<member_receiver_arg_t<F>>;

template <std::size_t I, typename F>
using member_nth_arg_t =
    typename member_function_traits<std::remove_cvref_t<F>>::template arg<I>;

template <typename F>
inline constexpr std::size_t last_arg_index_v =
    function_traits<std::remove_cvref_t<F>>::arity - 1;

template <typename F>
using last_arg_t =
    nth_arg_t<last_arg_index_v<F>, F>;

template <typename F>
inline constexpr std::size_t member_last_arg_index_v =
    member_function_traits<std::remove_cvref_t<F>>::arity - 1;

template <typename F>
using member_last_arg_t =
    member_nth_arg_t<member_last_arg_index_v<F>, F>;

template <typename T>
struct direct_out_arg : std::false_type {};

template <typename T>
struct direct_out_arg<direct_out<T>> : std::true_type {};

template <typename T>
inline constexpr bool direct_out_arg_v =
    direct_out_arg<std::remove_cvref_t<T>>::value;

template <typename T>
concept direct_out_arg_like =
    direct_out_arg_v<T>;

template <typename T, typename = void>
struct direct_out_payload;

template <typename T>
struct direct_out_payload<direct_out<T>, void> {
    using type = T;
};

template <typename T>
using direct_out_payload_t = typename direct_out_payload<T>::type;

template <typename F>
concept bind_callable =
    requires {
        { function_traits<std::remove_cvref_t<F>>::arity } -> std::convertible_to<std::size_t>;
    };

template <typename F>
concept member_bind_callable =
    is_member_function_pointer_callable_v<F> &&
    requires {
        typename member_function_traits<std::remove_cvref_t<F>>::class_type;
    };

template <typename F>
concept inferable_direct_output_callable =
    bind_callable<F> &&
    !member_bind_callable<F> &&
    function_traits<std::remove_cvref_t<F>>::arity > 0 &&
    direct_out_arg_like<last_arg_t<std::remove_cvref_t<F>>>;

template <typename F>
concept ordinary_bind_callable =
    bind_callable<F> &&
    !member_bind_callable<F> &&
    !inferable_direct_output_callable<F>;

template <typename F>
concept inferable_direct_output_member_callable =
    member_bind_callable<F> &&
    member_function_traits<std::remove_cvref_t<F>>::arity > 0 &&
    direct_out_arg_like<member_last_arg_t<std::remove_cvref_t<F>>>;

template <typename F>
concept ordinary_member_bind_callable =
    member_bind_callable<F> &&
    !inferable_direct_output_member_callable<F>;

template <typename F, typename... Specs>
concept bind_signature_matches =
    ordinary_bind_callable<F> &&
    function_traits<std::remove_cvref_t<F>>::arity == sizeof...(Specs);

template <typename F, typename ReceiverSpec>
concept member_receiver_bindable =
    member_bind_callable<F>;

template <typename F, typename... Specs>
concept member_bound_signature_matches =
    ordinary_member_bind_callable<F> &&
    member_function_traits<std::remove_cvref_t<F>>::arity == sizeof...(Specs);

template <typename T, typename F, typename... Specs>
concept bind_into_signature_matches =
    inferable_direct_output_callable<F> &&
    function_traits<std::remove_cvref_t<F>>::arity == sizeof...(Specs) + 1 &&
    std::is_same_v<
        std::remove_cvref_t<last_arg_t<std::remove_cvref_t<F>>>,
        direct_out<T>>;

template <typename T, typename F, typename... Specs>
concept member_bound_into_signature_matches =
    inferable_direct_output_member_callable<F> &&
    member_function_traits<std::remove_cvref_t<F>>::arity == sizeof...(Specs) + 1 &&
    std::is_same_v<
        std::remove_cvref_t<member_last_arg_t<std::remove_cvref_t<F>>>,
        direct_out<T>>;

template <typename F, typename... Specs>
concept inferred_bind_into_signature_matches =
    inferable_direct_output_callable<F> &&
    function_traits<std::remove_cvref_t<F>>::arity == sizeof...(Specs) + 1;

template <typename F, typename... Specs>
concept inferred_member_bound_into_signature_matches =
    inferable_direct_output_member_callable<F> &&
    member_function_traits<std::remove_cvref_t<F>>::arity == sizeof...(Specs) + 1;

} // namespace yorch::detail
