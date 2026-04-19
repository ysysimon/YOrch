#pragma once

#include <tuple>
#include <type_traits>

#include "common.hpp"
#include "../traits.hpp"

namespace yorch::detail {

enum class bind_error {
    ok,
    member_callable_not_supported,
    callable_shape_invalid,
    arity_mismatch,
};

enum class bind_into_error {
    ok,
    invalid_output_type,
    member_callable_not_supported,
    callable_shape_invalid,
    missing_output_parameter,
    arity_mismatch,
    last_parameter_not_direct_out,
};

enum class bind_member_error {
    ok,
    callable_not_member,
    direct_output_member_not_supported,
    arity_mismatch,
};

enum class bind_into_member_error {
    ok,
    invalid_output_type,
    callable_not_member,
    missing_output_parameter,
    arity_mismatch,
    last_parameter_not_direct_out,
};

template <typename F, typename... Specs>
[[nodiscard]] consteval bind_error validate_bind() {
    using fn_t = std::remove_cvref_t<F>;

    if constexpr (member_bind_callable<fn_t>) {
        return bind_error::member_callable_not_supported;
    } else if constexpr (!ordinary_bind_callable<fn_t>) {
        return bind_error::callable_shape_invalid;
    } else if constexpr (sizeof...(Specs) != function_traits<fn_t>::arity) {
        return bind_error::arity_mismatch;
    } else {
        return bind_error::ok;
    }
}

template <typename F, typename... Specs>
consteval void emit_bind_diagnostic() {
    constexpr auto error = validate_bind<F, Specs...>();
    using diagnostic_t = std::tuple<
        std::type_identity<std::remove_cvref_t<F>>,
        std::type_identity<std::remove_cvref_t<Specs>>...>;

    if constexpr (error == bind_error::ok) {
        return;
    } else if constexpr (error == bind_error::member_callable_not_supported) {
        static_assert(bind_tasks_always_false_v<diagnostic_t>,
                      "bind(...) does not accept member function pointers; use bind_member(...) instead");
    } else if constexpr (error == bind_error::callable_shape_invalid) {
        static_assert(bind_tasks_always_false_v<diagnostic_t>,
                      "bind(...) requires a callable with one non-overloaded concrete signature");
    } else {
        static_assert(bind_tasks_always_false_v<diagnostic_t>,
                      "bind(...) requires exactly one spec per function parameter");
    }
}

template <typename T, typename F, typename... Specs>
[[nodiscard]] consteval bind_into_error validate_bind_into() {
    using fn_t = std::remove_cvref_t<F>;

    if constexpr (std::is_reference_v<T> || std::is_void_v<T>) {
        return bind_into_error::invalid_output_type;
    } else if constexpr (member_bind_callable<fn_t>) {
        return bind_into_error::member_callable_not_supported;
    } else if constexpr (!bind_callable<fn_t>) {
        return bind_into_error::callable_shape_invalid;
    } else if constexpr (function_traits<fn_t>::arity == 0) {
        return bind_into_error::missing_output_parameter;
    } else if constexpr (sizeof...(Specs) + 1 != function_traits<fn_t>::arity) {
        return bind_into_error::arity_mismatch;
    } else if constexpr (!std::is_same_v<
                             std::remove_cvref_t<last_arg_t<fn_t>>,
                             direct_out<T>>) {
        return bind_into_error::last_parameter_not_direct_out;
    } else {
        return bind_into_error::ok;
    }
}

template <typename T, typename F, typename... Specs>
consteval void emit_bind_into_diagnostic() {
    constexpr auto error = validate_bind_into<T, F, Specs...>();
    using diagnostic_t = std::tuple<
        std::type_identity<T>,
        std::type_identity<std::remove_cvref_t<F>>,
        std::type_identity<std::remove_cvref_t<Specs>>...>;

    if constexpr (error == bind_into_error::ok) {
        return;
    } else if constexpr (error == bind_into_error::invalid_output_type) {
        static_assert(bind_tasks_always_false_v<diagnostic_t>,
                      "bind_into<T>(...) requires a non-void owned payload type T");
    } else if constexpr (error == bind_into_error::member_callable_not_supported) {
        static_assert(bind_tasks_always_false_v<diagnostic_t>,
                      "bind_into(...) does not accept member function pointers; use bind_into_member(...) instead");
    } else if constexpr (error == bind_into_error::callable_shape_invalid) {
        static_assert(bind_tasks_always_false_v<diagnostic_t>,
                      "bind_into(...) requires a callable with one non-overloaded concrete signature");
    } else if constexpr (error == bind_into_error::missing_output_parameter) {
        static_assert(bind_tasks_always_false_v<diagnostic_t>,
                      "bind_into(...) requires a callable whose last parameter is yorch::direct_out<T>");
    } else if constexpr (error == bind_into_error::arity_mismatch) {
        static_assert(bind_tasks_always_false_v<diagnostic_t>,
                      "bind_into(...) requires exactly one spec per non-output function parameter");
    } else {
        static_assert(bind_tasks_always_false_v<diagnostic_t>,
                      "bind_into(...) callable must take yorch::direct_out<T> as its last parameter");
    }
}

template <typename F, typename ReceiverSpec, typename... Specs>
[[nodiscard]] consteval bind_member_error validate_bind_member() {
    using fn_t = std::remove_cvref_t<F>;

    if constexpr (!member_bind_callable<fn_t>) {
        return bind_member_error::callable_not_member;
    } else if constexpr (!ordinary_member_bind_callable<fn_t>) {
        return bind_member_error::direct_output_member_not_supported;
    } else if constexpr (sizeof...(Specs) != member_function_traits<fn_t>::arity) {
        return bind_member_error::arity_mismatch;
    } else {
        return bind_member_error::ok;
    }
}

template <typename F, typename ReceiverSpec, typename... Specs>
consteval void emit_bind_member_diagnostic() {
    constexpr auto error = validate_bind_member<F, ReceiverSpec, Specs...>();
    using diagnostic_t = std::tuple<
        std::type_identity<std::remove_cvref_t<F>>,
        std::type_identity<std::remove_cvref_t<ReceiverSpec>>,
        std::type_identity<std::remove_cvref_t<Specs>>...>;

    if constexpr (error == bind_member_error::ok) {
        return;
    } else if constexpr (error == bind_member_error::callable_not_member) {
        static_assert(bind_tasks_always_false_v<diagnostic_t>,
                      "bind_member(...) requires a non-static member function pointer");
    } else if constexpr (error == bind_member_error::direct_output_member_not_supported) {
        static_assert(bind_tasks_always_false_v<diagnostic_t>,
                      "bind_member(...) does not accept direct-output member functions; use bind_into_member(...) instead");
    } else {
        static_assert(bind_tasks_always_false_v<diagnostic_t>,
                      "bind_member(...) requires one receiver binding plus exactly one spec per member-function parameter");
    }
}

template <typename T, typename F, typename ReceiverSpec, typename... Specs>
[[nodiscard]] consteval bind_into_member_error validate_bind_into_member() {
    using fn_t = std::remove_cvref_t<F>;

    if constexpr (std::is_reference_v<T> || std::is_void_v<T>) {
        return bind_into_member_error::invalid_output_type;
    } else if constexpr (!member_bind_callable<fn_t>) {
        return bind_into_member_error::callable_not_member;
    } else if constexpr (member_function_traits<fn_t>::arity == 0) {
        return bind_into_member_error::missing_output_parameter;
    } else if constexpr (sizeof...(Specs) + 1 != member_function_traits<fn_t>::arity) {
        return bind_into_member_error::arity_mismatch;
    } else if constexpr (!std::is_same_v<
                             std::remove_cvref_t<member_last_arg_t<fn_t>>,
                             direct_out<T>>) {
        return bind_into_member_error::last_parameter_not_direct_out;
    } else {
        return bind_into_member_error::ok;
    }
}

template <typename T, typename F, typename ReceiverSpec, typename... Specs>
consteval void emit_bind_into_member_diagnostic() {
    constexpr auto error = validate_bind_into_member<T, F, ReceiverSpec, Specs...>();
    using diagnostic_t = std::tuple<
        std::type_identity<T>,
        std::type_identity<std::remove_cvref_t<F>>,
        std::type_identity<std::remove_cvref_t<ReceiverSpec>>,
        std::type_identity<std::remove_cvref_t<Specs>>...>;

    if constexpr (error == bind_into_member_error::ok) {
        return;
    } else if constexpr (error == bind_into_member_error::invalid_output_type) {
        static_assert(bind_tasks_always_false_v<diagnostic_t>,
                      "bind_into_member<T>(...) requires a non-void owned payload type T");
    } else if constexpr (error == bind_into_member_error::callable_not_member) {
        static_assert(bind_tasks_always_false_v<diagnostic_t>,
                      "bind_into_member(...) requires a non-static member function pointer");
    } else if constexpr (error == bind_into_member_error::missing_output_parameter) {
        static_assert(bind_tasks_always_false_v<diagnostic_t>,
                      "bind_into_member(...) requires a member function whose last parameter is yorch::direct_out<T>");
    } else if constexpr (error == bind_into_member_error::arity_mismatch) {
        static_assert(bind_tasks_always_false_v<diagnostic_t>,
                      "bind_into_member(...) requires one receiver binding plus exactly one spec per non-output member-function parameter");
    } else {
        static_assert(bind_tasks_always_false_v<diagnostic_t>,
                      "bind_into_member(...) callable must take yorch::direct_out<T> as its last parameter");
    }
}

} // namespace yorch::detail
