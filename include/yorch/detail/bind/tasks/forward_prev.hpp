#pragma once

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "../../../task_adapters.hpp"
#include "../../executor/prev_access_specs.hpp"
#include "../traits.hpp"
#include "common.hpp"

namespace yorch::detail {

template <typename... Specs>
inline constexpr std::size_t forward_prev_prev_access_count_v =
    (0U + ... + (is_prev_access_spec_v<std::remove_cvref_t<Specs>> ? 1U : 0U));

enum class bind_forward_prev_error {
    ok,
    invalid_output_type,
    member_callable_not_supported,
    direct_output_callable_not_supported,
    callable_shape_invalid,
    arity_mismatch,
    invalid_result_type,
    prev_access_count_invalid,
    payload_type_mismatch,
    consume_by_value_not_supported,
    binding_mode_not_supported,
};

enum class bind_forward_prev_member_error {
    ok,
    invalid_output_type,
    callable_not_member,
    direct_output_member_not_supported,
    arity_mismatch,
    invalid_result_type,
    prev_access_count_invalid,
    payload_type_mismatch,
    consume_by_value_not_supported,
    binding_mode_not_supported,
};

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

template <typename T, typename F, typename ReceiverSpec>
inline constexpr bool forward_prev_member_receiver_binding_supported_v =
    (!is_prev_access_spec_v<std::remove_cvref_t<ReceiverSpec>>) ||
    ((is_borrow_prev_mut_spec_v<std::remove_cvref_t<ReceiverSpec>> ||
      is_consume_prev_spec_v<std::remove_cvref_t<ReceiverSpec>>) &&
     std::is_same_v<
         typename normalized_prev_access_spec_traits<std::remove_cvref_t<ReceiverSpec>>::payload_type,
         T> &&
     member_receiver_prev_access_valid_v<std::remove_cvref_t<ReceiverSpec>, F>);

template <typename T, typename F, typename ReceiverSpec, typename... Specs, std::size_t... I>
[[nodiscard]] consteval bool forward_prev_member_bindings_supported_impl(std::index_sequence<I...>) {
    return forward_prev_member_receiver_binding_supported_v<T, F, ReceiverSpec> &&
           (((!is_prev_access_spec_v<std::remove_cvref_t<Specs>>) ||
             forward_prev_spec_matches_binding_v<
                 T,
                 member_nth_arg_t<I, std::remove_cvref_t<F>>,
                 std::remove_cvref_t<Specs>>) &&
            ...);
}

template <typename T, typename F, typename... Specs, std::size_t... I>
[[nodiscard]] consteval bool forward_prev_member_consume_by_value_requested_impl(std::index_sequence<I...>) {
    return ((is_prev_access_spec_v<std::remove_cvref_t<Specs>> &&
             forward_prev_consume_bound_to_value_v<
                 T,
                 member_nth_arg_t<I, std::remove_cvref_t<F>>,
                 std::remove_cvref_t<Specs>>) ||
            ...);
}

template <typename T, typename F, typename ReceiverSpec, typename... Specs>
inline constexpr bool bind_forward_prev_member_payload_matches_v =
    std::is_same_v<T, forward_prev_unique_prev_payload_t<ReceiverSpec, Specs...>>;

template <typename T, typename F, typename ReceiverSpec, typename... Specs>
inline constexpr bool bind_forward_prev_member_bindings_supported_v =
    forward_prev_member_bindings_supported_impl<T, F, ReceiverSpec, Specs...>(
        std::index_sequence_for<Specs...> {});

template <typename T, typename F, typename ReceiverSpec, typename... Specs>
inline constexpr bool bind_forward_prev_member_consume_by_value_requested_v =
    forward_prev_member_consume_by_value_requested_impl<T, F, Specs...>(
        std::index_sequence_for<Specs...> {});

template <typename T, typename F, typename... Specs>
[[nodiscard]] consteval bind_forward_prev_error validate_bind_forward_prev() {
    using fn_t = std::remove_cvref_t<F>;

    if constexpr (std::is_reference_v<T> || std::is_void_v<T>) {
        return bind_forward_prev_error::invalid_output_type;
    } else if constexpr (member_bind_callable<fn_t>) {
        return bind_forward_prev_error::member_callable_not_supported;
    } else if constexpr (inferable_direct_output_callable<fn_t>) {
        return bind_forward_prev_error::direct_output_callable_not_supported;
    } else if constexpr (!ordinary_bind_callable<fn_t>) {
        return bind_forward_prev_error::callable_shape_invalid;
    } else if constexpr (sizeof...(Specs) != function_traits<fn_t>::arity) {
        return bind_forward_prev_error::arity_mismatch;
    } else if constexpr (!(std::is_void_v<result_t<fn_t>> ||
                           std::is_same_v<result_t<fn_t>, step_result>)) {
        return bind_forward_prev_error::invalid_result_type;
    } else if constexpr (forward_prev_prev_access_count_v<std::decay_t<Specs>...> != 1) {
        return bind_forward_prev_error::prev_access_count_invalid;
    } else if constexpr (!bind_forward_prev_payload_matches_v<T, fn_t, std::decay_t<Specs>...>) {
        return bind_forward_prev_error::payload_type_mismatch;
    } else if constexpr (bind_forward_prev_consume_by_value_requested_v<T, fn_t, std::decay_t<Specs>...>) {
        return bind_forward_prev_error::consume_by_value_not_supported;
    } else if constexpr (!bind_forward_prev_bindings_supported_v<T, fn_t, std::decay_t<Specs>...>) {
        return bind_forward_prev_error::binding_mode_not_supported;
    } else {
        return bind_forward_prev_error::ok;
    }
}

template <typename T, typename F, typename... Specs>
consteval void emit_bind_forward_prev_diagnostic() {
    constexpr auto error = validate_bind_forward_prev<T, F, Specs...>();
    using diagnostic_t = std::tuple<
        std::type_identity<T>,
        std::type_identity<std::remove_cvref_t<F>>,
        std::type_identity<std::remove_cvref_t<Specs>>...>;

    if constexpr (error == bind_forward_prev_error::ok) {
        return;
    } else if constexpr (error == bind_forward_prev_error::invalid_output_type) {
        static_assert(bind_tasks_always_false_v<diagnostic_t>,
                      "bind_forward_prev<T>(...) requires a non-void owned payload type T");
    } else if constexpr (error == bind_forward_prev_error::member_callable_not_supported) {
        static_assert(bind_tasks_always_false_v<diagnostic_t>,
                      "bind_forward_prev<T>(...) does not support member function pointers in v1");
    } else if constexpr (error == bind_forward_prev_error::direct_output_callable_not_supported) {
        static_assert(bind_tasks_always_false_v<diagnostic_t>,
                      "bind_forward_prev<T>(...) does not accept callables with yorch::direct_out<T>; use bind_into(...) for direct-output materialization");
    } else if constexpr (error == bind_forward_prev_error::callable_shape_invalid) {
        static_assert(bind_tasks_always_false_v<diagnostic_t>,
                      "bind_forward_prev<T>(...) requires a callable with one non-overloaded concrete signature");
    } else if constexpr (error == bind_forward_prev_error::arity_mismatch) {
        static_assert(bind_tasks_always_false_v<diagnostic_t>,
                      "bind_forward_prev<T>(...) requires exactly one spec per function parameter");
    } else if constexpr (error == bind_forward_prev_error::invalid_result_type) {
        static_assert(bind_tasks_always_false_v<diagnostic_t>,
                      "bind_forward_prev<T>(...) callable must return void or yorch::step_result");
    } else if constexpr (error == bind_forward_prev_error::prev_access_count_invalid) {
        static_assert(bind_tasks_always_false_v<diagnostic_t>,
                      "bind_forward_prev<T>(...) requires exactly one prev-access binding");
    } else if constexpr (error == bind_forward_prev_error::payload_type_mismatch) {
        static_assert(bind_tasks_always_false_v<diagnostic_t>,
                      "bind_forward_prev<T>(...) requires the forwarded prev payload type to match T exactly");
    } else if constexpr (error == bind_forward_prev_error::consume_by_value_not_supported) {
        static_assert(bind_tasks_always_false_v<diagnostic_t>,
                      "bind_forward_prev<T>(...) does not support consume_prev<T>() bound to T; use T&& if you want to forward the direct parent object identity");
    } else {
        static_assert(bind_tasks_always_false_v<diagnostic_t>,
                      "bind_forward_prev<T>(...) only supports borrow_prev_mut<T>() -> T& or consume_prev<T>() -> T&&");
    }
}

template <typename T, typename F, typename ReceiverSpec, typename... Specs>
[[nodiscard]] consteval bind_forward_prev_member_error validate_bind_forward_prev_member() {
    using fn_t = std::remove_cvref_t<F>;

    if constexpr (std::is_reference_v<T> || std::is_void_v<T>) {
        return bind_forward_prev_member_error::invalid_output_type;
    } else if constexpr (!member_bind_callable<fn_t>) {
        return bind_forward_prev_member_error::callable_not_member;
    } else if constexpr (!ordinary_member_bind_callable<fn_t>) {
        return bind_forward_prev_member_error::direct_output_member_not_supported;
    } else if constexpr (sizeof...(Specs) != member_function_traits<fn_t>::arity) {
        return bind_forward_prev_member_error::arity_mismatch;
    } else if constexpr (!(std::is_void_v<member_result_t<fn_t>> ||
                           std::is_same_v<member_result_t<fn_t>, step_result>)) {
        return bind_forward_prev_member_error::invalid_result_type;
    } else if constexpr (forward_prev_prev_access_count_v<std::decay_t<ReceiverSpec>, std::decay_t<Specs>...> != 1) {
        return bind_forward_prev_member_error::prev_access_count_invalid;
    } else if constexpr (!bind_forward_prev_member_payload_matches_v<
                             T,
                             fn_t,
                             std::decay_t<ReceiverSpec>,
                             std::decay_t<Specs>...>) {
        return bind_forward_prev_member_error::payload_type_mismatch;
    } else if constexpr (bind_forward_prev_member_consume_by_value_requested_v<
                             T,
                             fn_t,
                             std::decay_t<ReceiverSpec>,
                             std::decay_t<Specs>...>) {
        return bind_forward_prev_member_error::consume_by_value_not_supported;
    } else if constexpr (!bind_forward_prev_member_bindings_supported_v<
                             T,
                             fn_t,
                             std::decay_t<ReceiverSpec>,
                             std::decay_t<Specs>...>) {
        return bind_forward_prev_member_error::binding_mode_not_supported;
    } else {
        return bind_forward_prev_member_error::ok;
    }
}

template <typename T, typename F, typename ReceiverSpec, typename... Specs>
consteval void emit_bind_forward_prev_member_diagnostic() {
    constexpr auto error = validate_bind_forward_prev_member<T, F, ReceiverSpec, Specs...>();
    using diagnostic_t = std::tuple<
        std::type_identity<T>,
        std::type_identity<std::remove_cvref_t<F>>,
        std::type_identity<std::remove_cvref_t<ReceiverSpec>>,
        std::type_identity<std::remove_cvref_t<Specs>>...>;

    if constexpr (error == bind_forward_prev_member_error::ok) {
        return;
    } else if constexpr (error == bind_forward_prev_member_error::invalid_output_type) {
        static_assert(bind_tasks_always_false_v<diagnostic_t>,
                      "bind_forward_prev_member<T>(...) requires a non-void owned payload type T");
    } else if constexpr (error == bind_forward_prev_member_error::callable_not_member) {
        static_assert(bind_tasks_always_false_v<diagnostic_t>,
                      "bind_forward_prev_member(...) requires a non-static member function pointer");
    } else if constexpr (error == bind_forward_prev_member_error::direct_output_member_not_supported) {
        static_assert(bind_tasks_always_false_v<diagnostic_t>,
                      "bind_forward_prev_member(...) does not accept direct-output member functions; use bind_into_member(...) instead");
    } else if constexpr (error == bind_forward_prev_member_error::arity_mismatch) {
        static_assert(bind_tasks_always_false_v<diagnostic_t>,
                      "bind_forward_prev_member(...) requires one receiver binding plus exactly one spec per member-function parameter");
    } else if constexpr (error == bind_forward_prev_member_error::invalid_result_type) {
        static_assert(bind_tasks_always_false_v<diagnostic_t>,
                      "bind_forward_prev_member(...) callable must return void or yorch::step_result");
    } else if constexpr (error == bind_forward_prev_member_error::prev_access_count_invalid) {
        static_assert(bind_tasks_always_false_v<diagnostic_t>,
                      "bind_forward_prev_member<T>(...) requires exactly one prev-access binding across receiver and member-function parameters");
    } else if constexpr (error == bind_forward_prev_member_error::payload_type_mismatch) {
        static_assert(bind_tasks_always_false_v<diagnostic_t>,
                      "bind_forward_prev_member<T>(...) requires the forwarded prev payload type to match T exactly");
    } else if constexpr (error == bind_forward_prev_member_error::consume_by_value_not_supported) {
        static_assert(bind_tasks_always_false_v<diagnostic_t>,
                      "bind_forward_prev_member<T>(...) does not support consume_prev<T>() bound to T; use T&& if you want to forward the direct parent object identity");
    } else {
        static_assert(bind_tasks_always_false_v<diagnostic_t>,
                      "bind_forward_prev_member<T>(...) only supports borrow_prev_mut<T>() -> T& or consume_prev<T>() -> T&&");
    }
}

} // namespace yorch::detail
