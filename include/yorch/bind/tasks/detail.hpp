#pragma once

#include <cstddef>
#include <exception>
#include <type_traits>
#include <utility>

#include "../../task_adapters.hpp"
#include "../../detail/executor/prev_access_specs.hpp"
#include "../../detail/bind/traits.hpp"

namespace yorch::detail {

template <typename>
inline constexpr bool bind_tasks_always_false_v = false;

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

} // namespace yorch::detail
