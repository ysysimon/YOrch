#pragma once
#include <cstddef>
#include <exception>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

#include "context.hpp"
#include "resolve.hpp"
#include "slots.hpp"
#include "task_adapters.hpp"

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

template <typename T>
struct borrowed_member_receiver {
    std::reference_wrapper<T> ref;
};

template <typename T>
struct copied_member_receiver {
    T value;
};

template <typename T>
struct consumed_member_receiver {
    T value;
};

template <typename T>
constexpr decltype(auto) forward_member_receiver(const borrowed_member_receiver<T>& holder) noexcept {
    return holder.ref.get();
}

template <typename T>
constexpr decltype(auto) forward_member_receiver(copied_member_receiver<T>& holder) noexcept {
    return (holder.value);
}

/**
 * Keep this rvalue overload so `invoke_member_with_receiver(...)` can accept a
 * forwarded temporary `copied_member_receiver<T>`.
 *
 * We still return `holder.value` as an lvalue on purpose: `copy_prev<T>()`
 * means "invoke the member function on a local copy", not "move the copied
 * receiver into the member call".
 */
template <typename T>
constexpr decltype(auto) forward_member_receiver
// NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
(copied_member_receiver<T>&& holder) noexcept {
    return (holder.value);
}

template <typename T>
constexpr decltype(auto) forward_member_receiver(consumed_member_receiver<T>& holder) noexcept {
    return std::move(holder.value);
}

template <typename T>
constexpr decltype(auto) forward_member_receiver(consumed_member_receiver<T>&& holder) noexcept {
    return std::move(holder.value);
}

template <typename F, typename Receiver, typename... Args>
constexpr decltype(auto) invoke_member_with_receiver(F&& func, Receiver&& receiver, Args&&... args)
    noexcept(noexcept(std::invoke(
        std::forward<F>(func),
        forward_member_receiver(std::forward<Receiver>(receiver)),
        std::forward<Args>(args)...))) {
    return std::invoke(
        std::forward<F>(func),
        forward_member_receiver(std::forward<Receiver>(receiver)),
        std::forward<Args>(args)...);
}

template <typename F, typename T, typename Ctx, typename Prev>
constexpr auto resolve_member_receiver(copy_prev_t<T> spec, exec_context<Ctx, Prev>& ec)
    noexcept(noexcept(resolve_as<member_class_t<F>>(spec, ec))) {
    using class_t = member_class_t<F>;

    static_assert(std::is_same_v<typename copy_prev_t<T>::type, class_t>,
                  "copy_prev<T>() receiver must match the member-function class type exactly");

    return copied_member_receiver<class_t> {
        resolve_as<class_t>(spec, ec)
    };
}

template <typename F, typename T, typename Ctx, typename Prev>
constexpr auto resolve_member_receiver(consume_prev_t<T> spec, exec_context<Ctx, Prev>& ec)
    noexcept(noexcept(resolve_as<member_class_t<F>>(spec, ec))) {
    using class_t = member_class_t<F>;

    static_assert(std::is_same_v<typename consume_prev_t<T>::type, class_t>,
                  "consume_prev<T>() receiver must match the member-function class type exactly");

    return consumed_member_receiver<class_t> {
        resolve_as<class_t>(spec, ec)
    };
}

template <typename F, typename Spec, typename Ctx, typename Prev>
constexpr auto resolve_member_receiver(Spec& spec, exec_context<Ctx, Prev>& ec)
    noexcept(noexcept(resolve_as<member_receiver_arg_t<F>>(spec, ec))) {
    using receiver_t = member_receiver_arg_t<F>;
    using borrowed_t = member_borrowed_receiver_t<F>;

    auto&& receiver = resolve_as<receiver_t>(spec, ec);

    return borrowed_member_receiver<borrowed_t> {
        std::reference_wrapper<borrowed_t> {receiver}
    };
}

template <typename Task>
concept bind_task_object =
    requires {
        typename std::remove_cvref_t<Task>::raw_result_type;
    };

template <typename F, typename... Specs>
concept bind_signature_matches =
    ordinary_bind_callable<F> &&
    function_traits<std::remove_cvref_t<F>>::arity == sizeof...(Specs);

template <typename T, typename F, typename... Specs>
concept bind_into_signature_matches =
    inferable_direct_output_callable<F> &&
    function_traits<std::remove_cvref_t<F>>::arity == sizeof...(Specs) + 1 &&
    std::is_same_v<
        std::remove_cvref_t<last_arg_t<std::remove_cvref_t<F>>>,
        direct_out<T>>;

template <typename F, typename... Specs>
concept inferred_bind_into_signature_matches =
    inferable_direct_output_callable<F> &&
    function_traits<std::remove_cvref_t<F>>::arity == sizeof...(Specs) + 1;

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

template <typename Self, typename T>
[[nodiscard]] constexpr decltype(auto) forward_member(T& value) noexcept {
    if constexpr (std::is_lvalue_reference_v<Self&&>) {
        return (value);
    } else {
        return std::move(value);
    }
}

} // namespace yorch::detail

namespace yorch {

/**
 * @brief Bound executable task composed of a callable and per-parameter specs.
 *
 * At execution time, each spec is resolved against the target parameter type
 * deduced from `F`, then the callable is invoked and its original return value
 * is forwarded back to the caller.
 *
 * @tparam F Stored callable type.
 * @tparam Specs Stored spec types, one for each function parameter.
 */
template <typename F, typename... Specs>
struct bound_task {
    /// Raw return type declared by the stored callable.
    using raw_result_type = detail::result_t<F>;

    /// Stored callable to execute.
    F func;

    /// Stored argument binding specs in parameter order.
    std::tuple<Specs...> specs;

    /**
     * @brief Resolves all specs and executes the bound callable.
     * @tparam Ctx Context schema used for this execution.
     * @tparam Prev Parent slot view type for this execution.
     * @param ec Borrowed execution context.
     * @return User callable's original return value.
     */
    template <typename Ctx, typename Prev>
    constexpr decltype(auto) invoke_raw(exec_context<Ctx, Prev>& ec)
        noexcept(noexcept(call_impl_raw(ec, std::index_sequence_for<Specs...>{}))) {
        return call_impl_raw(ec, std::index_sequence_for<Specs...>{});
    }

private:
    template <typename Ctx, typename Prev, std::size_t... I>
    constexpr decltype(auto) call_impl_raw(exec_context<Ctx, Prev>& ec, std::index_sequence<I...>)
        noexcept(noexcept(std::invoke(
            func,
            resolve_as<detail::nth_arg_t<I, F>>(std::get<I>(specs), ec)...))) {
        return std::invoke(
            func,
            resolve_as<detail::nth_arg_t<I, F>>(std::get<I>(specs), ec)...
        );
    }
};

/**
 * @brief Bound direct-output task that writes its payload into a provided slot.
 *
 * This form keeps parameter resolution identical to `bound_task`, but the
 * stored callable receives a trailing `direct_out<T>` sink instead of
 * returning the payload through its function result.
 *
 * @tparam F Stored callable type.
 * @tparam T Payload type written into the output sink.
 * @tparam Specs Stored input binding specs, one for each non-output parameter.
 */
template <typename F, typename T, typename... Specs>
struct bound_output_task {
    using raw_result_type = step_result;
    using output_type = T;

    F func;
    std::tuple<Specs...> specs;

    template <typename Ctx, typename Prev>
    constexpr step_result invoke_into(exec_context<Ctx, Prev>& ec, direct_out<T> out)
        noexcept(noexcept(call_impl_into(ec, out, std::index_sequence_for<Specs...> {}))) {
        using call_result_t =
            decltype(call_impl_into(ec, out, std::index_sequence_for<Specs...> {}));

        if constexpr (std::is_void_v<call_result_t>) {
            call_impl_into(ec, out, std::index_sequence_for<Specs...> {});
            return step_result::success();
        } else {
            static_assert(std::is_same_v<std::remove_cvref_t<call_result_t>, step_result>,
                          "bind_into(...) callable must return void or yorch::step_result");
            return call_impl_into(ec, out, std::index_sequence_for<Specs...> {});
        }
    }

private:
    template <typename Ctx, typename Prev, std::size_t... I>
    constexpr decltype(auto) call_impl_into(
        exec_context<Ctx, Prev>& ec,
        direct_out<T> out,
        std::index_sequence<I...>)
        noexcept(noexcept(std::invoke(
            func,
            resolve_as<detail::nth_arg_t<I, F>>(std::get<I>(specs), ec)...,
            out))) {
        return std::invoke(
            func,
            resolve_as<detail::nth_arg_t<I, F>>(std::get<I>(specs), ec)...,
            out);
    }
};

/**
 * @brief Bound executable task composed of a member function, receiver spec,
 * and per-parameter specs.
 *
 * This form models the receiver as an explicit binding source distinct from
 * the business parameters. At execution time the receiver is resolved first,
 * then all remaining specs are resolved against the member-function parameter
 * list and invoked through `std::invoke(...)`.
 *
 * @tparam F Stored member-function pointer type.
 * @tparam ReceiverSpec Stored receiver binding spec.
 * @tparam Specs Stored business-parameter specs.
 */
template <typename F, typename ReceiverSpec, typename... Specs>
struct bound_member_task {
    using raw_result_type = detail::member_result_t<F>;

    F func;
    ReceiverSpec receiver_spec;
    std::tuple<Specs...> specs;

    template <typename Ctx, typename Prev>
    constexpr decltype(auto) invoke_raw(exec_context<Ctx, Prev>& ec)
        noexcept(noexcept(call_impl_raw(ec, std::index_sequence_for<Specs...>{}))) {
        return call_impl_raw(ec, std::index_sequence_for<Specs...>{});
    }

private:
    template <typename Ctx, typename Prev, std::size_t... I>
    constexpr decltype(auto) call_impl_raw(exec_context<Ctx, Prev>& ec, std::index_sequence<I...>)
        noexcept(noexcept(detail::invoke_member_with_receiver(
            func,
            detail::resolve_member_receiver<F>(receiver_spec, ec),
            resolve_as<detail::member_nth_arg_t<I, F>>(std::get<I>(specs), ec)...))) {
        auto receiver = detail::resolve_member_receiver<F>(receiver_spec, ec);
        return detail::invoke_member_with_receiver(
            func,
            std::move(receiver),
            resolve_as<detail::member_nth_arg_t<I, F>>(std::get<I>(specs), ec)...);
    }
};

/**
 * @brief Bound direct-output task composed of a member function, receiver
 * spec, and non-output parameter specs.
 *
 * The receiver is resolved first, then all non-output business parameters are
 * resolved in order, and finally the provided `direct_out<T>` sink is passed
 * to the stored member function via `std::invoke(...)`.
 *
 * @tparam F Stored member-function pointer type.
 * @tparam ReceiverSpec Stored receiver binding spec.
 * @tparam T Output payload type.
 * @tparam Specs Stored non-output business-parameter specs.
 */
template <typename F, typename ReceiverSpec, typename T, typename... Specs>
struct bound_member_output_task {
    using raw_result_type = step_result;
    using output_type = T;

    F func;
    ReceiverSpec receiver_spec;
    std::tuple<Specs...> specs;

    template <typename Ctx, typename Prev>
    constexpr step_result invoke_into(exec_context<Ctx, Prev>& ec, direct_out<T> out)
        noexcept(noexcept(call_impl_into(ec, out, std::index_sequence_for<Specs...> {}))) {
        using call_result_t =
            decltype(call_impl_into(ec, out, std::index_sequence_for<Specs...> {}));

        if constexpr (std::is_void_v<call_result_t>) {
            call_impl_into(ec, out, std::index_sequence_for<Specs...> {});
            return step_result::success();
        } else {
            static_assert(std::is_same_v<std::remove_cvref_t<call_result_t>, step_result>,
                          "bind_into_member(...) callable must return void or yorch::step_result");
            return call_impl_into(ec, out, std::index_sequence_for<Specs...> {});
        }
    }

private:
    template <typename Ctx, typename Prev, std::size_t... I>
    constexpr decltype(auto) call_impl_into(
        exec_context<Ctx, Prev>& ec,
        direct_out<T> out,
        std::index_sequence<I...>)
        noexcept(noexcept(detail::invoke_member_with_receiver(
            func,
            detail::resolve_member_receiver<F>(receiver_spec, ec),
            resolve_as<detail::member_nth_arg_t<I, F>>(std::get<I>(specs), ec)...,
            out))) {
        auto receiver = detail::resolve_member_receiver<F>(receiver_spec, ec);
        return detail::invoke_member_with_receiver(
            func,
            std::move(receiver),
            resolve_as<detail::member_nth_arg_t<I, F>>(std::get<I>(specs), ec)...,
            out);
    }
};

/**
 * @brief Creates a `bound_task` from a callable and matching argument specs.
 *
 * The callable and specs are stored by decayed value. The number of specs must
 * exactly match the callable arity so each parameter can be resolved with the
 * corresponding target type.
 *
 * Current boundary: `bind(...)` still requires a callable with one unambiguous
 * signature. Overloaded function names, generic lambdas, and callables with
 * overloaded `operator()` are intentionally not deduced automatically in this
 * phase. For example, `bind(foo, ...)` is ambiguous if `foo` is overloaded,
 * `bind([](auto&& x) { ... }, ...)` is unsupported because the lambda is
 * generic, and a type such as `struct X { void operator()(int) const; void
 * operator()(double) const; };` is also out of scope for this phase.
 *
 * @tparam F Callable type.
 * @tparam Specs Spec types.
 * @param f Callable to execute later.
 * @param specs Parameter binding specs in call order.
 * @return A `bound_task` that can be run with an `exec_context`.
 */
template <typename F, typename... Specs>
constexpr auto bind(F&& f, Specs&&... specs) { // NOLINT(readability-identifier-length)
    static_assert(
        !detail::member_bind_callable<F>,
        "bind(...) does not accept member function pointers; use bind_member(...) instead");
    static_assert(
        sizeof...(Specs) == detail::function_traits<std::remove_cvref_t<F>>::arity,
        "bind(...) requires exactly one spec per function parameter"
    );

    return bound_task<
        std::decay_t<F>,
        std::decay_t<Specs>...
    >{
        std::forward<F>(f),
        std::tuple<std::decay_t<Specs>...>{std::forward<Specs>(specs)...}
    };
}

/**
 * @brief Creates a direct-output task from a callable and matching input specs.
 *
 * The callable's last parameter must be a `direct_out<T>`-compatible output
 * sink. All earlier parameters are resolved exactly like `bind(...)`.
 *
 * @tparam T Payload type produced into the destination slot.
 * @tparam F Callable type.
 * @tparam Specs Input spec types.
 * @param f Callable to execute later.
 * @param specs Input binding specs, one for each non-output parameter.
 * @return A `bound_output_task` that can be executed through `invoke_into(...)`.
 */
template <typename T, typename F, typename... Specs>
constexpr auto bind_into(F&& f, Specs&&... specs) {
    using fn_t = std::remove_cvref_t<F>;

    static_assert(
        !detail::member_bind_callable<fn_t>,
        "bind_into(...) does not accept member function pointers; use bind_into_member(...) instead");
    static_assert(detail::function_traits<fn_t>::arity > 0,
                  "bind_into(...) requires a callable whose last parameter is yorch::direct_out<T>");
    static_assert(
        sizeof...(Specs) + 1 == detail::function_traits<fn_t>::arity,
        "bind_into(...) requires exactly one spec per non-output function parameter");

    using last_arg_t = detail::last_arg_t<fn_t>;

    static_assert(
        std::is_same_v<std::remove_cvref_t<last_arg_t>, direct_out<T>>,
        "bind_into(...) callable must take yorch::direct_out<T> as its last parameter");

    return bound_output_task<
        std::decay_t<F>,
        T,
        std::decay_t<Specs>...
    >{
        std::forward<F>(f),
        std::tuple<std::decay_t<Specs>...> {std::forward<Specs>(specs)...}
    };
}

template <typename F>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
constexpr void bind_member(F&&) {
    static_assert(
        detail::always_false_v<F>,
        "bind_member(...) requires an explicit receiver binding as its second argument");
}

template <typename F, typename ReceiverSpec, typename... Specs>
constexpr auto bind_member(F&& f, ReceiverSpec&& receiver_spec, Specs&&... specs) {
    using fn_t = std::remove_cvref_t<F>;

    static_assert(
        detail::member_bind_callable<fn_t>,
        "bind_member(...) requires a non-static member function pointer");
    static_assert(
        detail::ordinary_member_bind_callable<fn_t>,
        "bind_member(...) does not accept direct-output member functions; use bind_into_member(...) instead");
    static_assert(
        sizeof...(Specs) == detail::member_function_traits<fn_t>::arity,
        "bind_member(...) requires one receiver binding plus exactly one spec per member-function parameter");

    return bound_member_task<
        std::decay_t<F>,
        std::decay_t<ReceiverSpec>,
        std::decay_t<Specs>...
    >{
        std::forward<F>(f),
        std::forward<ReceiverSpec>(receiver_spec),
        std::tuple<std::decay_t<Specs>...>{std::forward<Specs>(specs)...}
    };
}

template <typename T, typename F>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
constexpr void bind_into_member(F&&) {
    static_assert(
        detail::always_false_v<std::tuple<T, F>>,
        "bind_into_member(...) requires an explicit receiver binding as its second argument");
}

template <typename T, typename F, typename ReceiverSpec, typename... Specs>
constexpr auto bind_into_member(F&& f, ReceiverSpec&& receiver_spec, Specs&&... specs) {
    using fn_t = std::remove_cvref_t<F>;

    static_assert(
        detail::member_bind_callable<fn_t>,
        "bind_into_member(...) requires a non-static member function pointer");
    static_assert(detail::member_function_traits<fn_t>::arity > 0,
                  "bind_into_member(...) requires a member function whose last parameter is yorch::direct_out<T>");
    static_assert(
        sizeof...(Specs) + 1 == detail::member_function_traits<fn_t>::arity,
        "bind_into_member(...) requires one receiver binding plus exactly one spec per non-output member-function parameter");

    using last_arg_t = detail::member_last_arg_t<fn_t>;

    static_assert(
        std::is_same_v<std::remove_cvref_t<last_arg_t>, direct_out<T>>,
        "bind_into_member(...) callable must take yorch::direct_out<T> as its last parameter");

    return bound_member_output_task<
        std::decay_t<F>,
        std::decay_t<ReceiverSpec>,
        T,
        std::decay_t<Specs>...
    >{
        std::forward<F>(f),
        std::forward<ReceiverSpec>(receiver_spec),
        std::tuple<std::decay_t<Specs>...> {std::forward<Specs>(specs)...}
    };
}

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

template <typename F, typename AdapterChain>
struct task_into_binder;

template <typename F, typename AdapterChain>
struct task_binder {
    F func;
    AdapterChain adapter_specs;

    template <typename... Specs>
        requires detail::bind_signature_matches<F, Specs...>
    constexpr auto operator()(Specs&&... specs) const& {
        return yorch::apply_adapters(
            yorch::bind(func, std::forward<Specs>(specs)...),
            adapter_specs);
    }

    template <typename... Specs>
        requires detail::bind_signature_matches<F, Specs...>
    constexpr auto operator()(Specs&&... specs) && {
        return yorch::apply_adapters(
            yorch::bind(std::move(func), std::forward<Specs>(specs)...),
            std::move(adapter_specs));
    }
};

template <typename F, typename AdapterChain>
struct task_into_binder {
    F func;
    AdapterChain adapter_specs;
    using output_type =
        detail::direct_out_payload_t<detail::last_arg_t<F>>;

    template <typename... Specs>
        requires detail::inferred_bind_into_signature_matches<F, Specs...>
    constexpr auto operator()(Specs&&... specs) const& {
        return yorch::apply_adapters(
            yorch::bind_into<output_type>(func, std::forward<Specs>(specs)...),
            adapter_specs);
    }

    template <typename... Specs>
        requires detail::inferred_bind_into_signature_matches<F, Specs...>
    constexpr auto operator()(Specs&&... specs) && {
        return yorch::apply_adapters(
            yorch::bind_into<output_type>(std::move(func), std::forward<Specs>(specs)...),
            std::move(adapter_specs));
    }
};

template <typename F>
constexpr auto task(F&& f)
    requires detail::ordinary_bind_callable<F> {
    return task_binder<std::decay_t<F>, adapter_chain<>> {
        std::forward<F>(f),
        {}
    };
}

template <typename F, typename AdapterChain>
constexpr auto task(F&& f, AdapterChain&& adapter_specs)
    requires detail::ordinary_bind_callable<F> &&
             detail::adapter_chain_like<AdapterChain> {
    return task_binder<
        std::decay_t<F>,
        std::decay_t<AdapterChain>
    > {
        std::forward<F>(f),
        std::forward<AdapterChain>(adapter_specs)
    };
}

template <typename F>
constexpr auto task_into(F&& f)
    requires detail::inferable_direct_output_callable<F> {
    return task_into_binder<std::decay_t<F>, adapter_chain<>> {
        std::forward<F>(f),
        {}
    };
}

template <typename F, typename AdapterChain>
constexpr auto task_into(F&& f, AdapterChain&& adapter_specs)
    requires detail::inferable_direct_output_callable<F> &&
             detail::adapter_chain_like<AdapterChain> {
    return task_into_binder<
        std::decay_t<F>,
        std::decay_t<AdapterChain>
    > {
        std::forward<F>(f),
        std::forward<AdapterChain>(adapter_specs)
    };
}

template <typename F>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
constexpr void task(F&&)
    requires detail::inferable_direct_output_callable<F> {
    static_assert(
        detail::always_false_v<F>,
        "yorch::task(...) received a direct-output callable whose last parameter is yorch::direct_out<T>; use yorch::task_into(...) instead.");
}

template <typename F>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
constexpr void task(F&&)
    requires detail::member_bind_callable<F> {
    static_assert(
        detail::always_false_v<F>,
        "yorch::task(...) does not accept member function pointers; use bind_member(...) first.");
}

template <typename Task>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
constexpr void task(Task&&)
    requires detail::bind_task_object<Task> &&
             detail::has_declared_task_output_v<Task> {
    static_assert(
        detail::always_false_v<Task>,
        "yorch::task(...) does not accept prebuilt direct-output tasks; pass them directly to root_into(...) or node_into<Level>(...) instead.");
}

template <typename F, typename AdapterChain>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
constexpr void task(F&&, AdapterChain&&)
    requires detail::inferable_direct_output_callable<F> &&
             detail::adapter_chain_like<AdapterChain> {
    static_assert(
        detail::always_false_v<std::tuple<F, AdapterChain>>,
        "yorch::task(...) received a direct-output callable whose last parameter is yorch::direct_out<T>; use yorch::task_into(...) instead.");
}

template <typename F, typename AdapterChain>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
constexpr void task(F&&, AdapterChain&&)
    requires detail::member_bind_callable<F> &&
             detail::adapter_chain_like<AdapterChain> {
    static_assert(
        detail::always_false_v<std::tuple<F, AdapterChain>>,
        "yorch::task(...) does not accept member function pointers; use bind_member(...) first.");
}

template <typename Task>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
constexpr void task_into(Task&&)
    requires detail::bind_task_object<Task> {
    static_assert(
        detail::always_false_v<Task>,
        "yorch::task_into(...) only accepts a callable whose last parameter is yorch::direct_out<T>; pass prebuilt tasks directly to root/node instead.");
}

template <typename Task, typename AdapterChain>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
constexpr void task_into(Task&&, AdapterChain&&)
    requires detail::bind_task_object<Task> &&
             detail::adapter_chain_like<AdapterChain> {
    static_assert(
        detail::always_false_v<std::tuple<Task, AdapterChain>>,
        "yorch::task_into(...) only accepts a callable whose last parameter is yorch::direct_out<T>; pass prebuilt tasks directly to root/node instead.");
}

template <typename F>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
constexpr void task_into(F&&)
    requires detail::ordinary_bind_callable<F> {
    static_assert(
        detail::always_false_v<F>,
        "yorch::task_into(...) requires a callable whose last parameter is yorch::direct_out<T>; use yorch::task(...) for ordinary tasks.");
}

template <typename F>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
constexpr void task_into(F&&)
    requires detail::member_bind_callable<F> {
    static_assert(
        detail::always_false_v<F>,
        "yorch::task_into(...) does not accept member function pointers; use bind_into_member(...) first.");
}

template <typename F, typename AdapterChain>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
constexpr void task_into(F&&, AdapterChain&&)
    requires detail::ordinary_bind_callable<F> &&
             detail::adapter_chain_like<AdapterChain> {
    static_assert(
        detail::always_false_v<std::tuple<F, AdapterChain>>,
        "yorch::task_into(...) requires a callable whose last parameter is yorch::direct_out<T>; use yorch::task(...) for ordinary tasks.");
}

template <typename F, typename AdapterChain>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
constexpr void task_into(F&&, AdapterChain&&)
    requires detail::member_bind_callable<F> &&
             detail::adapter_chain_like<AdapterChain> {
    static_assert(
        detail::always_false_v<std::tuple<F, AdapterChain>>,
        "yorch::task_into(...) does not accept member function pointers; use bind_into_member(...) first.");
}

} // namespace yorch
