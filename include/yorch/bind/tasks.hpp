#pragma once

#include <cstddef>
#include <exception>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

#include "../context.hpp"
#include "../resolve.hpp"
#include "../slots.hpp"
#include "../task_adapters.hpp"
#include "../detail/bind/member_receiver.hpp"
#include "../detail/bind/traits.hpp"

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

} // namespace yorch
