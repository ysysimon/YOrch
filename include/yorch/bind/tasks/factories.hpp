#pragma once

#include <tuple>
#include <type_traits>
#include <utility>

#include "detail.hpp"
#include "types.hpp"

namespace yorch {

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
    detail::emit_bind_diagnostic<F, Specs...>();

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
    detail::emit_bind_into_diagnostic<T, F, Specs...>();

    return bound_output_task<
        std::decay_t<F>,
        T,
        std::decay_t<Specs>...
    >{
        std::forward<F>(f),
        std::tuple<std::decay_t<Specs>...> {std::forward<Specs>(specs)...}
    };
}

template <typename T, typename F, typename... Specs>
constexpr auto bind_forward_prev(F&& f, Specs&&... specs) {
    detail::emit_bind_forward_prev_diagnostic<T, F, Specs...>();

    return bound_forward_prev_task<
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
