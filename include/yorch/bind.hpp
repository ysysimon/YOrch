#pragma once
#include <cstddef>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

#include "context.hpp"
#include "resolve.hpp"
#include "slots.hpp"

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
template <typename T>
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
struct function_traits : function_traits<decltype(&std::remove_reference_t<F>::operator())> {};

template <std::size_t I, typename F>
using nth_arg_t = typename function_traits<std::remove_cvref_t<F>>::template arg<I>;

template <typename F>
using result_t = typename function_traits<std::remove_cvref_t<F>>::result_type;

template <typename F>
inline constexpr std::size_t last_arg_index_v =
    function_traits<std::remove_cvref_t<F>>::arity - 1;

template <typename F>
using last_arg_t =
    nth_arg_t<last_arg_index_v<F>, F>;

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
 * stored callable receives a trailing `result_out<T>` sink instead of
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
    constexpr step_result invoke_into(exec_context<Ctx, Prev>& ec, result_out<T> out)
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
        result_out<T> out,
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
 * The callable's last parameter must be a `result_out<T>`-compatible output
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

    static_assert(detail::function_traits<fn_t>::arity > 0,
                  "bind_into(...) requires a callable whose last parameter is yorch::result_out<T>");
    static_assert(
        sizeof...(Specs) + 1 == detail::function_traits<fn_t>::arity,
        "bind_into(...) requires exactly one spec per non-output function parameter");

    using last_arg_t = detail::last_arg_t<fn_t>;

    static_assert(
        std::is_same_v<std::remove_cvref_t<last_arg_t>, result_out<T>>,
        "bind_into(...) callable must take yorch::result_out<T> as its last parameter");

    return bound_output_task<
        std::decay_t<F>,
        T,
        std::decay_t<Specs>...
    >{
        std::forward<F>(f),
        std::tuple<std::decay_t<Specs>...> {std::forward<Specs>(specs)...}
    };
}

} // namespace yorch
