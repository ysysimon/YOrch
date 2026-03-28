#pragma once
#include <cstddef>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

#include "context.hpp"
#include "resolve.hpp"

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

} // namespace yorch
