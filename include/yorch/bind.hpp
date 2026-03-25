#pragma once
#include <cstddef>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

#include "context.hpp"
#include "resolve.hpp"
#include "result.hpp"

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
struct function_traits<R (&)(Args...)> : function_traits<R(Args...)> {};

template <typename C, typename R, typename... Args>
struct function_traits<R (C::*)(Args...)> : function_traits<R(Args...)> {};

template <typename C, typename R, typename... Args>
struct function_traits<R (C::*)(Args...) const> : function_traits<R(Args...)> {};

template <typename F>
struct function_traits : function_traits<decltype(&std::remove_reference_t<F>::operator())> {};

template <std::size_t I, typename F>
using nth_arg_t = typename function_traits<std::remove_cvref_t<F>>::template arg<I>;

template <typename F>
using result_t = typename function_traits<std::remove_cvref_t<F>>::result_type;

/**
 * @brief Converts a supported task return value into `step_result`.
 *
 * Supported inputs are `step_result` itself and `bool`. `bool` is interpreted
 * as a success/failure shortcut, while unsupported return categories fail at
 * compile time.
 *
 * @tparam R Concrete return object type.
 * @param r Return value produced by a bound callable.
 * @return Normalized scheduler-facing result.
 */
template <typename R>
constexpr step_result normalize_result(R&& r) { // NOLINT(readability-identifier-length)
    using raw_t = std::remove_cvref_t<R>;

    if constexpr (std::is_same_v<raw_t, step_result>) {
        return std::forward<R>(r);
    } else if constexpr (std::is_same_v<raw_t, bool>) {
        return r ? step_result::success()
                 : step_result::failure();
    } else {
        static_assert(std::is_same_v<raw_t, void>,
                      "Unsupported return type");
    }
}

/**
 * @brief Invokes a callable and normalizes its completion status.
 *
 * `void` is treated as success. Non-void results are forwarded to
 * `normalize_result(...)`.
 *
 * @tparam F Callable type.
 * @tparam Args Argument types used for invocation.
 * @param f Callable to invoke.
 * @param args Concrete arguments prepared for the call.
 * @return Normalized `step_result` for the scheduler.
 */
template <typename F, typename... Args>
constexpr step_result invoke_and_normalize(F&& f, Args&&... args) { // NOLINT(readability-identifier-length)
    using R = std::invoke_result_t<F, Args...>;

    if constexpr (std::is_void_v<R>) {
        std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
        return step_result::success();
    } else {
        return normalize_result(
            std::invoke(std::forward<F>(f), std::forward<Args>(args)...)
        );
    }
}

} // namespace yorch::detail

namespace yorch {

/**
 * @brief Bound executable task composed of a callable and per-parameter specs.
 *
 * At execution time, each spec is resolved against the target parameter type
 * deduced from `F`, then the callable is invoked and its return value is
 * normalized to `step_result`.
 *
 * @tparam F Stored callable type.
 * @tparam Specs Stored spec types, one for each function parameter.
 */
template <typename F, typename... Specs>
struct bound_task {
    /// Stored callable to execute.
    F func;

    /// Stored argument binding specs in parameter order.
    std::tuple<Specs...> specs;

    /**
     * @brief Resolves all specs and executes the bound callable.
     * @tparam Ctx Context schema used for this execution.
     * @param ec Borrowed execution context.
     * @return Normalized task result.
     */
    template <typename Ctx>
    constexpr step_result operator()(exec_context<Ctx>& ec) {
        return call_impl(ec, std::index_sequence_for<Specs...>{});
    }

private:
    template <typename Ctx, std::size_t... I>
    constexpr step_result call_impl(exec_context<Ctx>& ec, std::index_sequence<I...>) {
        return detail::invoke_and_normalize(
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