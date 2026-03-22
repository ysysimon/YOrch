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

template <typename T>
struct function_traits;

template <typename R, typename... Args>
struct function_traits<R(Args...)> {
    using result_type = R;
    static constexpr std::size_t arity = sizeof...(Args);

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

template <typename R>
constexpr step_result normalize_result(R&& r) {
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

template <typename F, typename... Args>
constexpr step_result invoke_and_normalize(F&& f, Args&&... args) {
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

template <typename F, typename... Specs>
struct bound_task {
    F func;
    std::tuple<Specs...> specs;

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

template <typename F, typename... Specs>
constexpr auto bind(F&& f, Specs&&... specs) {
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