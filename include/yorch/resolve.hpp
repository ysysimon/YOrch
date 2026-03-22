#pragma once
#include <type_traits>
#include "context.hpp"
#include "specs.hpp"

namespace yorch::detail {

template <typename>
inline constexpr bool always_false_v = false;

} // namespace yorch::detail

namespace yorch {

template <typename Arg, typename Source>
constexpr decltype(auto) bind_from_lvalue(Source& src) {
    static_assert(!std::is_rvalue_reference_v<Arg>,
                  "Does not support binding to T&&");

    using raw_arg_t = std::remove_cvref_t<Arg>;

    if constexpr (std::is_reference_v<Arg>) {
        if constexpr (std::is_const_v<std::remove_reference_t<Arg>>) {
            static_assert(std::is_convertible_v<Source&, const raw_arg_t&>,
                          "Source cannot bind to const reference argument");
            return static_cast<const raw_arg_t&>(src);
        } else {
            static_assert(std::is_convertible_v<Source&, raw_arg_t&>,
                          "Source cannot bind to non-const reference argument");
            return static_cast<raw_arg_t&>(src);
        }
    } else {
        static_assert(std::is_constructible_v<raw_arg_t, Source&>,
                      "Source cannot be copied/converted into value argument");
        return static_cast<raw_arg_t>(src);
    }
}

template <typename Arg, typename T, typename Ctx>
constexpr decltype(auto) resolve_as(from_ctx_t<T>, exec_context<Ctx>& ec) {
    static_assert(!std::is_void_v<Ctx>,
                  "from_ctx<T>() cannot be used with exec_context<void>");

    using source_t = typename from_ctx_t<T>::type;

    static_assert(Ctx::template contains<source_t>(),
                  "Requested type is not present in the context");

    auto& src = ec.ctx.template get<source_t>();
    return bind_from_lvalue<Arg>(src);
}

template <typename Arg, typename T, typename Ctx>
constexpr decltype(auto) resolve_as(value_t<T>& spec, exec_context<Ctx>&) {
    static_assert(!std::is_rvalue_reference_v<Arg>,
                  "Does not support binding value(...) to T&&");

    using raw_arg_t = std::remove_cvref_t<Arg>;

    if constexpr (std::is_reference_v<Arg>) {
        static_assert(std::is_const_v<std::remove_reference_t<Arg>>,
                      "value(...) can only bind to T or const T&");
        static_assert(std::is_convertible_v<T&, const raw_arg_t&>,
                      "Stored value cannot bind to requested const reference type");
        return static_cast<const raw_arg_t&>(spec.v);
    } else {
        static_assert(std::is_constructible_v<raw_arg_t, T&>,
                      "Stored value cannot be copied/converted into value argument");
        return static_cast<raw_arg_t>(spec.v);
    }
}

template <typename Arg, typename T, typename Ctx>
constexpr decltype(auto) resolve_as(const value_t<T>& spec, exec_context<Ctx>&) {
    static_assert(!std::is_rvalue_reference_v<Arg>,
                  "Does not support binding value(...) to T&&");

    using raw_arg_t = std::remove_cvref_t<Arg>;

    if constexpr (std::is_reference_v<Arg>) {
        static_assert(std::is_const_v<std::remove_reference_t<Arg>>,
                      "const value(...) can only bind to T or const T&");
        static_assert(std::is_convertible_v<const T&, const raw_arg_t&>,
                      "Stored value cannot bind to requested const reference type");
        return static_cast<const raw_arg_t&>(spec.v);
    } else {
        static_assert(std::is_constructible_v<raw_arg_t, const T&>,
                      "Stored value cannot be copied/converted into value argument");
        return static_cast<raw_arg_t>(spec.v);
    }
}

template <typename Arg, typename Spec, typename Ctx>
constexpr decltype(auto) resolve_as(Spec&&, exec_context<Ctx>&) {
    static_assert(detail::always_false_v<std::remove_cvref_t<Spec>>,
                  "Unsupported spec in resolve_as");
}

} // namespace yorch