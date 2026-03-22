#pragma once
#include <type_traits>
#include "context.hpp"
#include "specs.hpp"

namespace yorch::detail {

template <typename>
inline constexpr bool always_false_v = false;

} // namespace yorch::detail

namespace yorch {

/**
 * @brief Binds an existing lvalue source object to the requested argument type.
 *
 * This is the final adaptation step in resolution: it does not locate the
 * source object, it only turns an already available lvalue into `Arg`. The
 * function supports binding to `T&`, `const T&`, and value parameters, while
 * rejecting rvalue-reference arguments.
 *
 * @tparam Arg Target function parameter type.
 * @tparam Source Concrete source object type.
 * @param src Source object to bind from.
 * @return A reference or value matching `Arg`.
 */
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

/**
 * @brief Resolves a `from_ctx(...)` spec by fetching the requested object from
 * the execution context and binding it as `Arg`.
 *
 * The lookup type is normalized by `from_ctx_t<T>::type`, then validated
 * against the context schema before the retrieved object is passed to
 * `bind_from_lvalue`.
 *
 * @tparam Arg Target function parameter type.
 * @tparam T Requested context type encoded by the spec.
 * @tparam Ctx Context type carried by the current execution.
 * @param ec Execution context that provides the source object.
 * @return A reference or value matching `Arg`.
 */
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

/**
 * @brief Resolves a mutable `value(...)` spec into the requested argument type.
 *
 * The stored object is taken from `spec.v` and adapted under the stricter
 * `value(...)` rules: value parameters are copied or converted from the stored
 * lvalue, and reference parameters must be `const T&`.
 *
 * @tparam Arg Target function parameter type.
 * @tparam T Stored value type.
 * @tparam Ctx Context type, unused for this spec category.
 * @param spec Spec that owns the stored value.
 * @return A const reference or value matching `Arg`.
 */
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

/**
 * @brief Resolves a const `value(...)` spec into the requested argument type.
 *
 * This overload preserves the constness of the stored source expression. As a
 * result, reference binding is limited to `const T&`, and value construction
 * must be valid from `const T&`.
 *
 * @tparam Arg Target function parameter type.
 * @tparam T Stored value type.
 * @tparam Ctx Context type, unused for this spec category.
 * @param spec Const spec that owns the stored value.
 * @return A const reference or value matching `Arg`.
 */
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

/**
 * @brief Fallback overload for unsupported spec categories.
 *
 * This function exists only to produce a focused compile-time diagnostic when
 * no dedicated `resolve_as(...)` overload matches the provided spec.
 *
 * @tparam Arg Target function parameter type.
 * @tparam Spec Unsupported spec type.
 * @tparam Ctx Context type carried by the current execution.
 */
template <typename Arg, typename Spec, typename Ctx>
constexpr decltype(auto) resolve_as(Spec&&, exec_context<Ctx>&) {
    static_assert(detail::always_false_v<std::remove_cvref_t<Spec>>,
                  "Unsupported spec in resolve_as");
}

} // namespace yorch