#pragma once
#include <type_traits>
#include <utility>
#include "context.hpp"
#include "specs.hpp"

namespace yorch::detail {

template <typename>
inline constexpr bool always_false_v = false;

template <typename Arg, typename Source>
inline constexpr bool supports_bind_from_lvalue_v =
    !std::is_rvalue_reference_v<Arg> &&
    (
        std::is_reference_v<Arg> ||
        std::is_constructible_v<std::remove_cvref_t<Arg>, Source&>
    );

template <typename Arg, typename Source>
inline constexpr bool bind_from_lvalue_nothrow_v =
    supports_bind_from_lvalue_v<Arg, Source> &&
    (
        std::is_reference_v<Arg> ||
        std::is_nothrow_constructible_v<std::remove_cvref_t<Arg>, Source&>
    );

template <typename T>
using from_ctx_source_t = typename from_ctx_t<T>::type;

template <typename T>
using from_prev_source_t = typename from_prev_t<T>::type;

template <typename Ctx, typename T>
inline constexpr bool ctx_get_nothrow_v =
    noexcept(std::declval<Ctx&>().template get<from_ctx_source_t<T>>());

template <typename Prev, typename T>
inline constexpr bool prev_get_nothrow_v =
    noexcept(std::declval<Prev&>().template get<from_prev_source_t<T>>());

template <typename Arg, typename T, typename Ctx>
inline constexpr bool resolve_from_ctx_nothrow_v =
    ctx_get_nothrow_v<Ctx, T> &&
    bind_from_lvalue_nothrow_v<Arg, from_ctx_source_t<T>>;

template <typename Arg, typename T, typename Prev>
inline constexpr bool resolve_from_prev_nothrow_v =
    prev_get_nothrow_v<Prev, T> &&
    bind_from_lvalue_nothrow_v<Arg, from_prev_source_t<T>>;

template <typename Arg, typename T>
inline constexpr bool resolve_value_nothrow_v =
    std::is_reference_v<Arg> ||
    std::is_nothrow_constructible_v<std::remove_cvref_t<Arg>, T&>;

template <typename Arg, typename T>
inline constexpr bool resolve_const_value_nothrow_v =
    std::is_reference_v<Arg> ||
    std::is_nothrow_constructible_v<std::remove_cvref_t<Arg>, const T&>;

} // namespace yorch::detail

namespace yorch {

template <typename Arg, typename Source>
concept bindable_from_lvalue =
    detail::supports_bind_from_lvalue_v<Arg, Source>;

template <typename Arg, typename T>
concept resolvable_mutable_value =
    !std::is_rvalue_reference_v<Arg> &&
    (
        (
            std::is_reference_v<Arg> &&
            std::is_const_v<std::remove_reference_t<Arg>> &&
            std::is_convertible_v<T&, const std::remove_cvref_t<Arg>&>
        ) ||
        (
            !std::is_reference_v<Arg> &&
            std::is_constructible_v<std::remove_cvref_t<Arg>, T&>
        )
    );

template <typename Arg, typename T>
concept resolvable_const_value =
    !std::is_rvalue_reference_v<Arg> &&
    (
        (
            std::is_reference_v<Arg> &&
            std::is_const_v<std::remove_reference_t<Arg>> &&
            std::is_convertible_v<const T&, const std::remove_cvref_t<Arg>&>
        ) ||
        (
            !std::is_reference_v<Arg> &&
            std::is_constructible_v<std::remove_cvref_t<Arg>, const T&>
        )
    );

/**
 * @brief Binds an existing lvalue source object to the requested argument type.
 *
 * This is the final adaptation step in resolution: it does not locate the
 * source object, it only turns an already available lvalue into `Arg`. The
 * function supports binding to `T&`, `const T&`, and value parameters, while
 * rejecting rvalue-reference arguments.
 *
 * Rejecting `T&&` is intentional: this helper is used for stable lvalue
 * sources such as context entries, parent payloads, and stored `value(...)`
 * objects. Treating those sources as rvalues would let callees move from
 * shared or reusable state, which would make later retries or repeated
 * invocations observe moved-from objects.
 *
 * @tparam Arg Target function parameter type.
 * @tparam Source Concrete source object type.
 * @param src Source object to bind from.
 * @return A reference or value matching `Arg`.
 */
template <typename Arg, typename Source>
    requires bindable_from_lvalue<Arg, Source>
constexpr decltype(auto) bind_from_lvalue(Source& src)
    noexcept(detail::bind_from_lvalue_nothrow_v<Arg, Source>) {
    static_assert(detail::supports_bind_from_lvalue_v<Arg, Source>,
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
 * @tparam Prev Parent slot view type carried by the current execution.
 * @param ec Execution context that provides the source object.
 * @return A reference or value matching `Arg`.
 */
template <typename Arg, typename T, typename Ctx, typename Prev>
constexpr decltype(auto) resolve_as(from_ctx_t<T>, exec_context<Ctx, Prev>& ec)
    noexcept(detail::resolve_from_ctx_nothrow_v<Arg, T, Ctx>) {
    static_assert(!std::is_void_v<Ctx>,
                  "from_ctx<T>() cannot be used with exec_context<void>");

    using source_t = typename from_ctx_t<T>::type;

    static_assert(Ctx::template contains<source_t>(),
                  "Requested type is not present in the context");

    auto& src = ec.ctx.template get<source_t>();
    return bind_from_lvalue<Arg>(src);
}

/**
 * @brief Resolves a `from_prev(...)` spec by fetching the direct parent output
 * payload from the current execution's parent slot view.
 *
 * The lookup type is normalized by `from_prev_t<T>::type`, then validated
 * against the direct parent slot carried by the execution context before the
 * retrieved object is passed to `bind_from_lvalue`.
 *
 * @tparam Arg Target function parameter type.
 * @tparam T Requested parent payload type encoded by the spec.
 * @tparam Ctx Context type carried by the current execution.
 * @tparam Prev Direct-parent slot view type carried by the execution.
 * @param ec Execution context that provides the direct parent payload.
 * @return A reference or value matching `Arg`.
 */
template <typename Arg, typename T, typename Ctx, typename Prev>
constexpr decltype(auto) resolve_as(from_prev_t<T>, exec_context<Ctx, Prev>& ec)
    noexcept(detail::resolve_from_prev_nothrow_v<Arg, T, Prev>) {
    using source_t = typename from_prev_t<T>::type;
    using prev_t = std::remove_cvref_t<decltype(ec.prev_view())>;

    static_assert(prev_t::template contains<source_t>(),
                  "from_prev<T>() requires a direct parent slot carrying the requested type");

    auto& src = ec.prev_view().template get<source_t>();
    return bind_from_lvalue<Arg>(src);
}

/**
 * @brief Resolves a mutable `value(...)` spec into the requested argument type.
 *
 * The stored object is taken from `spec.v` and adapted under the stricter
 * `value(...)` rules: value parameters are copied or converted from the stored
 * lvalue, and reference parameters must be `const T&`.
 *
 * `T&&` is rejected here for the same reason as in `bind_from_lvalue`: a
 * `value(...)` spec owns a reusable object, so resolving it must not silently
 * consume that stored state and leave retries or subsequent invocations
 * observing a moved-from payload.
 *
 * @tparam Arg Target function parameter type.
 * @tparam T Stored value type.
 * @tparam Ctx Context type, unused for this spec category.
 * @tparam Prev Parent slot view type carried by the current execution.
 * @param spec Spec that owns the stored value.
 * @return A const reference or value matching `Arg`.
 */
template <typename Arg, typename T, typename Ctx, typename Prev>
    requires resolvable_mutable_value<Arg, T>
constexpr decltype(auto) resolve_as(value_t<T>& spec, exec_context<Ctx, Prev>&)
    noexcept(detail::resolve_value_nothrow_v<Arg, T>) {
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
 * must be valid from `const T&`. As with the mutable overload, `T&&` is
 * intentionally unsupported because `value(...)` models reusable stored state,
 * not a one-shot forwarding source.
 *
 * @tparam Arg Target function parameter type.
 * @tparam T Stored value type.
 * @tparam Ctx Context type, unused for this spec category.
 * @tparam Prev Parent slot view type carried by the current execution.
 * @param spec Const spec that owns the stored value.
 * @return A const reference or value matching `Arg`.
 */
template <typename Arg, typename T, typename Ctx, typename Prev>
    requires resolvable_const_value<Arg, T>
constexpr decltype(auto) resolve_as(const value_t<T>& spec, exec_context<Ctx, Prev>&)
    noexcept(detail::resolve_const_value_nothrow_v<Arg, T>) {
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
 * @tparam Prev Parent slot view type carried by the current execution.
 */
template <typename Arg, typename Spec, typename Ctx, typename Prev>
constexpr decltype(auto) resolve_as(Spec&&, exec_context<Ctx, Prev>&) {
    static_assert(detail::always_false_v<std::remove_cvref_t<Spec>>,
                  "Unsupported spec in resolve_as");
}

} // namespace yorch
