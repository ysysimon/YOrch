#pragma once
#include <type_traits>
#include <utility>

namespace yorch {

/**
 * @brief Describes a parameter sourced from the execution context by type.
 *
 * This is a lightweight spec object used by the binding layer. It does not
 * access the context eagerly; actual lookup happens later during resolution.
 *
 * @tparam T Requested context type after cv-ref normalization.
 */
template <typename T>
struct from_ctx_t {
    /// Canonical context key type used for lookup.
    using type = std::remove_cvref_t<T>;
};

/**
 * @brief Describes a parameter that borrows the direct parent output as
 * `const T&`.
 *
 * This spec models shared readonly access to the direct parent payload.
 *
 * @tparam T Requested parent payload type after cv-ref normalization.
 */
template <typename T>
struct borrow_prev_t {
    /// Canonical parent payload type used for lookup.
    using type = std::remove_cvref_t<T>;
};

/**
 * @brief Describes a parameter that borrows the direct parent output as `T&`.
 *
 * This spec models exclusive mutable access to the direct parent payload.
 *
 * @tparam T Requested parent payload type after cv-ref normalization.
 */
template <typename T>
struct borrow_prev_mut_t {
    /// Canonical parent payload type used for lookup.
    using type = std::remove_cvref_t<T>;
};

/**
 * @brief Describes a parameter that copies the direct parent output as `T`.
 *
 * This spec models readonly copy access to the direct parent payload.
 *
 * @tparam T Requested parent payload type after cv-ref normalization.
 */
template <typename T>
struct copy_prev_t {
    /// Canonical parent payload type used for lookup.
    using type = std::remove_cvref_t<T>;
};

/**
 * @brief Describes a parameter that consumes the direct parent output.
 *
 * This spec models one-shot ownership transfer from the direct parent payload.
 *
 * @tparam T Requested parent payload type after cv-ref normalization.
 */
template <typename T>
struct consume_prev_t {
    /// Canonical parent payload type used for lookup.
    using type = std::remove_cvref_t<T>;
};

/**
 * @brief Stores a concrete value inside a spec.
 *
 * `value_t<T>` owns the object it carries so the bound argument can be
 * resolved later without depending on the lifetime of the original expression.
 *
 * @tparam T Stored value type. Must not be a reference.
 */
template <typename T>
struct value_t {
    /// Materialized type kept by this spec.
    using stored_type = T;

    static_assert(!std::is_reference_v<T>,
                  "value_t<T> must store a non-reference type");

    /// Owned payload forwarded from `value(...)`.
    T v;
};

/**
 * @brief Creates a spec that asks execution to fetch `T` from the context.
 *
 * The returned object only records the requested type. It is typically used
 * in binding expressions such as `from_ctx<Scene>()`.
 *
 * @tparam T Context type to request.
 * @return A `from_ctx_t` carrying the normalized lookup type.
 */
template <typename T>
[[nodiscard]] constexpr auto from_ctx() noexcept
    -> from_ctx_t<std::remove_cvref_t<T>> {
    return {};
}

/**
 * @brief Creates a spec that borrows `T` from the direct parent output slot
 * as `const T&`.
 *
 * The returned object only records the requested payload type. Actual access
 * happens later during resolution against the current execution's parent slot
 * view.
 *
 * @tparam T Parent payload type to request.
 * @return A `borrow_prev_t` carrying the normalized lookup type.
 */
template <typename T>
[[nodiscard]] constexpr auto borrow_prev() noexcept
    -> borrow_prev_t<std::remove_cvref_t<T>> {
    return {};
}

/**
 * @brief Creates a spec that borrows `T` from the direct parent output slot as
 * `T&`.
 *
 * @tparam T Parent payload type to request.
 * @return A `borrow_prev_mut_t` carrying the normalized lookup type.
 */
template <typename T>
[[nodiscard]] constexpr auto borrow_prev_mut() noexcept
    -> borrow_prev_mut_t<std::remove_cvref_t<T>> {
    return {};
}

/**
 * @brief Creates a spec that copies `T` from the direct parent output slot as
 * a new value.
 *
 * The returned object only records the requested payload type. Actual access
 * happens later during resolution against the current execution's parent slot
 * view.
 *
 * @tparam T Parent payload type to request.
 * @return A `copy_prev_t` carrying the normalized lookup type.
 */
template <typename T>
[[nodiscard]] constexpr auto copy_prev() noexcept
    -> copy_prev_t<std::remove_cvref_t<T>> {
    return {};
}

/**
 * @brief Creates a spec that consumes `T` from the direct parent output slot.
 *
 * @tparam T Parent payload type to request.
 * @return A `consume_prev_t` carrying the normalized lookup type.
 */
template <typename T>
[[nodiscard]] constexpr auto consume_prev() noexcept
    -> consume_prev_t<std::remove_cvref_t<T>> {
    return {};
}

/**
 * @brief Wraps a value as an owning spec.
 *
 * The argument is copied or moved into the returned `value_t`, which keeps a
 * decayed version of the original type.
 *
 * @tparam T Source expression type.
 * @param v Value to capture.
 * @return A `value_t` that owns the forwarded value.
 */
template <typename T>
[[nodiscard]] constexpr auto value(T&& v) // NOLINT(readability-identifier-length)
    -> value_t<std::remove_cvref_t<T>> {
    using stored_t = std::remove_cvref_t<T>;
    return value_t<stored_t>{std::forward<T>(v)};
}

} // namespace yorch
