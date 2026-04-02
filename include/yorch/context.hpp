#pragma once

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace yorch::detail {

template <typename T, typename... Ts>
inline constexpr std::size_t type_count_v = (std::size_t{0} + ... + (std::is_same_v<T, Ts> ? 1 : 0));

template <typename... Ts>
inline constexpr bool unique_types_v = ((type_count_v<Ts, Ts...> == 1) && ...);

template <typename>
inline constexpr bool no_prev_always_false_v = false;

} // namespace yorch::detail

namespace yorch {

/**
 * @brief Sentinel view indicating that the current execution has no direct
 * parent output.
 */
struct no_prev {
    template <typename T>
    [[nodiscard]] static constexpr bool contains() noexcept {
        return false;
    }

    /**
     * @brief Always fails because `no_prev` carries no retrievable value.
     *
     * `no_prev` still exposes the same `contains<T>() / get<T>()` protocol
     * shape as a concrete parent-slot view so generic resolution code can be
     * written against one interface. The operation itself is invalid here, so
     * `get<T>()` emits a targeted compile-time error if some code path tries
     * to fetch a direct-parent value when none exists.
     */
    template <typename T>
    constexpr auto get() const -> T& {
        static_assert(detail::no_prev_always_false_v<T>,
                      "no_prev does not carry a direct parent value");
    }
};

/**
 * @brief Lightweight borrowed view over a direct parent output slot.
 *
 * The view carries exactly one payload object and exposes a type-based API so
 * `resolve_as(from_prev_t<...>, ...)` can stay structurally similar to
 * `from_ctx(...)`.
 *
 * @tparam T Stored payload type. Constness is preserved; references are not.
 */
template <typename T>
struct prev_slot_view {
    using stored_type = std::remove_reference_t<T>;
    using type = std::remove_cv_t<stored_type>;

    static_assert(!std::is_void_v<type>,
                  "prev_slot_view<T> requires a non-void payload type");

    /// Borrowed reference to the direct parent payload.
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
    stored_type& ref;

    template <typename U>
    [[nodiscard]] static constexpr bool contains() noexcept {
        return std::is_same_v<std::remove_cv_t<U>, type>;
    }

    template <typename U>
    constexpr stored_type& get() const noexcept {
        static_assert(contains<U>(),
                      "Requested type is not present in the direct parent slot");
        return ref;
    }
};

/**
 * @brief Creates a borrowed view over a direct parent payload object.
 *
 * @tparam T Payload expression type.
 * @param value Payload object stored in the current direct parent slot.
 * @return A `prev_slot_view` borrowing `value`.
 */
template <typename T>
[[nodiscard]] constexpr auto prev_slot(T& value) noexcept
    -> prev_slot_view<T> {
    return {value};
}

/**
 * @brief Statically typed context container with a compile-time schema.
 *
 * `context<Ts...>` stores a fixed set of unique types in a `std::tuple`
 * and exposes them through type-based access instead of runtime lookup.
 *
 * @tparam Ts Types stored in the context. Each type must be unique.
 */
template <typename... Ts>
struct context {
    static_assert(detail::unique_types_v<Ts...>,
                  "yorch::context<Ts...> requires unique types");

    /// Underlying storage whose element order matches `Ts...`.
    std::tuple<Ts...> storage;

    /**
     * @brief Default-constructs the context.
     *
     * This constructor is available only when every type in `Ts...`
     * is default-initializable.
     */
    constexpr context()
        requires (std::default_initializable<Ts> && ...)
    = default;

    /**
     * @brief Explicitly constructs the context from the provided objects.
     * @param xs Arguments forwarded to the underlying tuple in `Ts...` order.
     */
    template <typename... Us>
        requires (sizeof...(Us) == sizeof...(Ts)) &&
                 std::constructible_from<std::tuple<Ts...>, Us&&...>
    constexpr explicit context(Us&&... xs)
        : storage(std::forward<Us>(xs)...) {}

    /**
     * @brief Checks whether a type is present in this context schema.
     * @tparam T Type to query.
     * @return `true` if and only if `T` appears exactly once in `Ts...`.
     */
    template <typename T>
    [[nodiscard]] static constexpr bool contains() noexcept {
        return detail::type_count_v<T, Ts...> == 1;
    }

    /**
     * @brief Returns the stored object of type `T`.
     * @tparam T Type to access.
     * @return A reference to the stored object, preserving the value category
     * and const qualification of the current `context`.
     */
    template <typename T>
    constexpr T& get() & noexcept {
        static_assert(contains<T>(),
                      "T must appear exactly once in yorch::context");
        return std::get<T>(storage);
    }

    template <typename T>
    constexpr const T& get() const& noexcept {
        static_assert(contains<T>(),
                      "T must appear exactly once in yorch::context");
        return std::get<T>(storage);
    }

    template <typename T>
    constexpr T&& get() && noexcept {
        static_assert(contains<T>(),
                      "T must appear exactly once in yorch::context");
        return std::get<T>(std::move(storage));
    }

    template <typename T>
    constexpr const T&& get() const&& noexcept {
        static_assert(contains<T>(),
                      "T must appear exactly once in yorch::context");
        return std::get<T>(std::move(storage));
    }
};

/**
 * @brief Lightweight borrowed view used during execution.
 *
 * This type does not own the context object. It only exposes an external
 * `context` to execution and resolution logic for the duration of a run.
 * The caller must ensure that `ctx` outlives the `exec_context`.
 *
 * @tparam Ctx Borrowed context type.
 */
template <typename Ctx, typename Prev = no_prev>
struct exec_context {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
    Ctx& ctx;

    /// Optional direct-parent payload view used by `from_prev(...)`.
    Prev prev;

    [[nodiscard]] constexpr Prev& prev_view() & noexcept {
        return prev;
    }

    [[nodiscard]] constexpr const Prev& prev_view() const& noexcept {
        return prev;
    }
};

template <typename Ctx>
struct exec_context<Ctx, no_prev> {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
    Ctx& ctx;

    [[nodiscard]] constexpr no_prev prev_view() const noexcept {
        return {};
    }
};

/// @brief Indicates that the current execution path carries no context.
template <>
struct exec_context<void, no_prev> {
    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    [[nodiscard]] constexpr no_prev prev_view() const noexcept {
        return {};
    }
};

template <typename Prev>
struct exec_context<void, Prev> {
    /// Optional direct-parent payload view used by `from_prev(...)`.
    Prev prev;

    [[nodiscard]] constexpr Prev& prev_view() & noexcept {
        return prev;
    }

    [[nodiscard]] constexpr const Prev& prev_view() const& noexcept {
        return prev;
    }
};

}  // namespace yorch
