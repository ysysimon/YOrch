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

} // namespace yorch::detail

namespace yorch {

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
    constexpr T& get() & {
        static_assert(contains<T>(),
                      "T must appear exactly once in yorch::context");
        return std::get<T>(storage);
    }

    template <typename T>
    constexpr const T& get() const& {
        static_assert(contains<T>(),
                      "T must appear exactly once in yorch::context");
        return std::get<T>(storage);
    }

    template <typename T>
    constexpr T&& get() && {
        static_assert(contains<T>(),
                      "T must appear exactly once in yorch::context");
        return std::get<T>(std::move(storage));
    }

    template <typename T>
    constexpr const T&& get() const&& {
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
template <typename Ctx>
struct exec_context {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
    Ctx& ctx;
};

/// @brief Indicates that the current execution path carries no context.
template <>
struct exec_context<void> {};

}  // namespace yorch
