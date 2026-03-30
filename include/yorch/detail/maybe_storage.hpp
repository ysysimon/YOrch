#pragma once

#include <cstddef>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

#include "yorch/assert.hpp"

namespace yorch::detail {

/**
 * @brief Manual-lifetime storage for an optional in-place `T`.
 *
 * This helper underpins both public task-return wrappers and per-plan typed
 * slots. It owns raw storage plus an engagement flag, then explicitly starts
 * and ends `T`'s lifetime via `emplace(...)` and `destroy()`.
 *
 * @tparam T Stored non-reference, non-void object type.
 */
template <typename T>
struct maybe_storage {
    static_assert(!std::is_reference_v<T>,
                  "yorch::detail::maybe_storage<T> does not support reference types");
    static_assert(!std::is_void_v<T>,
                  "yorch::detail::maybe_storage<T> requires a non-void type");

public:
    constexpr maybe_storage() noexcept = default;

    constexpr maybe_storage(const maybe_storage& other)
        noexcept(std::is_nothrow_copy_constructible_v<T>)
        requires std::copy_constructible<T>
    {
        if (other.has_value()) {
            emplace(other.get());
        }
    }

    constexpr maybe_storage(const maybe_storage&)
        requires (!std::copy_constructible<T>)
        = delete;

    constexpr maybe_storage(maybe_storage&& other)
        noexcept(std::is_nothrow_move_constructible_v<T>)
        requires std::move_constructible<T>
    {
        if (other.has_value()) {
            emplace(std::move(other.get()));
        }
    }

    constexpr maybe_storage(maybe_storage&&)
        requires (!std::move_constructible<T>)
        = delete;

    constexpr maybe_storage& operator=(const maybe_storage& other)
        noexcept(std::is_nothrow_copy_constructible_v<T> &&
                 std::is_nothrow_copy_assignable_v<T>)
        requires std::copy_constructible<T>
    {
        if (this == &other) {
            return *this;
        }

        assign_from(other);
        return *this;
    }

    constexpr maybe_storage& operator=(const maybe_storage&)
        requires (!std::copy_constructible<T>)
        = delete;

    constexpr maybe_storage& operator=(maybe_storage&& other)
        noexcept(std::is_nothrow_move_constructible_v<T> &&
                 std::is_nothrow_move_assignable_v<T>)
        requires std::move_constructible<T>
    {
        if (this == &other) {
            return *this;
        }

        assign_from(std::move(other));
        return *this;
    }

    constexpr maybe_storage& operator=(maybe_storage&&)
        requires (!std::move_constructible<T>)
        = delete;

    ~maybe_storage() {
        destroy();
    }

    template <typename... Args>
    constexpr T& emplace(Args&&... args)
        noexcept(std::is_nothrow_constructible_v<T, Args&&...>)
    {
        YORCH_ASSERT(!engaged_ &&
                     "yorch::detail::maybe_storage<T>::emplace() called on a live value");

        auto* object = std::construct_at(ptr(), std::forward<Args>(args)...);
        engaged_ = true;
        return *object;
    }

    constexpr void destroy() noexcept {
        if (!engaged_) {
            return;
        }

        std::destroy_at(ptr());
        engaged_ = false;
    }

    [[nodiscard]] constexpr bool has_value() const noexcept {
        return engaged_;
    }

    [[nodiscard]] constexpr T& get() & noexcept {
        YORCH_ASSERT(engaged_ &&
                     "yorch::detail::maybe_storage<T>::get() called on an empty value");
        return *ptr();
    }

    [[nodiscard]] constexpr const T& get() const& noexcept {
        YORCH_ASSERT(engaged_ &&
                     "yorch::detail::maybe_storage<T>::get() called on an empty value");
        return *ptr();
    }

    [[nodiscard]] constexpr T&& get() && noexcept {
        YORCH_ASSERT(engaged_ &&
                     "yorch::detail::maybe_storage<T>::get() called on an empty value");
        return std::move(*ptr());
    }

private:
    template <typename Storage>
    constexpr void assign_from(Storage&& other) {
        if (other.has_value()) {
            if (engaged_) {
                if constexpr (std::is_assignable_v<T&, decltype(std::forward<Storage>(other).get())>) {
                    get() = std::forward<Storage>(other).get();
                } else {
                    destroy();
                    emplace(std::forward<Storage>(other).get());
                }
            } else {
                emplace(std::forward<Storage>(other).get());
            }
        } else {
            destroy();
        }
    }

    [[nodiscard]] constexpr T* ptr() noexcept {
        return std::launder(reinterpret_cast<T*>(storage_));
    }

    [[nodiscard]] constexpr const T* ptr() const noexcept {
        return std::launder(reinterpret_cast<const T*>(storage_));
    }

    alignas(T) std::byte storage_[sizeof(T)] {};
    bool engaged_ = false;
};

} // namespace yorch::detail
