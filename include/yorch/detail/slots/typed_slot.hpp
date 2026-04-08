#pragma once

#include <cstddef>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

#include "../storage/maybe_storage.hpp"
#include "policy.hpp"

namespace yorch::detail {

template <typename T, slot_logical_policy Policy = slot_logical_policy::maybe_payload>
struct typed_slot {
    static_assert(!std::is_void_v<T>,
                  "yorch::detail::typed_slot<T> requires a non-void type");
    static_assert(Policy == slot_logical_policy::maybe_payload ||
                      Policy == slot_logical_policy::must_payload,
                  "yorch::detail::typed_slot<T> only supports maybe_payload or must_payload policies");

    static constexpr slot_physical_policy physical_policy =
        Policy == slot_logical_policy::must_payload
            ? slot_physical_policy::must_payload
            : slot_physical_policy::maybe_payload;

    typed_slot() = default;
    typed_slot(const typed_slot&) = delete;
    typed_slot& operator=(const typed_slot&) = delete;
    typed_slot(typed_slot&&) = delete;
    typed_slot& operator=(typed_slot&&) = delete;

    ~typed_slot() = default;

    template <typename... Args>
    constexpr T& emplace(Args&&... args)
        noexcept(noexcept(storage_.emplace(std::forward<Args>(args)...))) {
        return storage_.emplace(std::forward<Args>(args)...);
    }

    [[nodiscard]] constexpr T& get() & noexcept {
        return storage_.get();
    }

    [[nodiscard]] constexpr const T& get() const& noexcept {
        return storage_.get();
    }

    constexpr void destroy() noexcept {
        storage_.destroy();
    }

    [[nodiscard]] constexpr bool has_value() const noexcept {
        return storage_.has_value();
    }

    [[nodiscard]] constexpr std::byte* raw_storage() noexcept {
        return reinterpret_cast<std::byte*>(storage_.raw_ptr());
    }

    [[nodiscard]] constexpr const std::byte* raw_storage() const noexcept {
        return reinterpret_cast<const std::byte*>(storage_.raw_ptr());
    }

    [[nodiscard]] constexpr bool* engaged_ptr() noexcept {
        return storage_.engaged_ptr();
    }

    [[nodiscard]] constexpr const bool* engaged_ptr() const noexcept {
        return storage_.engaged_ptr();
    }

    [[nodiscard]] static constexpr std::size_t* owner_node_ptr() noexcept {
        return nullptr;
    }

private:
    detail::maybe_storage<T> storage_ {};
};

template <>
struct typed_slot<void, slot_logical_policy::none> {
    static constexpr slot_physical_policy physical_policy = slot_physical_policy::none;

    typed_slot() = default;
    typed_slot(const typed_slot&) = delete;
    typed_slot& operator=(const typed_slot&) = delete;
    typed_slot(typed_slot&&) = delete;
    typed_slot& operator=(typed_slot&&) = delete;
    ~typed_slot() = default;

    constexpr void destroy() noexcept {}

    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    [[nodiscard]] constexpr bool has_value() const noexcept {
        return false;
    }
};

template <typename T>
struct typed_slot<T, slot_logical_policy::must_payload> {
    static_assert(!std::is_void_v<T>,
                  "yorch::detail::typed_slot<T, must_payload> requires a non-void type");

    static constexpr slot_physical_policy physical_policy = slot_physical_policy::must_payload;

    typed_slot() = default;
    typed_slot(const typed_slot&) = delete;
    typed_slot& operator=(const typed_slot&) = delete;
    typed_slot(typed_slot&&) = delete;
    typed_slot& operator=(typed_slot&&) = delete;

    ~typed_slot() = default;

    template <typename... Args>
    constexpr T& emplace(Args&&... args)
        noexcept(std::is_nothrow_constructible_v<T, Args&&...>) {
        return *std::construct_at(ptr(), std::forward<Args>(args)...);
    }

    [[nodiscard]] constexpr T& get() & noexcept {
        return *ptr();
    }

    [[nodiscard]] constexpr const T& get() const& noexcept {
        return *ptr();
    }

    constexpr void destroy() noexcept {
        std::destroy_at(ptr());
    }

    [[nodiscard]] constexpr std::byte* raw_storage() noexcept {
        return storage_;
    }

    [[nodiscard]] constexpr const std::byte* raw_storage() const noexcept {
        return storage_;
    }

    [[nodiscard]] static constexpr bool* engaged_ptr() noexcept {
        return nullptr;
    }

    [[nodiscard]] static constexpr std::size_t* owner_node_ptr() noexcept {
        return nullptr;
    }

private:
    [[nodiscard]] constexpr T* ptr() noexcept {
        return std::launder(reinterpret_cast<T*>(storage_));
    }

    [[nodiscard]] constexpr const T* ptr() const noexcept {
        return std::launder(reinterpret_cast<const T*>(storage_));
    }

    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays, modernize-avoid-c-arrays)
    alignas(T) std::byte storage_[sizeof(T)] {};
};

} // namespace yorch::detail
