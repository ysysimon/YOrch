#pragma once

#include <cstddef>

#include "policy.hpp"

namespace yorch::detail {

template <std::size_t Size, std::size_t Align, slot_physical_policy Policy>
struct erased_slot;

template <std::size_t Size, std::size_t Align>
struct erased_slot<Size, Align, slot_physical_policy::maybe_payload> {
    static constexpr slot_physical_policy physical_policy = slot_physical_policy::maybe_payload;

    [[nodiscard]] constexpr std::byte* raw_storage() noexcept {
        return storage_;
    }

    [[nodiscard]] constexpr const std::byte* raw_storage() const noexcept {
        return storage_;
    }

    [[nodiscard]] constexpr bool* engaged_ptr() noexcept {
        return &engaged_;
    }

    [[nodiscard]] constexpr const bool* engaged_ptr() const noexcept {
        return &engaged_;
    }

    [[nodiscard]] constexpr std::size_t* owner_node_ptr() noexcept {
        return &owner_node_;
    }

    [[nodiscard]] constexpr const std::size_t* owner_node_ptr() const noexcept {
        return &owner_node_;
    }

private:
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays, modernize-avoid-c-arrays)
    alignas(Align) std::byte storage_[Size] {};
    bool engaged_ = false;
    std::size_t owner_node_ = no_physical_slot;
};

template <std::size_t Size, std::size_t Align>
struct erased_slot<Size, Align, slot_physical_policy::must_payload> {
    static constexpr slot_physical_policy physical_policy = slot_physical_policy::must_payload;

    [[nodiscard]] constexpr std::byte* raw_storage() noexcept {
        return storage_;
    }

    [[nodiscard]] constexpr const std::byte* raw_storage() const noexcept {
        return storage_;
    }

    [[nodiscard]] static constexpr bool* engaged_ptr() noexcept {
        return nullptr;
    }

    [[nodiscard]] constexpr std::size_t* owner_node_ptr() noexcept {
        return &owner_node_;
    }

    [[nodiscard]] constexpr const std::size_t* owner_node_ptr() const noexcept {
        return &owner_node_;
    }

private:
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays, modernize-avoid-c-arrays)
    alignas(Align) std::byte storage_[Size] {};
    std::size_t owner_node_ = no_physical_slot;
};

} // namespace yorch::detail
