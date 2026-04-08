#pragma once

#include <cstddef>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

#include "../../assert.hpp"
#include "policy.hpp"

namespace yorch::detail {

/**
 * @brief Non-owning typed access view over a slot-like storage object.
 *
 * `slot_view<T>` lets higher layers treat either a `typed_slot<T, ...>` or an
 * erased compact-layout slot as storage for `T`. It carries the physical policy
 * because maybe slots need an engagement flag while must slots do not. For
 * erased compact slots, it also carries owner-node tracking so a later cleanup
 * pass can know which node's `T` currently lives inside the shared raw storage.
 */
template <typename T>
struct slot_view {
    static_assert(!std::is_void_v<T>,
                  "yorch::detail::slot_view<T> requires a non-void type");

    std::byte* storage = nullptr;
    bool* engaged = nullptr;
    // Points at the owning erased slot's persistent owner-node record.
    std::size_t* slot_owner_node = nullptr;
    // The node id this transient view writes into `slot_owner_node`.
    std::size_t view_owner_node = no_physical_slot;
    slot_physical_policy policy = slot_physical_policy::maybe_payload;

    /**
     * @brief Creates a typed view over `slot`.
     *
     * `owner_node` is written into the underlying owner-node record when the
     * view materializes a value. Typed one-to-one slots return a null owner
     * record, while erased compact slots use it to remember which node owns the
     * currently live object.
     */
    template <typename Slot>
        requires (!std::is_same_v<std::remove_cvref_t<Slot>, slot_view>)
    constexpr explicit slot_view(Slot& slot, std::size_t owner_node = no_physical_slot) noexcept
        : storage(slot.raw_storage()),
          engaged(slot.engaged_ptr()),
          slot_owner_node(slot.owner_node_ptr()),
          view_owner_node(owner_node),
          policy(Slot::physical_policy) {}

    slot_view() = default;

    template <typename... Args>
    constexpr T& emplace(Args&&... args)
        noexcept(std::is_nothrow_constructible_v<T, Args&&...>) {
        if (policy == slot_physical_policy::maybe_payload) {
            YORCH_ASSERT(engaged != nullptr &&
                         "yorch::detail::slot_view<T> maybe_payload slots require an engagement flag");
            YORCH_ASSERT(!*engaged &&
                         "yorch::detail::slot_view<T>::emplace() called on a live value");
            auto* object = std::construct_at(ptr(), std::forward<Args>(args)...);
            *engaged = true;
            if (slot_owner_node != nullptr) {
                *slot_owner_node = view_owner_node;
            }
            return *object;
        }

        auto* object = std::construct_at(ptr(), std::forward<Args>(args)...);
        if (slot_owner_node != nullptr) {
            *slot_owner_node = view_owner_node;
        }
        return *object;
    }

    [[nodiscard]] constexpr T& get() & noexcept {
        if (policy == slot_physical_policy::maybe_payload) {
            YORCH_ASSERT(engaged != nullptr && *engaged &&
                         "yorch::detail::slot_view<T>::get() called on an empty value");
        }

        return *ptr();
    }

    [[nodiscard]] constexpr const T& get() const& noexcept {
        if (policy == slot_physical_policy::maybe_payload) {
            YORCH_ASSERT(engaged != nullptr && *engaged &&
                         "yorch::detail::slot_view<T>::get() called on an empty value");
        }

        return *ptr();
    }

    constexpr void destroy() noexcept {
        if (policy == slot_physical_policy::maybe_payload) {
            if (engaged == nullptr || !*engaged) {
                return;
            }

            std::destroy_at(ptr());
            *engaged = false;
            if (slot_owner_node != nullptr) {
                *slot_owner_node = no_physical_slot;
            }
            return;
        }

        std::destroy_at(ptr());
        if (slot_owner_node != nullptr) {
            *slot_owner_node = no_physical_slot;
        }
    }

    [[nodiscard]] constexpr bool has_value() const noexcept {
        return policy == slot_physical_policy::maybe_payload
            ? (engaged != nullptr && *engaged)
            : true;
    }

private:
    [[nodiscard]] constexpr T* ptr() noexcept {
        return std::launder(reinterpret_cast<T*>(storage));
    }

    [[nodiscard]] constexpr const T* ptr() const noexcept {
        return std::launder(reinterpret_cast<const T*>(storage));
    }
};

} // namespace yorch::detail
