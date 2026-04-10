#pragma once

#include <type_traits>
#include <utility>

#include "../detail/slots/slot_view.hpp"
#include "../detail/slots/typed_slot.hpp"
#include "../result.hpp"

namespace yorch {

/**
 * @brief Output sink passed to direct-output tasks.
 *
 * `direct_out<T>` models maybe-payload semantics: receiving the sink does not
 * guarantee that the task will call `emplace(...)` before returning. The sink
 * always materializes an owned payload in the destination slot; it does not
 * model alias/reference output semantics. This is why it only accepts
 * maybe-capable storage and stores a `slot_view<T>` instead of a plain slot
 * reference; the same public sink can target a typed one-to-one slot or an
 * erased compact-layout slot.
 *
 * If a task needs to mutate a direct-parent payload and then forward that same
 * object into its own output slot, prefer `consume_prev<T>()` plus
 * `out.success(std::move(value))` so the ownership transfer stays explicit.
 */
template <typename T>
struct direct_out {
    static_assert(!std::is_reference_v<T>,
                  "yorch::direct_out<T> does not support reference types; direct-output payloads must be owned values");
    static_assert(!std::is_void_v<T>,
                  "yorch::direct_out<T> requires a non-void payload type");

    constexpr explicit direct_out(
        detail::typed_slot<T, detail::slot_logical_policy::maybe_payload>& slot) noexcept
        : slot_(slot) {}

    /**
     * @brief Binds the sink to an already-created typed slot view.
     */
    constexpr explicit direct_out(detail::slot_view<T> slot) noexcept
        : slot_(slot) {}

    template <typename... Args>
    constexpr T& emplace(Args&&... args)
        noexcept(noexcept(slot_.emplace(std::forward<Args>(args)...))) {
        return slot_.emplace(std::forward<Args>(args)...);
    }

    template <typename... Args>
    [[nodiscard]] constexpr step_result success(Args&&... args)
        noexcept(noexcept(emplace(std::forward<Args>(args)...))) {
        emplace(std::forward<Args>(args)...);
        return step_result::success();
    }

    constexpr void destroy() noexcept {
        slot_.destroy();
    }

    [[nodiscard]] constexpr bool has_value() const noexcept {
        return slot_.has_value();
    }

private:
    detail::slot_view<T> slot_ {};
};

} // namespace yorch
