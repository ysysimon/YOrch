#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

#include "../../assert.hpp"
#include "../../result.hpp"

namespace yorch::detail {

/**
 * @brief Normalizes a raw task return into scheduler-facing `step_result`.
 *
 * Plain value payloads are treated as successful completion, `step_result`
 * passes through unchanged, and `task_result<T>` contributes only its control
 * status here. Payload extraction itself is handled by later execution stages.
 *
 * @tparam R Raw return object type.
 * @param r Raw return object emitted by `invoke_raw(...)`.
 * @return Normalized `step_result`.
 */
template <typename R>
[[nodiscard]] constexpr step_result normalize_task_result(R&& r) { // NOLINT(readability-identifier-length)
    using raw_t = std::remove_cvref_t<R>;

    if constexpr (std::is_same_v<raw_t, step_result>) {
        return std::forward<R>(r);
    } else if constexpr (is_task_result_v<raw_t>) {
        return std::forward<R>(r).step;
    } else {
        static_cast<void>(r);
        return step_result::success();
    }
}

template <typename R>
[[nodiscard]] constexpr step_result extract_step_result(R&& r) { // NOLINT(readability-identifier-length)
    return normalize_task_result(std::forward<R>(r));
}

/**
 * @brief Materializes the payload portion of a node's raw task return into the
 * node's compiled slot, when that return category actually carries a value.
 *
 * This helper is the value-storage counterpart to `extract_step_result(...)`.
 * The executor first derives control-flow status from the raw return, then
 * uses this function to decide whether the current node produced a payload that
 * should remain alive for its subtree and therefore be written into slot `I`.
 *
 * The storage rules are:
 *
 * - `void` and `step_result` do not carry a payload, so no slot write happens
 * - plain `T` writes that returned value directly into slot `I`
 * - `task_result<T>` writes only the wrapped `value` and only when the step
 *   status is `success`; non-success `task_result<T>` values leave the slot
 *   empty
 *
 * Payloads are moved into the slot because the slot becomes the owning storage
 * for the remainder of the node's subtree traversal.
 *
 * @tparam I Node index whose slot should receive the payload.
 * @tparam Slots Concrete `plan_exec_slots<...>`-like storage type.
 * @tparam Raw Raw return object type produced by `invoke_raw(...)`.
 * @param slots Plan slot storage used by the current DFS execution.
 * @param raw Raw task return object to inspect and, when applicable, move from.
 */
template <std::size_t I, typename Slots, typename Raw>
[[nodiscard]] constexpr bool store_node_output(Slots& slots, Raw& raw) {
    using raw_t = std::remove_cvref_t<Raw>;

    if constexpr (std::is_void_v<raw_t> ||
                  std::is_same_v<raw_t, step_result>) {
        static_cast<void>(slots);
        static_cast<void>(raw);
        return false;
    } else if constexpr (is_task_result_v<raw_t>) {
        if (raw.step.ok()) {
            YORCH_ASSERT(raw.has_value() &&
                         "Successful task_result<T> must carry a payload before slot storage");
            slots.template emplace<I>(std::move(raw).value());
            return true;
        }

        return false;
    } else {
        slots.template emplace<I>(std::move(raw));
        return true;
    }
}

template <typename Slots, std::size_t I>
struct node_slot_guard {
    Slots& slots;
    bool armed = false;

    constexpr explicit node_slot_guard(Slots& stored_slots) noexcept
        : slots(stored_slots) {}

    node_slot_guard(const node_slot_guard&) = delete;
    node_slot_guard& operator=(const node_slot_guard&) = delete;
    node_slot_guard(node_slot_guard&&) = delete;
    node_slot_guard& operator=(node_slot_guard&&) = delete;

    constexpr void arm() noexcept {
        armed = true;
    }

    ~node_slot_guard() {
        if (armed) {
            slots.template destroy<I>();
        }
    }
};

template <std::size_t I, typename Slots>
[[nodiscard]] constexpr step_result finalize_direct_output_step(Slots& slots, step_result step) {
    if (step.ok()) {
        YORCH_ASSERT(slots.template has_value<I>() &&
                     "Direct-output tasks returning success must construct their payload");
    } else {
        slots.template destroy<I>();
    }

    return step;
}

} // namespace yorch::detail
