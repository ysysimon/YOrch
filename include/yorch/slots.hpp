#pragma once

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "context.hpp"
#include "detail/maybe_storage.hpp"

namespace yorch::detail {

/**
 * @brief Fixed in-place payload slot for one statically known node output type.
 *
 * `typed_slot<T>` is a small storage primitive used by plan-compiled execution.
 * It owns exactly one optional in-place `T`, then explicitly starts and ends
 * the payload lifetime with `emplace(...)` and `destroy()`.
 *
 * This is intentionally not modeled as "allocate a `T` with runtime `new`":
 * the slot itself is part of the static plan shape, while the payload object is
 * only materialized when the node actually produces a value. That keeps storage
 * local to the plan, avoids per-value heap allocation, and gives later
 * executors precise control over parent-payload lifetime during subtree
 * traversal.
 *
 * @tparam T Concrete payload type stored in this slot. Must not be `void`.
 */
template <typename T>
struct typed_slot {
    static_assert(!std::is_void_v<T>,
                  "yorch::detail::typed_slot<T> requires a non-void type");

    typed_slot() = default;
    typed_slot(const typed_slot&) = delete;
    typed_slot& operator=(const typed_slot&) = delete;
    typed_slot(typed_slot&&) = delete;
    typed_slot& operator=(typed_slot&&) = delete;

    ~typed_slot() = default;

    /**
     * @brief Constructs a `T` in this slot's in-place storage.
     *
     * This operation is only available for an empty slot. Its exception
     * specification intentionally mirrors `T`'s constructor: if constructing
     * the payload may throw, `emplace(...)` may also throw. In other words,
     * this storage layer does not by itself guarantee an end-to-end no-throw
     * execution path; later executor stages must impose any stronger
     * "payload write must be noexcept" rule they require.
     *
     * @tparam Args Constructor argument types forwarded to `T`.
     * @param args Arguments used to construct the payload in place.
     * @return A reference to the newly constructed payload object.
     */
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

private:
    detail::maybe_storage<T> storage_ {};
};

template <>
struct typed_slot<void> {
    typed_slot() = default;
    typed_slot(const typed_slot&) = delete;
    typed_slot& operator=(const typed_slot&) = delete;
    typed_slot(typed_slot&&) = delete;
    typed_slot& operator=(typed_slot&&) = delete;
    ~typed_slot() = default;

    constexpr void destroy() noexcept {}

    [[nodiscard]] static constexpr bool has_value() noexcept {
        return false;
    }
};

/**
 * @brief Expands a plan's per-node payload types into a `typed_slot` tuple type.
 *
 * This declaration is used purely as a compile-time type bridge. Given a
 * `Plan` and an index pack `I...`, it spells the tuple type that stores one
 * `typed_slot<Plan::output_type<I>>` per node in plan order.
 *
 * The function is intentionally declared only for use with `decltype(...)`;
 * it is not meant to be called at runtime.
 *
 * @tparam Plan Compiled plan type exposing `output_type<I>`.
 * @tparam I Node indices to expand.
 * @param Unused compile-time index sequence selecting the nodes to expand.
 * @return `std::tuple<typed_slot<typename Plan::output_type<I>>...>`
 */
template <typename Plan, std::size_t... I>
[[nodiscard]] auto make_plan_slots_tuple(std::index_sequence<I...>)
    -> std::tuple<typed_slot<typename Plan::template output_type<I>>...>;

/**
 * @brief Tuple storage type holding one `typed_slot` per node of `Plan`.
 *
 * This alias materializes the full per-node slot layout by expanding
 * `Plan::output_type<I>` over the range `[0, Plan::node_count)`.
 *
 * @tparam Plan Compiled plan type whose node outputs define the slot types.
 */
template <typename Plan>
using plan_slots_tuple_t =
    decltype(make_plan_slots_tuple<Plan>(std::make_index_sequence<Plan::node_count> {}));

/**
 * @brief Reports whether node `I` of `Plan` carries a concrete payload slot.
 *
 * Nodes whose `output_type<I>` is `void` do not expose payload operations such
 * as `emplace<I>()` or `get<I>()`.
 *
 * @tparam Plan Compiled plan type exposing `output_type<I>`.
 * @tparam I Node index within `Plan`.
 */
template <typename Plan, std::size_t I>
concept payload_node =
    !std::is_void_v<typename Plan::template output_type<I>>;

} // namespace yorch::detail

namespace yorch {

/**
 * @brief Per-plan payload storage with one logical slot per node.
 *
 * `plan_slots<Plan>` is the current storage backend for compiled tree plans.
 * It materializes one `typed_slot<Plan::output_type<I>>` for each node and
 * exposes node-indexed accessors such as `emplace<I>()`, `get<I>()`, and
 * `prev_view_for<I>()`.
 *
 * In the current phase this storage is intentionally not compacted or reused:
 * when a node payload lifetime ends, `destroy<I>()` only destroys the live
 * object inside that slot. The slot itself remains part of the fixed plan
 * layout and is not reassigned to another node. This keeps lifetime semantics
 * simple while `from_prev(...)` and serial subtree execution are being settled.
 *
 * `slot_index<I>` is still exposed as part of the plan-facing contract even
 * though it currently matches the node index. That indirection is preserved so
 * later phases can introduce slot reuse or storage compaction without changing
 * the public node-indexed API of `plan_slots`.
 *
 * @tparam Plan Compiled plan type providing `node_count`, `output_type<I>`,
 * and `slot_index<I>`.
 */
template <typename Plan>
struct plan_slots {
    using tuple_type = detail::plan_slots_tuple_t<Plan>;

    tuple_type storage {};

    template <std::size_t I>
    using output_type = typename Plan::template output_type<I>;

    template <std::size_t I>
    static constexpr std::size_t slot_index = Plan::template slot_index<I>;

    template <std::size_t I>
    [[nodiscard]] constexpr bool has_value() const noexcept {
        if constexpr (std::is_void_v<output_type<I>>) {
            return false;
        } else {
            return slot<I>().has_value();
        }
    }

    /**
     * @brief Constructs node `I`'s payload object inside its assigned slot.
     *
     * This member forwards directly to the underlying `typed_slot`, so its
     * exception specification truthfully follows the payload construction path.
     * If `output_type<I>` is nothrow-constructible from `Args...`, then this
     * call is `noexcept`; otherwise it may throw. The current storage layer
     * does not normalize or catch such exceptions on its own.
     *
     * @tparam I Node index in `Plan`.
     * @tparam Args Constructor argument types forwarded to the payload object.
     * @param args Arguments used to construct the payload in place.
     * @return A reference to the newly constructed node payload.
     */
    template <std::size_t I, typename... Args>
        requires detail::payload_node<Plan, I>
    constexpr auto emplace(Args&&... args)
        noexcept(noexcept(slot<I>().emplace(std::forward<Args>(args)...)))
        -> output_type<I>& {
        return slot<I>().emplace(std::forward<Args>(args)...);
    }

    template <std::size_t I>
        requires detail::payload_node<Plan, I>
    [[nodiscard]] constexpr auto get() & noexcept -> output_type<I>& {
        return slot<I>().get();
    }

    template <std::size_t I>
        requires detail::payload_node<Plan, I>
    [[nodiscard]] constexpr auto get() const& noexcept -> const output_type<I>& {
        return slot<I>().get();
    }

    template <std::size_t I>
    constexpr void destroy() noexcept {
        slot<I>().destroy();
    }

    template <std::size_t I>
    [[nodiscard]] constexpr auto prev_view_for() & {
        if constexpr (std::is_void_v<output_type<I>>) {
            return no_prev {};
        } else {
            return prev_slot(get<I>());
        }
    }

    template <std::size_t I>
    [[nodiscard]] constexpr auto prev_view_for() const& {
        if constexpr (std::is_void_v<output_type<I>>) {
            return no_prev {};
        } else {
            return prev_slot(get<I>());
        }
    }

private:
    template <std::size_t I>
    using slot_type = std::tuple_element_t<slot_index<I>, tuple_type>;

    template <std::size_t I>
    [[nodiscard]] constexpr auto slot() & noexcept -> slot_type<I>& {
        return std::get<slot_index<I>>(storage);
    }

    template <std::size_t I>
    [[nodiscard]] constexpr auto slot() const& noexcept -> const slot_type<I>& {
        return std::get<slot_index<I>>(storage);
    }
};

} // namespace yorch
