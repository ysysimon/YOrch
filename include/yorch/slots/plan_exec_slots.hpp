#pragma once

#include <array>
#include <cstddef>
#include <new>
#include <tuple>
#include <type_traits>
#include <utility>

#include "../assert.hpp"
#include "../context.hpp"
#include "../detail/slots/layout.hpp"
#include "policies.hpp"
#include "result_out.hpp"

namespace yorch {

namespace detail {

/**
 * @brief Implementation backend for layout-driven erased plan payload storage.
 *
 * This implementation is used by compact layouts such as
 * `slot_layout_serial_dfs_compact_policy`. It stores physical slots as
 * type-erased cells sized by the selected layout, then exposes a node-indexed
 * API (`emplace<I>`, `get<I>`, `destroy<I>`, `out<I>`, `prev_view_for<I>`) to
 * the executor.
 *
 * The storage tuple is intentionally private so layout/backend details do not
 * become part of the public API.
 */
template <typename Plan, typename LayoutPolicy>
    requires detail::slot_layout_policy<LayoutPolicy>
struct plan_exec_slots_impl {
    using layout_type = detail::plan_slot_layout<Plan, LayoutPolicy>;
    using tuple_type = detail::plan_erased_slots_tuple_t<layout_type>;

    plan_exec_slots_impl() = default;
    plan_exec_slots_impl(const plan_exec_slots_impl&) = delete;
    plan_exec_slots_impl& operator=(const plan_exec_slots_impl&) = delete;
    plan_exec_slots_impl(plan_exec_slots_impl&&) = delete;
    plan_exec_slots_impl& operator=(plan_exec_slots_impl&&) = delete;
    ~plan_exec_slots_impl() {
        destroy_active_slots();
    }

    template <std::size_t I>
    using output_type = typename Plan::template output_type<I>;

    template <std::size_t I>
    static constexpr std::size_t slot_index = Plan::template slot_index<I>;

    /**
     * @brief Node-local logical slot policy inherited from the compiled plan.
     */
    template <std::size_t I>
    static constexpr detail::slot_logical_policy slot_logical_policy =
        Plan::template slot_logical_policy_for<I>;

    /**
     * @brief Number of physical slots allocated by the selected layout.
     */
    static constexpr std::size_t physical_slot_count = layout_type::physical_slot_count;

    /**
     * @brief Physical slot index assigned to node `I`.
     */
    template <std::size_t I>
    static constexpr std::size_t physical_slot_index =
        layout_type::template physical_slot_index<I>;

    /**
     * @brief Storage-level policy assigned to physical slot `K`.
     */
    template <std::size_t K>
    static constexpr detail::slot_physical_policy slot_physical_policy =
        layout_type::template physical_slot_policy<K>;

    /**
     * @brief Void-output nodes never have a stored payload value.
     */
    template <std::size_t I>
        requires std::is_void_v<output_type<I>>
    [[nodiscard]] constexpr bool has_value() const noexcept {
        return false;
    }

    /**
     * @brief Reports whether maybe-payload node `I` currently stores a value.
     */
    template <std::size_t I>
        requires (!std::is_void_v<output_type<I>>) &&
                 (slot_logical_policy<I> != detail::slot_logical_policy::must_payload)
    [[nodiscard]] constexpr bool has_value() const noexcept {
        if constexpr (slot_physical_policy<physical_slot_index<I>> == detail::slot_physical_policy::maybe_payload) {
            const auto& slot = physical_slot<I>();
            const auto* engaged = slot.engaged_ptr();
            return engaged != nullptr && *engaged;
        } else {
            return true;
        }
    }

    /**
     * @brief Constructs node `I`'s payload in its assigned slot.
     */
    template <std::size_t I, typename... Args>
        requires detail::payload_node<Plan, I>
    constexpr auto emplace(Args&&... args)
        noexcept(noexcept(slot_view_for<I>().emplace(std::forward<Args>(args)...)))
        -> output_type<I>& {
        auto slot = slot_view_for<I>();
        return slot.emplace(std::forward<Args>(args)...);
    }

    /**
     * @brief Returns a direct-output sink for maybe-payload node `I`.
     */
    template <std::size_t I>
        requires detail::payload_node<Plan, I>
              && (slot_logical_policy<I> == detail::slot_logical_policy::maybe_payload)
    [[nodiscard]] constexpr auto out() & noexcept -> result_out<output_type<I>> {
        return result_out<output_type<I>> {slot_view_for<I>()};
    }

    /**
     * @brief Returns node `I`'s stored payload.
     */
    template <std::size_t I>
        requires detail::payload_node<Plan, I>
    [[nodiscard]] constexpr auto get() & noexcept -> output_type<I>& {
        auto slot = slot_view_for<I>();
        return slot.get();
    }

    /**
     * @brief Returns node `I`'s stored payload from a const slots object.
     */
    template <std::size_t I>
        requires detail::payload_node<Plan, I>
    [[nodiscard]] constexpr auto get() const& noexcept -> const output_type<I>& {
        return *slot_ptr<I>();
    }

    /**
     * @brief Destroys node `I`'s payload if the node has one.
     */
    template <std::size_t I>
    constexpr void destroy() noexcept {
        if constexpr (!std::is_void_v<output_type<I>>) {
            auto slot = slot_view_for<I>();
            slot.destroy();
        }
    }

    /**
     * @brief Returns the `from_prev` view produced by node `I`'s payload.
     */
    template <std::size_t I>
    [[nodiscard]] constexpr auto prev_view_for() & {
        if constexpr (std::is_void_v<output_type<I>>) {
            return no_prev {};
        } else {
            return prev_slot(get<I>());
        }
    }

    /**
     * @brief Returns the const `from_prev` view produced by node `I`'s payload.
     */
    template <std::size_t I>
    [[nodiscard]] constexpr auto prev_view_for() const& {
        if constexpr (std::is_void_v<output_type<I>>) {
            return no_prev {};
        } else {
            return prev_slot(get<I>());
        }
    }

private:
    tuple_type storage {};

    template <std::size_t I>
    using physical_slot_type =
        std::tuple_element_t<physical_slot_index<I>, tuple_type>;

    template <std::size_t I>
    [[nodiscard]] constexpr auto physical_slot() & noexcept -> physical_slot_type<I>& {
        return std::get<physical_slot_index<I>>(storage);
    }

    template <std::size_t I>
    [[nodiscard]] constexpr auto physical_slot() const& noexcept -> const physical_slot_type<I>& {
        return std::get<physical_slot_index<I>>(storage);
    }

    template <std::size_t I>
        requires detail::payload_node<Plan, I>
    [[nodiscard]] constexpr auto slot_view_for() & noexcept -> detail::slot_view<output_type<I>> {
        return detail::slot_view<output_type<I>> {physical_slot<I>(), I};
    }

    template <std::size_t I>
        requires detail::payload_node<Plan, I>
    [[nodiscard]] constexpr auto slot_ptr() const noexcept -> const output_type<I>* {
        if constexpr (slot_physical_policy<physical_slot_index<I>> == detail::slot_physical_policy::maybe_payload) {
            const auto& slot = physical_slot<I>();
            const auto* engaged = slot.engaged_ptr();
            YORCH_ASSERT(engaged != nullptr && *engaged &&
                         "yorch::plan_exec_slots<...>::get() called on an empty slot");
        }

        return std::launder(reinterpret_cast<const output_type<I>*>(physical_slot<I>().raw_storage()));
    }

    template <std::size_t I = 0>
    constexpr void destroy_active_slots() noexcept {
        if constexpr (I < Plan::node_count) {
            if constexpr (detail::payload_node<Plan, I>) {
                auto& slot = physical_slot<I>();
                auto* owner_node = slot.owner_node_ptr();
                if (owner_node != nullptr && *owner_node == I) {
                    auto view = slot_view_for<I>();
                    view.destroy();
                }
            }

            destroy_active_slots<I + 1>();
        }
    }
};

/**
 * @brief Per-run plan payload storage for the default one-to-one layout.
 *
 * This specialization keeps each logical node in a typed slot, avoiding the
 * erased-slot indirection needed by compact layouts. Public behavior matches
 * the primary template, while private `live_slots_` tracking lets the
 * destructor safely clean up must-payload nodes that were manually emplaced.
 */
template <typename Plan>
struct plan_exec_slots_impl<Plan, slot_layout_one_to_one_policy> {
    using layout_type = detail::plan_slot_layout<Plan, slot_layout_one_to_one_policy>;
    using tuple_type = detail::plan_typed_slots_tuple_t<Plan>;

    plan_exec_slots_impl() = default;
    plan_exec_slots_impl(const plan_exec_slots_impl&) = delete;
    plan_exec_slots_impl& operator=(const plan_exec_slots_impl&) = delete;
    plan_exec_slots_impl(plan_exec_slots_impl&&) = delete;
    plan_exec_slots_impl& operator=(plan_exec_slots_impl&&) = delete;
    ~plan_exec_slots_impl() {
        destroy_active_slots();
    }

    template <std::size_t I>
    using output_type = typename Plan::template output_type<I>;

    template <std::size_t I>
    static constexpr std::size_t slot_index = Plan::template slot_index<I>;

    /**
     * @brief Node-local logical slot policy inherited from the compiled plan.
     */
    template <std::size_t I>
    static constexpr detail::slot_logical_policy slot_logical_policy =
        Plan::template slot_logical_policy_for<I>;

    /**
     * @brief Number of physical slots allocated by the one-to-one layout.
     */
    static constexpr std::size_t physical_slot_count = layout_type::physical_slot_count;

    /**
     * @brief Physical slot index assigned to node `I`.
     */
    template <std::size_t I>
    static constexpr std::size_t physical_slot_index =
        layout_type::template physical_slot_index<I>;

    /**
     * @brief Storage-level policy assigned to physical slot `K`.
     */
    template <std::size_t K>
    static constexpr detail::slot_physical_policy slot_physical_policy =
        layout_type::template physical_slot_policy<K>;

    /**
     * @brief Void-output nodes never have a stored payload value.
     */
    template <std::size_t I>
        requires std::is_void_v<output_type<I>>
    [[nodiscard]] constexpr bool has_value() const noexcept {
        return false;
    }

    /**
     * @brief Reports whether maybe-payload node `I` currently stores a value.
     */
    template <std::size_t I>
        requires (!std::is_void_v<output_type<I>>) &&
                 (slot_logical_policy<I> != detail::slot_logical_policy::must_payload)
    [[nodiscard]] constexpr bool has_value() const noexcept {
        return logical_slot<I>().has_value();
    }

    /**
     * @brief Constructs node `I`'s payload in its typed slot.
     */
    template <std::size_t I, typename... Args>
        requires detail::payload_node<Plan, I>
    constexpr auto emplace(Args&&... args)
        noexcept(noexcept(logical_slot<I>().emplace(std::forward<Args>(args)...)))
        -> output_type<I>& {
        auto& value = logical_slot<I>().emplace(std::forward<Args>(args)...);
        live_slots_[I] = true;
        return value;
    }

    /**
     * @brief Returns a direct-output sink for maybe-payload node `I`.
     */
    template <std::size_t I>
        requires detail::payload_node<Plan, I>
              && (slot_logical_policy<I> == detail::slot_logical_policy::maybe_payload)
    [[nodiscard]] constexpr auto out() & noexcept -> result_out<output_type<I>> {
        return result_out<output_type<I>> {logical_slot<I>()};
    }

    /**
     * @brief Returns node `I`'s stored payload.
     */
    template <std::size_t I>
        requires detail::payload_node<Plan, I>
    [[nodiscard]] constexpr auto get() & noexcept -> output_type<I>& {
        return logical_slot<I>().get();
    }

    /**
     * @brief Returns node `I`'s stored payload from a const slots object.
     */
    template <std::size_t I>
        requires detail::payload_node<Plan, I>
    [[nodiscard]] constexpr auto get() const& noexcept -> const output_type<I>& {
        return logical_slot<I>().get();
    }

    /**
     * @brief Destroys node `I`'s payload if the node has one.
     */
    template <std::size_t I>
    constexpr void destroy() noexcept {
        if constexpr (!std::is_void_v<output_type<I>>) {
            if constexpr (slot_logical_policy<I> == detail::slot_logical_policy::maybe_payload) {
                if (logical_slot<I>().has_value()) {
                    logical_slot<I>().destroy();
                }
            } else {
                if (live_slots_[I]) {
                    logical_slot<I>().destroy();
                }
            }

            live_slots_[I] = false;
        }
    }

    /**
     * @brief Returns the `from_prev` view produced by node `I`'s payload.
     */
    template <std::size_t I>
    [[nodiscard]] constexpr auto prev_view_for() & {
        if constexpr (std::is_void_v<output_type<I>>) {
            return no_prev {};
        } else {
            return prev_slot(get<I>());
        }
    }

    /**
     * @brief Returns the const `from_prev` view produced by node `I`'s payload.
     */
    template <std::size_t I>
    [[nodiscard]] constexpr auto prev_view_for() const& {
        if constexpr (std::is_void_v<output_type<I>>) {
            return no_prev {};
        } else {
            return prev_slot(get<I>());
        }
    }

private:
    tuple_type storage {};

    template <std::size_t I>
    using logical_slot_type =
        std::tuple_element_t<physical_slot_index<I>, tuple_type>;

    template <std::size_t I>
    [[nodiscard]] constexpr auto logical_slot() & noexcept -> logical_slot_type<I>& {
        return std::get<physical_slot_index<I>>(storage);
    }

    template <std::size_t I>
    [[nodiscard]] constexpr auto logical_slot() const& noexcept -> const logical_slot_type<I>& {
        return std::get<physical_slot_index<I>>(storage);
    }

    template <std::size_t I = 0>
    constexpr void destroy_active_slots() noexcept {
        if constexpr (I < Plan::node_count) {
            destroy<I>();
            destroy_active_slots<I + 1>();
        }
    }

    std::array<bool, Plan::node_count> live_slots_ {};
};

} // namespace detail

/**
 * @brief Public per-run plan payload storage entry point.
 *
 * The default layout policy is `slot_layout_one_to_one_policy`, but the public
 * template itself is only a thin wrapper. Concrete storage backends live in
 * `detail::plan_exec_slots_impl`, where one-to-one uses typed slots and compact
 * layouts use erased slots.
 */
template <typename Plan, typename LayoutPolicy = slot_layout_one_to_one_policy>
    requires detail::slot_layout_policy<LayoutPolicy>
struct plan_exec_slots : detail::plan_exec_slots_impl<Plan, LayoutPolicy> {};

template <typename Plan, typename LayoutPolicy = slot_layout_one_to_one_policy>
using plan_slots = plan_exec_slots<Plan, LayoutPolicy>;

} // namespace yorch
