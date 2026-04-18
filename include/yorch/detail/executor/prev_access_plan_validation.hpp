#pragma once

#include <concepts> // IWYU pragma: keep
#include <cstddef>
#include <type_traits>
#include <utility>

#include "prev_access_task_traits.hpp"

namespace yorch::detail {

/**
 * @brief Minimal static-plan protocol required by the prev-access validation
 * path for a specific node index.
 */
template <typename Plan, std::size_t I>
concept prev_access_validatable_plan_node =
    requires {
        { Plan::no_parent } -> std::convertible_to<std::size_t>;
        { Plan::template parent_index<I> } -> std::convertible_to<std::size_t>;
        { Plan::template child_count<I> } -> std::convertible_to<std::size_t>;
        typename Plan::template task_type<I>;
    };

/**
 * @brief Checks whether node `I` is structurally allowed to use direct-parent
 * access at all.
 */
template <typename Plan, std::size_t I>
    requires prev_access_validatable_plan_node<Plan, I>
[[nodiscard]] consteval bool node_prev_source_valid() {
    using task_t = typename Plan::template task_type<I>;

    if constexpr (Plan::template parent_index<I> == Plan::no_parent) {
        return !task_uses_prev_access_v<task_t>;
    } else {
        constexpr auto parent = Plan::template parent_index<I>;

        if constexpr (std::is_void_v<typename Plan::template output_type<parent>>) {
            return !task_uses_prev_access_v<task_t>;
        } else {
            return true;
        }
    }
}

/**
 * @brief Checks whether node `I`'s declared prev-access mode is locally valid.
 *
 * Fan-out compatibility is validated separately by the fan-out policy layer.
 */
template <typename Plan, std::size_t I>
    requires prev_access_validatable_plan_node<Plan, I>
[[nodiscard]] consteval bool node_prev_access_valid() {
    using task_t = typename Plan::template task_type<I>;

    return task_prev_access_valid_v<task_t>;
}

template <typename Plan, std::size_t... I>
    requires (prev_access_validatable_plan_node<Plan, I> && ...)
[[nodiscard]] consteval bool plan_prev_source_valid_impl(std::index_sequence<I...>) {
    return (node_prev_source_valid<Plan, I>() && ...);
}

template <typename Plan, std::size_t... I>
    requires (prev_access_validatable_plan_node<Plan, I> && ...)
[[nodiscard]] consteval bool plan_prev_access_valid_impl(std::index_sequence<I...>) {
    return (node_prev_access_valid<Plan, I>() && ...);
}

template <typename Plan>
inline constexpr bool plan_prev_source_valid_v =
    plan_prev_source_valid_impl<Plan>(std::make_index_sequence<Plan::node_count> {});

template <typename Plan>
inline constexpr bool plan_prev_access_valid_v =
    plan_prev_access_valid_impl<Plan>(std::make_index_sequence<Plan::node_count> {});

} // namespace yorch::detail
