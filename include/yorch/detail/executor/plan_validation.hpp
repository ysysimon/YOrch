#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

#include "fanout.hpp"
#include "prev_access_plan_validation.hpp"

namespace yorch::detail {

template <typename>
inline constexpr bool plan_validation_always_false_v = false;

enum class plan_validation_error {
    ok,
    prev_access_root_node,
    prev_access_void_parent,
    forward_prev_root_node,
    forward_prev_void_parent,
    forward_prev_parent_type_mismatch,
    prev_access_mode_invalid,
    fanout_policy_invalid,
};

template <typename Plan, std::size_t I>
concept plan_validatable_node =
    prev_access_validatable_plan_node<Plan, I> &&
    fanout_validatable_plan_node<Plan, I>;

template <typename Plan, std::size_t I>
    requires plan_validatable_node<Plan, I>
[[nodiscard]] consteval bool node_uses_root_prev_access() {
    using task_t = typename Plan::template task_type<I>;

    return !task_uses_forward_prev_output_protocol_v<task_t> &&
           task_uses_prev_access_v<task_t> &&
           Plan::template parent_index<I> == Plan::no_parent;
}

template <typename Plan, std::size_t I>
    requires plan_validatable_node<Plan, I>
[[nodiscard]] consteval bool node_uses_void_parent_prev_access() {
    using task_t = typename Plan::template task_type<I>;

    if constexpr (Plan::template parent_index<I> == Plan::no_parent) {
        return false;
    } else {
        constexpr auto parent = Plan::template parent_index<I>;

        return !task_uses_forward_prev_output_protocol_v<task_t> &&
               task_uses_prev_access_v<task_t> &&
               std::is_void_v<typename Plan::template output_type<parent>>;
    }
}

template <typename Plan, std::size_t I>
    requires plan_validatable_node<Plan, I>
[[nodiscard]] consteval bool node_uses_root_forward_prev() {
    using task_t = typename Plan::template task_type<I>;

    return task_uses_forward_prev_output_protocol_v<task_t> &&
           Plan::template parent_index<I> == Plan::no_parent;
}

template <typename Plan, std::size_t I>
    requires plan_validatable_node<Plan, I>
[[nodiscard]] consteval bool node_uses_void_parent_forward_prev() {
    using task_t = typename Plan::template task_type<I>;

    if constexpr (!task_uses_forward_prev_output_protocol_v<task_t> ||
                  Plan::template parent_index<I> == Plan::no_parent) {
        return false;
    } else {
        constexpr auto parent = Plan::template parent_index<I>;
        return std::is_void_v<typename Plan::template output_type<parent>>;
    }
}

template <typename Plan, std::size_t I>
    requires plan_validatable_node<Plan, I>
[[nodiscard]] consteval bool node_uses_mismatched_forward_prev_parent_type() {
    using task_t = typename Plan::template task_type<I>;

    if constexpr (!task_uses_forward_prev_output_protocol_v<task_t> ||
                  Plan::template parent_index<I> == Plan::no_parent) {
        return false;
    } else {
        constexpr auto parent = Plan::template parent_index<I>;

        if constexpr (std::is_void_v<typename Plan::template output_type<parent>>) {
            return false;
        } else {
            return !std::is_same_v<
                typename Plan::template output_type<I>,
                typename Plan::template output_type<parent>>;
        }
    }
}

template <typename Plan, std::size_t... I>
    requires (plan_validatable_node<Plan, I> && ...)
[[nodiscard]] consteval plan_validation_error validate_plan_impl(std::index_sequence<I...>) {
    if constexpr ((node_uses_root_prev_access<Plan, I>() || ...)) {
        return plan_validation_error::prev_access_root_node;
    } else if constexpr ((node_uses_void_parent_prev_access<Plan, I>() || ...)) {
        return plan_validation_error::prev_access_void_parent;
    } else if constexpr ((node_uses_root_forward_prev<Plan, I>() || ...)) {
        return plan_validation_error::forward_prev_root_node;
    } else if constexpr ((node_uses_void_parent_forward_prev<Plan, I>() || ...)) {
        return plan_validation_error::forward_prev_void_parent;
    } else if constexpr ((node_uses_mismatched_forward_prev_parent_type<Plan, I>() || ...)) {
        return plan_validation_error::forward_prev_parent_type_mismatch;
    } else if constexpr (!plan_prev_access_valid_v<Plan>) {
        return plan_validation_error::prev_access_mode_invalid;
    } else if constexpr (!plan_fanout_policy_valid_v<Plan>) {
        return plan_validation_error::fanout_policy_invalid;
    } else {
        return plan_validation_error::ok;
    }
}

template <typename Plan>
inline constexpr plan_validation_error plan_validation_error_v =
    validate_plan_impl<Plan>(std::make_index_sequence<Plan::node_count> {});

template <typename Plan>
[[nodiscard]] consteval plan_validation_error validate_plan() {
    return plan_validation_error_v<Plan>;
}

template <typename Plan>
consteval void emit_plan_diagnostic() {
    constexpr auto error = validate_plan<Plan>();

    if constexpr (error == plan_validation_error::ok) {
        return;
    } else if constexpr (error == plan_validation_error::prev_access_root_node) {
        static_assert(plan_validation_always_false_v<Plan>,
                      "compile_plan(...) does not allow prev-access nodes at the root; prev-access requires a direct parent carrying a payload");
    } else if constexpr (error == plan_validation_error::prev_access_void_parent) {
        static_assert(plan_validation_always_false_v<Plan>,
                      "compile_plan(...) requires prev-access nodes to have a direct parent carrying a non-void payload");
    } else if constexpr (error == plan_validation_error::forward_prev_root_node) {
        static_assert(plan_validation_always_false_v<Plan>,
                      "compile_plan(...) does not allow forward-prev nodes at the root");
    } else if constexpr (error == plan_validation_error::forward_prev_void_parent) {
        static_assert(plan_validation_always_false_v<Plan>,
                      "compile_plan(...) requires forward-prev nodes to have a direct parent carrying a non-void payload");
    } else if constexpr (error == plan_validation_error::forward_prev_parent_type_mismatch) {
        static_assert(plan_validation_always_false_v<Plan>,
                      "compile_plan(...) requires forward-prev nodes to match the direct parent logical output type exactly");
    } else if constexpr (error == plan_validation_error::prev_access_mode_invalid) {
        static_assert(plan_validation_always_false_v<Plan>,
                      "compile_plan(...) rejected the task tree because a task uses an invalid prev-access combination");
    } else {
        static_assert(plan_validation_always_false_v<Plan>,
                      "compile_plan(...) rejected the task tree because the selected fanout policy is incompatible with child prev-access modes");
    }
}

template <typename Plan>
inline constexpr bool plan_valid_v =
    plan_validation_error_v<Plan> == plan_validation_error::ok;

} // namespace yorch::detail
