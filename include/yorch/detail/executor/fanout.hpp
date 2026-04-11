#pragma once

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "../../context.hpp"
#include "../../detail/maybe_storage.hpp"
#include "../../task_tree/policies.hpp"
#include "prev_validation.hpp"

namespace yorch::detail {

template <typename Task>
inline constexpr bool task_uses_copy_only_prev_access_v =
    task_uses_copy_prev_v<Task> &&
    !task_uses_borrow_prev_v<Task> &&
    !task_uses_borrow_prev_mut_v<Task> &&
    !task_uses_consume_prev_v<Task>;

template <typename Task>
inline constexpr bool task_uses_consume_only_prev_access_v =
    task_uses_consume_prev_v<Task>;

template <typename Plan, std::size_t I>
concept fanout_validatable_plan_node =
    requires {
        { Plan::no_parent } -> std::convertible_to<std::size_t>;
        { Plan::template parent_index<I> } -> std::convertible_to<std::size_t>;
        { Plan::template child_count<I> } -> std::convertible_to<std::size_t>;
        typename Plan::template task_type<I>;
        typename Plan::template output_type<I>;
        typename Plan::template fanout_policy_type<I>;
    };

template <typename Plan, std::size_t Child>
    requires fanout_validatable_plan_node<Plan, Child>
inline constexpr bool node_uses_copy_only_prev_access_v =
    task_uses_copy_only_prev_access_v<typename Plan::template task_type<Child>>;

template <typename Plan, std::size_t Child>
    requires fanout_validatable_plan_node<Plan, Child>
inline constexpr bool node_uses_consume_only_prev_access_v =
    task_uses_consume_only_prev_access_v<typename Plan::template task_type<Child>>;

template <typename Plan, std::size_t Parent, std::size_t... Ord>
[[nodiscard]] consteval std::size_t parent_exclusive_prev_child_count_impl(std::index_sequence<Ord...>) {
    return (std::size_t {0} + ... +
            (task_uses_exclusive_prev_access_v<
                 typename Plan::template task_type<Plan::template child_index<Parent, Ord>>>
                 ? std::size_t {1}
                 : std::size_t {0}));
}

template <typename Plan, std::size_t Parent, std::size_t... Ord>
[[nodiscard]] consteval std::size_t parent_copy_only_prev_child_count_impl(std::index_sequence<Ord...>) {
    return (std::size_t {0} + ... +
            (node_uses_copy_only_prev_access_v<
                 Plan,
                 Plan::template child_index<Parent, Ord>>
                 ? std::size_t {1}
                 : std::size_t {0}));
}

template <typename Plan, std::size_t Parent, std::size_t... Ord>
[[nodiscard]] consteval std::size_t parent_consume_only_prev_child_count_impl(
    std::index_sequence<Ord...>) {
    return (std::size_t {0} + ... +
            (node_uses_consume_only_prev_access_v<
                 Plan,
                 Plan::template child_index<Parent, Ord>>
                 ? std::size_t {1}
                 : std::size_t {0}));
}

template <typename Plan, std::size_t Parent, std::size_t... Ord>
[[nodiscard]] consteval bool parent_has_non_copy_or_consume_prev_child_impl(std::index_sequence<Ord...>) {
    return ((task_uses_prev_access_v<
                 typename Plan::template task_type<Plan::template child_index<Parent, Ord>>> &&
             !node_uses_copy_only_prev_access_v<
                 Plan,
                 Plan::template child_index<Parent, Ord>> &&
             !node_uses_consume_only_prev_access_v<
                 Plan,
                 Plan::template child_index<Parent, Ord>>) || ...);
}

template <typename Plan, std::size_t Parent>
    requires fanout_validatable_plan_node<Plan, Parent>
[[nodiscard]] consteval bool parent_fanout_policy_valid() {
    using policy_t = typename Plan::template fanout_policy_type<Parent>;

    if constexpr (Plan::template child_count<Parent> == 0) {
        return true;
    } else if constexpr (std::is_same_v<policy_t, yorch::fanout_auto_policy>) {
        if constexpr (Plan::template child_count<Parent> <= 1) {
            return true;
        } else {
            return parent_exclusive_prev_child_count_impl<Plan, Parent>(
                       std::make_index_sequence<Plan::template child_count<Parent>> {}) == 0;
        }
    } else if constexpr (std::is_same_v<policy_t, yorch::fanout_shared_readonly_policy>) {
        return parent_exclusive_prev_child_count_impl<Plan, Parent>(
                   std::make_index_sequence<Plan::template child_count<Parent>> {}) == 0;
    } else {
        static_assert(std::is_same_v<policy_t, yorch::fanout_consume_with_copies_policy>,
                      "Unsupported fanout policy");

        return !parent_has_non_copy_or_consume_prev_child_impl<Plan, Parent>(
                   std::make_index_sequence<Plan::template child_count<Parent>> {}) &&
               parent_consume_only_prev_child_count_impl<Plan, Parent>(
                   std::make_index_sequence<Plan::template child_count<Parent>> {}) <= 1;
    }
}

template <typename Plan, std::size_t... I>
    requires (fanout_validatable_plan_node<Plan, I> && ...)
[[nodiscard]] consteval bool plan_fanout_policy_valid_impl(std::index_sequence<I...>) {
    return (parent_fanout_policy_valid<Plan, I>() && ...);
}

template <typename Plan>
inline constexpr bool plan_fanout_policy_valid_v =
    plan_fanout_policy_valid_impl<Plan>(std::make_index_sequence<Plan::node_count> {});

template <typename Plan, std::size_t Parent>
    requires fanout_validatable_plan_node<Plan, Parent>
inline constexpr bool parent_requires_fanout_staging_v =
    std::is_same_v<typename Plan::template fanout_policy_type<Parent>,
                   yorch::fanout_consume_with_copies_policy> &&
    (parent_consume_only_prev_child_count_impl<Plan, Parent>(
         std::make_index_sequence<Plan::template child_count<Parent>> {}) == 1) &&
    (parent_copy_only_prev_child_count_impl<Plan, Parent>(
         std::make_index_sequence<Plan::template child_count<Parent>> {}) != 0);

template <typename Plan, std::size_t Parent>
    requires fanout_validatable_plan_node<Plan, Parent>
inline constexpr bool parent_has_staged_copy_children_v =
    parent_requires_fanout_staging_v<Plan, Parent>;

template <typename Plan, std::size_t Child>
    requires fanout_validatable_plan_node<Plan, Child>
inline constexpr bool node_uses_staged_copy_prev_v = [] {
    if constexpr (Plan::template parent_index<Child> == Plan::no_parent) {
        return false;
    } else {
        return parent_requires_fanout_staging_v<Plan, Plan::template parent_index<Child>> &&
               node_uses_copy_only_prev_access_v<Plan, Child>;
    }
}();

struct empty_fanout_stage {
    constexpr void destroy() noexcept {}

    [[nodiscard]] static constexpr bool has_value() noexcept {
        return false;
    }
};

template <typename Plan, std::size_t Parent, typename = void>
struct fanout_stage_slot {
    using type = empty_fanout_stage;
};

template <typename Plan, std::size_t Parent>
struct fanout_stage_slot<
    Plan,
    Parent,
    std::enable_if_t<parent_has_staged_copy_children_v<Plan, Parent>>> {
    using type = maybe_storage<typename Plan::template output_type<Parent>>;
};

template <typename Plan, std::size_t... I>
[[nodiscard]] consteval auto make_plan_fanout_stage_tuple(std::index_sequence<I...>)
    -> std::tuple<typename fanout_stage_slot<Plan, I>::type...>;

template <typename Plan>
using plan_fanout_stage_tuple_t =
    decltype(make_plan_fanout_stage_tuple<Plan>(std::make_index_sequence<Plan::node_count> {}));

template <typename Plan>
struct plan_fanout_state {
    template <std::size_t Parent>
    constexpr void prepare_fanout_staging(auto& slots) {
        if constexpr (parent_requires_fanout_staging_v<Plan, Parent>) {
            auto& source = slots.template get<Parent>();
            using parent_output_t = typename Plan::template output_type<Parent>;

            static_assert(
                std::is_constructible_v<parent_output_t, const parent_output_t&>,
                "fanout_consume_with_copies_policy requires the parent output type "
                "to be constructible from const T& for copy_prev children");

            stage_slot<Parent>().emplace(source);
        }
    }

    template <std::size_t Parent>
    constexpr void destroy_fanout_staging() noexcept {
        if constexpr (parent_requires_fanout_staging_v<Plan, Parent>) {
            stage_slot<Parent>().destroy();
        }
    }

    template <std::size_t Parent>
    [[nodiscard]] constexpr auto staged_prev_view_for_parent() & {
        static_assert(parent_has_staged_copy_children_v<Plan, Parent>,
                      "staged_prev_view_for_parent<I>() requires a parent with staged copy children");
        return prev_slot(stage_slot<Parent>().get());
    }

    template <std::size_t Parent>
    [[nodiscard]] constexpr auto staged_prev_view_for_parent() const& {
        static_assert(parent_has_staged_copy_children_v<Plan, Parent>,
                      "staged_prev_view_for_parent<I>() requires a parent with staged copy children");
        return prev_slot(stage_slot<Parent>().get());
    }

private:
    template <std::size_t Parent>
    [[nodiscard]] constexpr auto& stage_slot() & noexcept {
        return std::get<Parent>(staging_);
    }

    template <std::size_t Parent>
    [[nodiscard]] constexpr const auto& stage_slot() const& noexcept {
        return std::get<Parent>(staging_);
    }

    plan_fanout_stage_tuple_t<Plan> staging_ {};
};

} // namespace yorch::detail
