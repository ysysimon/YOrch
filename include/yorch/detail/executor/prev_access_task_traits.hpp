#pragma once

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "../../bind/tasks.hpp"
#include "../../task_adapters/catch_as_failure.hpp"
#include "../../task_adapters/retry.hpp"
#include "prev_access_specs.hpp"

namespace yorch::detail {

struct prev_access_summary {
    std::size_t prev_access_count = 0;
    std::size_t exclusive_prev_access_count = 0;
    std::size_t borrow_prev_count = 0;
    std::size_t borrow_prev_mut_count = 0;
    std::size_t copy_prev_count = 0;
    std::size_t consume_prev_count = 0;
};

[[nodiscard]] constexpr prev_access_summary merge_prev_access_summary(
    prev_access_summary lhs,
    const prev_access_summary& rhs) noexcept {
    lhs.prev_access_count += rhs.prev_access_count;
    lhs.exclusive_prev_access_count += rhs.exclusive_prev_access_count;
    lhs.borrow_prev_count += rhs.borrow_prev_count;
    lhs.borrow_prev_mut_count += rhs.borrow_prev_mut_count;
    lhs.copy_prev_count += rhs.copy_prev_count;
    lhs.consume_prev_count += rhs.consume_prev_count;
    return lhs;
}

template <typename Spec>
[[nodiscard]] consteval prev_access_summary summarize_prev_access_spec() {
    prev_access_summary summary {};

    if constexpr (is_prev_access_spec_v<Spec>) {
        ++summary.prev_access_count;
    }

    if constexpr (is_exclusive_prev_access_spec_v<Spec>) {
        ++summary.exclusive_prev_access_count;
    }

    if constexpr (is_borrow_prev_spec_v<Spec>) {
        ++summary.borrow_prev_count;
    }

    if constexpr (is_borrow_prev_mut_spec_v<Spec>) {
        ++summary.borrow_prev_mut_count;
    }

    if constexpr (is_copy_prev_spec_v<Spec>) {
        ++summary.copy_prev_count;
    }

    if constexpr (is_consume_prev_spec_v<Spec>) {
        ++summary.consume_prev_count;
    }

    return summary;
}

[[nodiscard]] constexpr bool prev_access_summary_is_locally_valid(
    const prev_access_summary& summary) noexcept {
    return summary.exclusive_prev_access_count == 0 ||
           (summary.exclusive_prev_access_count == 1 &&
            summary.prev_access_count == 1);
}

template <typename Task>
struct task_binding_view;

template <typename F, typename... Specs>
struct task_binding_view<bound_task<F, Specs...>> {
    using callable_type = std::remove_cvref_t<F>;
    using receiver_spec = void;
    using specs_tuple = std::tuple<Specs...>;

    static constexpr bool has_receiver = false;

    template <std::size_t I>
    using arg = nth_arg_t<I, callable_type>;
};

template <typename F, typename T, typename... Specs>
struct task_binding_view<bound_output_task<F, T, Specs...>> {
    using callable_type = std::remove_cvref_t<F>;
    using receiver_spec = void;
    using specs_tuple = std::tuple<Specs...>;

    static constexpr bool has_receiver = false;

    template <std::size_t I>
    using arg = nth_arg_t<I, callable_type>;
};

template <typename F, typename T, typename... Specs>
struct task_binding_view<bound_forward_prev_task<F, T, Specs...>> {
    using callable_type = std::remove_cvref_t<F>;
    using receiver_spec = void;
    using specs_tuple = std::tuple<Specs...>;

    static constexpr bool has_receiver = false;

    template <std::size_t I>
    using arg = nth_arg_t<I, callable_type>;
};

template <typename F, typename ReceiverSpec, typename... Specs>
struct task_binding_view<bound_member_task<F, ReceiverSpec, Specs...>> {
    using callable_type = std::remove_cvref_t<F>;
    using receiver_spec = std::remove_cvref_t<ReceiverSpec>;
    using specs_tuple = std::tuple<Specs...>;

    static constexpr bool has_receiver = true;

    template <std::size_t I>
    using arg = member_nth_arg_t<I, callable_type>;
};

template <typename F, typename ReceiverSpec, typename T, typename... Specs>
struct task_binding_view<bound_member_output_task<F, ReceiverSpec, T, Specs...>> {
    using callable_type = std::remove_cvref_t<F>;
    using receiver_spec = std::remove_cvref_t<ReceiverSpec>;
    using specs_tuple = std::tuple<Specs...>;

    static constexpr bool has_receiver = true;

    template <std::size_t I>
    using arg = member_nth_arg_t<I, callable_type>;
};

template <typename View, std::size_t... I>
[[nodiscard]] consteval bool task_binding_prev_access_bindings_valid_impl(
    std::index_sequence<I...>) {
    using specs_tuple = typename View::specs_tuple;

    if constexpr (View::has_receiver) {
        return ((!is_prev_access_spec_v<typename View::receiver_spec>) ||
                member_receiver_prev_access_valid_v<
                    typename View::receiver_spec,
                    typename View::callable_type>) &&
               (((!is_prev_access_spec_v<std::tuple_element_t<I, specs_tuple>>) ||
                 prev_access_binding_valid_v<
                     std::tuple_element_t<I, specs_tuple>,
                     typename View::template arg<I>>) &&
                ...);
    } else {
        return (((!is_prev_access_spec_v<std::tuple_element_t<I, specs_tuple>>) ||
                 prev_access_binding_valid_v<
                     std::tuple_element_t<I, specs_tuple>,
                     typename View::template arg<I>>) &&
                ...);
    }
}

template <typename View, std::size_t... I>
[[nodiscard]] consteval prev_access_summary summarize_task_binding_impl(
    std::index_sequence<I...>) {
    using specs_tuple = typename View::specs_tuple;

    prev_access_summary summary {};

    if constexpr (View::has_receiver) {
        summary = merge_prev_access_summary(
            summary,
            summarize_prev_access_spec<typename View::receiver_spec>());
    }

    ((summary = merge_prev_access_summary(
          summary,
          summarize_prev_access_spec<std::tuple_element_t<I, specs_tuple>>())),
     ...);

    return summary;
}

template <typename Task>
struct task_prev_access_traits_impl {
    static constexpr prev_access_summary summary {};
    static constexpr bool valid = true;
};

template <typename Task>
struct task_prev_access_traits_from_binding_view {
    using view = task_binding_view<Task>;

    static constexpr prev_access_summary summary =
        summarize_task_binding_impl<view>(
            std::make_index_sequence<
                std::tuple_size_v<typename view::specs_tuple>> {});

    static constexpr bool valid =
        task_binding_prev_access_bindings_valid_impl<view>(
            std::make_index_sequence<
                std::tuple_size_v<typename view::specs_tuple>> {}) &&
        prev_access_summary_is_locally_valid(summary);
};

template <typename F, typename... Specs>
struct task_prev_access_traits_impl<bound_task<F, Specs...>>
    : task_prev_access_traits_from_binding_view<bound_task<F, Specs...>> {};

template <typename F, typename T, typename... Specs>
struct task_prev_access_traits_impl<bound_output_task<F, T, Specs...>>
    : task_prev_access_traits_from_binding_view<bound_output_task<F, T, Specs...>> {};

template <typename F, typename T, typename... Specs>
struct task_prev_access_traits_impl<bound_forward_prev_task<F, T, Specs...>>
    : task_prev_access_traits_from_binding_view<bound_forward_prev_task<F, T, Specs...>> {};

template <typename F, typename ReceiverSpec, typename... Specs>
struct task_prev_access_traits_impl<bound_member_task<F, ReceiverSpec, Specs...>>
    : task_prev_access_traits_from_binding_view<
          bound_member_task<F, ReceiverSpec, Specs...>> {};

template <typename F, typename ReceiverSpec, typename T, typename... Specs>
struct task_prev_access_traits_impl<bound_member_output_task<F, ReceiverSpec, T, Specs...>>
    : task_prev_access_traits_from_binding_view<
          bound_member_output_task<F, ReceiverSpec, T, Specs...>> {};

template <typename Task>
struct passthrough_task_prev_access_traits {
    static constexpr prev_access_summary summary =
        task_prev_access_traits_impl<Task>::summary;
    static constexpr bool valid =
        task_prev_access_traits_impl<Task>::valid;
};

template <typename Task>
struct task_prev_access_traits_impl<catch_failure_task<Task>>
    : passthrough_task_prev_access_traits<Task> {};

template <typename Task, typename Policy>
struct task_prev_access_traits_impl<catch_failure_with_policy_task<Task, Policy>>
    : passthrough_task_prev_access_traits<Task> {};

template <typename Task, typename Policy>
struct task_prev_access_traits_impl<retry_task<Task, Policy>> {
    static constexpr prev_access_summary summary =
        task_prev_access_traits_impl<Task>::summary;
    static constexpr bool valid =
        task_prev_access_traits_impl<Task>::valid &&
        (summary.consume_prev_count == 0);
};

template <typename Task>
struct task_prev_access_valid
    : std::bool_constant<
          task_prev_access_traits_impl<std::remove_cvref_t<Task>>::valid> {};

template <typename Task>
struct task_uses_prev_access
    : std::bool_constant<
          task_prev_access_traits_impl<std::remove_cvref_t<Task>>::summary.prev_access_count != 0> {};

template <typename Task>
struct task_uses_exclusive_prev_access
    : std::bool_constant<
          task_prev_access_traits_impl<std::remove_cvref_t<Task>>::summary
              .exclusive_prev_access_count != 0> {};

template <typename Task>
struct task_uses_consume_prev
    : std::bool_constant<
          task_prev_access_traits_impl<std::remove_cvref_t<Task>>::summary.consume_prev_count != 0> {};

template <typename Task>
struct task_uses_borrow_prev
    : std::bool_constant<
          task_prev_access_traits_impl<std::remove_cvref_t<Task>>::summary.borrow_prev_count != 0> {};

template <typename Task>
struct task_uses_borrow_prev_mut
    : std::bool_constant<
          task_prev_access_traits_impl<std::remove_cvref_t<Task>>::summary
              .borrow_prev_mut_count != 0> {};

template <typename Task>
struct task_uses_copy_prev
    : std::bool_constant<
          task_prev_access_traits_impl<std::remove_cvref_t<Task>>::summary.copy_prev_count != 0> {};

template <typename Task>
inline constexpr bool task_prev_access_valid_v =
    task_prev_access_valid<Task>::value;

template <typename Task>
inline constexpr bool task_uses_prev_access_v =
    task_uses_prev_access<Task>::value;

template <typename Task>
inline constexpr bool task_uses_exclusive_prev_access_v =
    task_uses_exclusive_prev_access<Task>::value;

template <typename Task>
inline constexpr bool task_uses_consume_prev_v =
    task_uses_consume_prev<Task>::value;

template <typename Task>
inline constexpr bool task_uses_borrow_prev_v =
    task_uses_borrow_prev<Task>::value;

template <typename Task>
inline constexpr bool task_uses_borrow_prev_mut_v =
    task_uses_borrow_prev_mut<Task>::value;

template <typename Task>
inline constexpr bool task_uses_copy_prev_v =
    task_uses_copy_prev<Task>::value;

} // namespace yorch::detail
