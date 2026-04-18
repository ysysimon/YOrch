#pragma once

#include <concepts> // IWYU pragma: keep
#include <cstddef>
#include <tuple>
#include <type_traits>

#include "../../bind.hpp"
#include "../../specs.hpp"
#include "../../task_adapters/catch_as_failure.hpp"
#include "../../task_adapters/retry.hpp"

namespace yorch::detail {

template <typename Spec>
struct is_borrow_prev_spec : std::false_type {};

template <typename T>
struct is_borrow_prev_spec<borrow_prev_t<T>> : std::true_type {};

template <typename Spec>
inline constexpr bool is_borrow_prev_spec_v =
    is_borrow_prev_spec<std::remove_cvref_t<Spec>>::value;

template <typename Spec>
struct is_borrow_prev_mut_spec : std::false_type {};

template <typename T>
struct is_borrow_prev_mut_spec<borrow_prev_mut_t<T>> : std::true_type {};

template <typename Spec>
inline constexpr bool is_borrow_prev_mut_spec_v =
    is_borrow_prev_mut_spec<std::remove_cvref_t<Spec>>::value;

template <typename Spec>
struct is_copy_prev_spec : std::false_type {};

template <typename T>
struct is_copy_prev_spec<copy_prev_t<T>> : std::true_type {};

template <typename Spec>
inline constexpr bool is_copy_prev_spec_v =
    is_copy_prev_spec<std::remove_cvref_t<Spec>>::value;

template <typename Spec>
struct is_consume_prev_spec : std::false_type {};

template <typename T>
struct is_consume_prev_spec<consume_prev_t<T>> : std::true_type {};

template <typename Spec>
inline constexpr bool is_consume_prev_spec_v =
    is_consume_prev_spec<std::remove_cvref_t<Spec>>::value;

template <typename Spec>
inline constexpr bool is_prev_access_spec_v =
    is_borrow_prev_spec_v<Spec> ||
    is_borrow_prev_mut_spec_v<Spec> ||
    is_copy_prev_spec_v<Spec> ||
    is_consume_prev_spec_v<Spec>;

template <typename Spec>
inline constexpr bool is_exclusive_prev_access_spec_v =
    is_borrow_prev_mut_spec_v<Spec> ||
    is_consume_prev_spec_v<Spec>;

template <typename Spec, typename Arg>
struct prev_access_binding_valid : std::false_type {};

template <typename T, typename Arg>
struct prev_access_binding_valid<borrow_prev_t<T>, Arg>
    : std::bool_constant<std::is_same_v<Arg, const typename borrow_prev_t<T>::type&>> {};

template <typename T, typename Arg>
struct prev_access_binding_valid<borrow_prev_mut_t<T>, Arg>
    : std::bool_constant<std::is_same_v<Arg, typename borrow_prev_mut_t<T>::type&>> {};

template <typename T, typename Arg>
struct prev_access_binding_valid<copy_prev_t<T>, Arg>
    : std::bool_constant<std::is_same_v<Arg, typename copy_prev_t<T>::type>> {};

template <typename T, typename Arg>
struct prev_access_binding_valid<consume_prev_t<T>, Arg>
    : std::bool_constant<
          std::is_same_v<Arg, typename consume_prev_t<T>::type> ||
          std::is_same_v<Arg, typename consume_prev_t<T>::type&&>> {};

template <typename Spec, typename Arg>
inline constexpr bool prev_access_binding_valid_v =
    prev_access_binding_valid<std::remove_cvref_t<Spec>, Arg>::value;

template <typename F>
inline constexpr bool member_receiver_is_const_v =
    std::is_const_v<std::remove_reference_t<member_receiver_arg_t<F>>>;

template <typename Spec, typename F>
struct member_receiver_prev_access_valid : std::false_type {};

template <typename T, typename F>
struct member_receiver_prev_access_valid<borrow_prev_t<T>, F>
    : std::bool_constant<
          std::is_same_v<typename borrow_prev_t<T>::type, member_class_t<F>> &&
          member_receiver_is_const_v<F>> {};

template <typename T, typename F>
struct member_receiver_prev_access_valid<borrow_prev_mut_t<T>, F>
    : std::bool_constant<std::is_same_v<typename borrow_prev_mut_t<T>::type, member_class_t<F>>> {};

template <typename T, typename F>
struct member_receiver_prev_access_valid<copy_prev_t<T>, F>
    : std::bool_constant<std::is_same_v<typename copy_prev_t<T>::type, member_class_t<F>>> {};

template <typename T, typename F>
struct member_receiver_prev_access_valid<consume_prev_t<T>, F>
    : std::bool_constant<std::is_same_v<typename consume_prev_t<T>::type, member_class_t<F>>> {};

template <typename Spec, typename F>
inline constexpr bool member_receiver_prev_access_valid_v =
    member_receiver_prev_access_valid<std::remove_cvref_t<Spec>, std::remove_cvref_t<F>>::value;

template <typename F, typename SpecsTuple, std::size_t... I>
[[nodiscard]] consteval bool bound_task_prev_access_bindings_valid_impl(std::index_sequence<I...>) {
    return (((!is_prev_access_spec_v<std::tuple_element_t<I, SpecsTuple>>) ||
             prev_access_binding_valid_v<
                 std::tuple_element_t<I, SpecsTuple>,
                 nth_arg_t<I, F>>) && ...);
}

template <typename F, typename ReceiverSpec, typename SpecsTuple, std::size_t... I>
[[nodiscard]] consteval bool bound_member_task_prev_access_bindings_valid_impl(
    std::index_sequence<I...>) {
    return ((!is_prev_access_spec_v<ReceiverSpec>) ||
            member_receiver_prev_access_valid_v<ReceiverSpec, F>) &&
           (((!is_prev_access_spec_v<std::tuple_element_t<I, SpecsTuple>>) ||
             prev_access_binding_valid_v<
                 std::tuple_element_t<I, SpecsTuple>,
                 member_nth_arg_t<I, F>>) && ...);
}

template <typename SpecsTuple, std::size_t... I>
[[nodiscard]] consteval std::size_t bound_task_prev_access_count_impl(std::index_sequence<I...>) {
    return (std::size_t {0} + ... +
            (is_prev_access_spec_v<std::tuple_element_t<I, SpecsTuple>> ? std::size_t {1}
                                                                        : std::size_t {0}));
}

template <typename ReceiverSpec, typename SpecsTuple, std::size_t... I>
[[nodiscard]] consteval std::size_t bound_member_task_prev_access_count_impl(
    std::index_sequence<I...>) {
    return (is_prev_access_spec_v<ReceiverSpec> ? std::size_t {1} : std::size_t {0}) +
           (std::size_t {0} + ... +
            (is_prev_access_spec_v<std::tuple_element_t<I, SpecsTuple>> ? std::size_t {1}
                                                                        : std::size_t {0}));
}

template <typename SpecsTuple, std::size_t... I>
[[nodiscard]] consteval std::size_t bound_task_consume_prev_count_impl(std::index_sequence<I...>) {
    return (std::size_t {0} + ... +
            (is_consume_prev_spec_v<std::tuple_element_t<I, SpecsTuple>> ? std::size_t {1}
                                                                         : std::size_t {0}));
}

template <typename ReceiverSpec, typename SpecsTuple, std::size_t... I>
[[nodiscard]] consteval std::size_t bound_member_task_consume_prev_count_impl(
    std::index_sequence<I...>) {
    return (is_consume_prev_spec_v<ReceiverSpec> ? std::size_t {1} : std::size_t {0}) +
           (std::size_t {0} + ... +
            (is_consume_prev_spec_v<std::tuple_element_t<I, SpecsTuple>> ? std::size_t {1}
                                                                         : std::size_t {0}));
}

template <typename SpecsTuple, std::size_t... I>
[[nodiscard]] consteval std::size_t bound_task_exclusive_prev_access_count_impl(
    std::index_sequence<I...>) {
    return (std::size_t {0} + ... +
            (is_exclusive_prev_access_spec_v<std::tuple_element_t<I, SpecsTuple>>
                 ? std::size_t {1}
                 : std::size_t {0}));
}

template <typename ReceiverSpec, typename SpecsTuple, std::size_t... I>
[[nodiscard]] consteval std::size_t bound_member_task_exclusive_prev_access_count_impl(
    std::index_sequence<I...>) {
    return (is_exclusive_prev_access_spec_v<ReceiverSpec> ? std::size_t {1} : std::size_t {0}) +
           (std::size_t {0} + ... +
            (is_exclusive_prev_access_spec_v<std::tuple_element_t<I, SpecsTuple>>
                 ? std::size_t {1}
                 : std::size_t {0}));
}

template <typename SpecsTuple, std::size_t... I>
[[nodiscard]] consteval std::size_t bound_task_borrow_prev_count_impl(std::index_sequence<I...>) {
    return (std::size_t {0} + ... +
            (is_borrow_prev_spec_v<std::tuple_element_t<I, SpecsTuple>> ? std::size_t {1}
                                                                        : std::size_t {0}));
}

template <typename ReceiverSpec, typename SpecsTuple, std::size_t... I>
[[nodiscard]] consteval std::size_t bound_member_task_borrow_prev_count_impl(
    std::index_sequence<I...>) {
    return (is_borrow_prev_spec_v<ReceiverSpec> ? std::size_t {1} : std::size_t {0}) +
           (std::size_t {0} + ... +
            (is_borrow_prev_spec_v<std::tuple_element_t<I, SpecsTuple>> ? std::size_t {1}
                                                                        : std::size_t {0}));
}

template <typename SpecsTuple, std::size_t... I>
[[nodiscard]] consteval std::size_t bound_task_borrow_prev_mut_count_impl(
    std::index_sequence<I...>) {
    return (std::size_t {0} + ... +
            (is_borrow_prev_mut_spec_v<std::tuple_element_t<I, SpecsTuple>> ? std::size_t {1}
                                                                            : std::size_t {0}));
}

template <typename ReceiverSpec, typename SpecsTuple, std::size_t... I>
[[nodiscard]] consteval std::size_t bound_member_task_borrow_prev_mut_count_impl(
    std::index_sequence<I...>) {
    return (is_borrow_prev_mut_spec_v<ReceiverSpec> ? std::size_t {1} : std::size_t {0}) +
           (std::size_t {0} + ... +
            (is_borrow_prev_mut_spec_v<std::tuple_element_t<I, SpecsTuple>> ? std::size_t {1}
                                                                            : std::size_t {0}));
}

template <typename SpecsTuple, std::size_t... I>
[[nodiscard]] consteval std::size_t bound_task_copy_prev_count_impl(std::index_sequence<I...>) {
    return (std::size_t {0} + ... +
            (is_copy_prev_spec_v<std::tuple_element_t<I, SpecsTuple>> ? std::size_t {1}
                                                                      : std::size_t {0}));
}

template <typename ReceiverSpec, typename SpecsTuple, std::size_t... I>
[[nodiscard]] consteval std::size_t bound_member_task_copy_prev_count_impl(
    std::index_sequence<I...>) {
    return (is_copy_prev_spec_v<ReceiverSpec> ? std::size_t {1} : std::size_t {0}) +
           (std::size_t {0} + ... +
            (is_copy_prev_spec_v<std::tuple_element_t<I, SpecsTuple>> ? std::size_t {1}
                                                                      : std::size_t {0}));
}

template <typename SpecsTuple, std::size_t... I>
[[nodiscard]] consteval bool bound_task_uses_exclusive_prev_access_impl(std::index_sequence<I...>) {
    return (is_exclusive_prev_access_spec_v<std::tuple_element_t<I, SpecsTuple>> || ...);
}

template <typename ReceiverSpec, typename SpecsTuple, std::size_t... I>
[[nodiscard]] consteval bool bound_member_task_uses_exclusive_prev_access_impl(
    std::index_sequence<I...>) {
    return is_exclusive_prev_access_spec_v<ReceiverSpec> ||
           (is_exclusive_prev_access_spec_v<std::tuple_element_t<I, SpecsTuple>> || ...);
}

template <typename Task>
struct task_prev_access_valid : std::true_type {};

template <typename Task>
struct task_uses_prev_access : std::false_type {};

template <typename Task>
struct task_uses_exclusive_prev_access : std::false_type {};

template <typename Task>
struct task_uses_consume_prev : std::false_type {};

template <typename Task>
struct task_uses_borrow_prev : std::false_type {};

template <typename Task>
struct task_uses_borrow_prev_mut : std::false_type {};

template <typename Task>
struct task_uses_copy_prev : std::false_type {};

/**
 * @brief Validates the declared prev-access usage of a bound task.
 *
 * There are two layers of checks here:
 *
 * - every prev-access spec must match the callable parameter type exactly
 *   according to its declared mode:
 *   - `borrow_prev<T>()` -> `const T&`
 *   - `borrow_prev_mut<T>()` -> `T&`
 *   - `copy_prev<T>()` -> `T`
 *   - `consume_prev<T>()` -> `T` or `T&&`
 * - `borrow_prev(...)` and `copy_prev(...)` may appear multiple times because
 *   they are shared access modes over the same direct parent source
 * - any exclusive access (`borrow_prev_mut(...)` or `consume_prev(...)`) must
 *   be the task's only prev-access declaration
 *
 * That exclusivity rule is what the local `prev_access_count /
 * exclusive_prev_access_count` calculation enforces below:
 *
 * - `exclusive_prev_access_count == 0` means the task uses only shared access
 *   (`borrow_prev(...)` and/or `copy_prev(...)`), which may repeat or mix
 * - `exclusive_prev_access_count == 1 && prev_access_count == 1` means there
 *   is exactly one exclusive access spec, and it is also the task's only
 *   prev-access spec
 *
 * As a result, the validation rejects:
 *
 * - multiple `borrow_prev_mut(...)` specs in the same task
 * - multiple `consume_prev(...)` specs in the same task
 * - mixing `borrow_prev_mut(...)` with `borrow_prev(...)`
 * - mixing `borrow_prev_mut(...)` with `copy_prev(...)`
 * - mixing `borrow_prev_mut(...)` with `consume_prev(...)`
 * - mixing `consume_prev(...)` with `borrow_prev(...)`
 * - mixing `consume_prev(...)` with `copy_prev(...)`
 *
 * This keeps exclusive access unambiguous and avoids depending on function
 * argument evaluation order when one parameter would request unique mutable or
 * consuming access while another parameter still tries to access the same
 * direct parent source.
 */
template <typename F, typename... Specs>
struct task_prev_access_valid<bound_task<F, Specs...>>
    : std::bool_constant<
          bound_task_prev_access_bindings_valid_impl<
              std::remove_cvref_t<F>,
              std::tuple<Specs...>>(
              std::index_sequence_for<Specs...> {}) &&
          [] {
              constexpr auto prev_access_count = bound_task_prev_access_count_impl<
                  std::tuple<Specs...>>(
                  std::index_sequence_for<Specs...> {});
              constexpr auto exclusive_prev_access_count =
                  bound_task_exclusive_prev_access_count_impl<
                      std::tuple<Specs...>>(
                      std::index_sequence_for<Specs...> {});

              return exclusive_prev_access_count == 0 ||
                     (exclusive_prev_access_count == 1 && prev_access_count == 1);
          }()> {};

template <typename F, typename... Specs>
struct task_uses_prev_access<bound_task<F, Specs...>>
    : std::bool_constant<
          bound_task_prev_access_count_impl<
              std::tuple<Specs...>>(
              std::index_sequence_for<Specs...> {}) != 0> {};

template <typename F, typename... Specs>
struct task_uses_exclusive_prev_access<bound_task<F, Specs...>>
    : std::bool_constant<
          bound_task_uses_exclusive_prev_access_impl<
              std::tuple<Specs...>>(
              std::index_sequence_for<Specs...> {})> {};

template <typename F, typename... Specs>
struct task_uses_consume_prev<bound_task<F, Specs...>>
    : std::bool_constant<
          bound_task_consume_prev_count_impl<
              std::tuple<Specs...>>(
              std::index_sequence_for<Specs...> {}) != 0> {};

template <typename F, typename... Specs>
struct task_uses_borrow_prev<bound_task<F, Specs...>>
    : std::bool_constant<
          bound_task_borrow_prev_count_impl<
              std::tuple<Specs...>>(
              std::index_sequence_for<Specs...> {}) != 0> {};

template <typename F, typename... Specs>
struct task_uses_borrow_prev_mut<bound_task<F, Specs...>>
    : std::bool_constant<
          bound_task_borrow_prev_mut_count_impl<
              std::tuple<Specs...>>(
              std::index_sequence_for<Specs...> {}) != 0> {};

template <typename F, typename... Specs>
struct task_uses_copy_prev<bound_task<F, Specs...>>
    : std::bool_constant<
          bound_task_copy_prev_count_impl<
              std::tuple<Specs...>>(
              std::index_sequence_for<Specs...> {}) != 0> {};

template <typename F, typename T, typename... Specs>
struct task_prev_access_valid<bound_output_task<F, T, Specs...>>
    : std::bool_constant<
          bound_task_prev_access_bindings_valid_impl<
              std::remove_cvref_t<F>,
              std::tuple<Specs...>>(
              std::index_sequence_for<Specs...> {}) &&
          [] {
              constexpr auto prev_access_count = bound_task_prev_access_count_impl<
                  std::tuple<Specs...>>(
                  std::index_sequence_for<Specs...> {});
              constexpr auto exclusive_prev_access_count =
                  bound_task_exclusive_prev_access_count_impl<
                      std::tuple<Specs...>>(
                      std::index_sequence_for<Specs...> {});

              return exclusive_prev_access_count == 0 ||
                     (exclusive_prev_access_count == 1 && prev_access_count == 1);
          }()> {};

template <typename F, typename T, typename... Specs>
struct task_uses_prev_access<bound_output_task<F, T, Specs...>>
    : std::bool_constant<
          bound_task_prev_access_count_impl<
              std::tuple<Specs...>>(
              std::index_sequence_for<Specs...> {}) != 0> {};

template <typename F, typename T, typename... Specs>
struct task_uses_exclusive_prev_access<bound_output_task<F, T, Specs...>>
    : std::bool_constant<
          bound_task_uses_exclusive_prev_access_impl<
              std::tuple<Specs...>>(
              std::index_sequence_for<Specs...> {})> {};

template <typename F, typename T, typename... Specs>
struct task_uses_consume_prev<bound_output_task<F, T, Specs...>>
    : std::bool_constant<
          bound_task_consume_prev_count_impl<
              std::tuple<Specs...>>(
              std::index_sequence_for<Specs...> {}) != 0> {};

template <typename F, typename T, typename... Specs>
struct task_uses_borrow_prev<bound_output_task<F, T, Specs...>>
    : std::bool_constant<
          bound_task_borrow_prev_count_impl<
              std::tuple<Specs...>>(
              std::index_sequence_for<Specs...> {}) != 0> {};

template <typename F, typename T, typename... Specs>
struct task_uses_borrow_prev_mut<bound_output_task<F, T, Specs...>>
    : std::bool_constant<
          bound_task_borrow_prev_mut_count_impl<
              std::tuple<Specs...>>(
              std::index_sequence_for<Specs...> {}) != 0> {};

template <typename F, typename T, typename... Specs>
struct task_uses_copy_prev<bound_output_task<F, T, Specs...>>
    : std::bool_constant<
          bound_task_copy_prev_count_impl<
              std::tuple<Specs...>>(
              std::index_sequence_for<Specs...> {}) != 0> {};

template <typename F, typename ReceiverSpec, typename... Specs>
struct task_prev_access_valid<bound_member_task<F, ReceiverSpec, Specs...>>
    : std::bool_constant<
          bound_member_task_prev_access_bindings_valid_impl<
              std::remove_cvref_t<F>,
              std::remove_cvref_t<ReceiverSpec>,
              std::tuple<Specs...>>(
              std::index_sequence_for<Specs...> {}) &&
          [] {
              constexpr auto prev_access_count =
                  bound_member_task_prev_access_count_impl<
                      std::remove_cvref_t<ReceiverSpec>,
                      std::tuple<Specs...>>(
                      std::index_sequence_for<Specs...> {});
              constexpr auto exclusive_prev_access_count =
                  bound_member_task_exclusive_prev_access_count_impl<
                      std::remove_cvref_t<ReceiverSpec>,
                      std::tuple<Specs...>>(
                      std::index_sequence_for<Specs...> {});

              return exclusive_prev_access_count == 0 ||
                     (exclusive_prev_access_count == 1 && prev_access_count == 1);
          }()> {};

template <typename F, typename ReceiverSpec, typename... Specs>
struct task_uses_prev_access<bound_member_task<F, ReceiverSpec, Specs...>>
    : std::bool_constant<
          bound_member_task_prev_access_count_impl<
              std::remove_cvref_t<ReceiverSpec>,
              std::tuple<Specs...>>(
              std::index_sequence_for<Specs...> {}) != 0> {};

template <typename F, typename ReceiverSpec, typename... Specs>
struct task_uses_exclusive_prev_access<bound_member_task<F, ReceiverSpec, Specs...>>
    : std::bool_constant<
          bound_member_task_uses_exclusive_prev_access_impl<
              std::remove_cvref_t<ReceiverSpec>,
              std::tuple<Specs...>>(
              std::index_sequence_for<Specs...> {})> {};

template <typename F, typename ReceiverSpec, typename... Specs>
struct task_uses_consume_prev<bound_member_task<F, ReceiverSpec, Specs...>>
    : std::bool_constant<
          bound_member_task_consume_prev_count_impl<
              std::remove_cvref_t<ReceiverSpec>,
              std::tuple<Specs...>>(
              std::index_sequence_for<Specs...> {}) != 0> {};

template <typename F, typename ReceiverSpec, typename... Specs>
struct task_uses_borrow_prev<bound_member_task<F, ReceiverSpec, Specs...>>
    : std::bool_constant<
          bound_member_task_borrow_prev_count_impl<
              std::remove_cvref_t<ReceiverSpec>,
              std::tuple<Specs...>>(
              std::index_sequence_for<Specs...> {}) != 0> {};

template <typename F, typename ReceiverSpec, typename... Specs>
struct task_uses_borrow_prev_mut<bound_member_task<F, ReceiverSpec, Specs...>>
    : std::bool_constant<
          bound_member_task_borrow_prev_mut_count_impl<
              std::remove_cvref_t<ReceiverSpec>,
              std::tuple<Specs...>>(
              std::index_sequence_for<Specs...> {}) != 0> {};

template <typename F, typename ReceiverSpec, typename... Specs>
struct task_uses_copy_prev<bound_member_task<F, ReceiverSpec, Specs...>>
    : std::bool_constant<
          bound_member_task_copy_prev_count_impl<
              std::remove_cvref_t<ReceiverSpec>,
              std::tuple<Specs...>>(
              std::index_sequence_for<Specs...> {}) != 0> {};

template <typename F, typename ReceiverSpec, typename T, typename... Specs>
struct task_prev_access_valid<bound_member_output_task<F, ReceiverSpec, T, Specs...>>
    : std::bool_constant<
          bound_member_task_prev_access_bindings_valid_impl<
              std::remove_cvref_t<F>,
              std::remove_cvref_t<ReceiverSpec>,
              std::tuple<Specs...>>(
              std::index_sequence_for<Specs...> {}) &&
          [] {
              constexpr auto prev_access_count =
                  bound_member_task_prev_access_count_impl<
                      std::remove_cvref_t<ReceiverSpec>,
                      std::tuple<Specs...>>(
                      std::index_sequence_for<Specs...> {});
              constexpr auto exclusive_prev_access_count =
                  bound_member_task_exclusive_prev_access_count_impl<
                      std::remove_cvref_t<ReceiverSpec>,
                      std::tuple<Specs...>>(
                      std::index_sequence_for<Specs...> {});

              return exclusive_prev_access_count == 0 ||
                     (exclusive_prev_access_count == 1 && prev_access_count == 1);
          }()> {};

template <typename F, typename ReceiverSpec, typename T, typename... Specs>
struct task_uses_prev_access<bound_member_output_task<F, ReceiverSpec, T, Specs...>>
    : std::bool_constant<
          bound_member_task_prev_access_count_impl<
              std::remove_cvref_t<ReceiverSpec>,
              std::tuple<Specs...>>(
              std::index_sequence_for<Specs...> {}) != 0> {};

template <typename F, typename ReceiverSpec, typename T, typename... Specs>
struct task_uses_exclusive_prev_access<bound_member_output_task<F, ReceiverSpec, T, Specs...>>
    : std::bool_constant<
          bound_member_task_uses_exclusive_prev_access_impl<
              std::remove_cvref_t<ReceiverSpec>,
              std::tuple<Specs...>>(
              std::index_sequence_for<Specs...> {})> {};

template <typename F, typename ReceiverSpec, typename T, typename... Specs>
struct task_uses_consume_prev<bound_member_output_task<F, ReceiverSpec, T, Specs...>>
    : std::bool_constant<
          bound_member_task_consume_prev_count_impl<
              std::remove_cvref_t<ReceiverSpec>,
              std::tuple<Specs...>>(
              std::index_sequence_for<Specs...> {}) != 0> {};

template <typename F, typename ReceiverSpec, typename T, typename... Specs>
struct task_uses_borrow_prev<bound_member_output_task<F, ReceiverSpec, T, Specs...>>
    : std::bool_constant<
          bound_member_task_borrow_prev_count_impl<
              std::remove_cvref_t<ReceiverSpec>,
              std::tuple<Specs...>>(
              std::index_sequence_for<Specs...> {}) != 0> {};

template <typename F, typename ReceiverSpec, typename T, typename... Specs>
struct task_uses_borrow_prev_mut<bound_member_output_task<F, ReceiverSpec, T, Specs...>>
    : std::bool_constant<
          bound_member_task_borrow_prev_mut_count_impl<
              std::remove_cvref_t<ReceiverSpec>,
              std::tuple<Specs...>>(
              std::index_sequence_for<Specs...> {}) != 0> {};

template <typename F, typename ReceiverSpec, typename T, typename... Specs>
struct task_uses_copy_prev<bound_member_output_task<F, ReceiverSpec, T, Specs...>>
    : std::bool_constant<
          bound_member_task_copy_prev_count_impl<
              std::remove_cvref_t<ReceiverSpec>,
              std::tuple<Specs...>>(
              std::index_sequence_for<Specs...> {}) != 0> {};

template <typename Task>
struct task_prev_access_valid<catch_failure_task<Task>>
    : task_prev_access_valid<Task> {};

template <typename Task>
struct task_uses_prev_access<catch_failure_task<Task>>
    : task_uses_prev_access<Task> {};

template <typename Task>
struct task_uses_exclusive_prev_access<catch_failure_task<Task>>
    : task_uses_exclusive_prev_access<Task> {};

template <typename Task>
struct task_uses_consume_prev<catch_failure_task<Task>>
    : task_uses_consume_prev<Task> {};

template <typename Task>
struct task_uses_borrow_prev<catch_failure_task<Task>>
    : task_uses_borrow_prev<Task> {};

template <typename Task>
struct task_uses_borrow_prev_mut<catch_failure_task<Task>>
    : task_uses_borrow_prev_mut<Task> {};

template <typename Task>
struct task_uses_copy_prev<catch_failure_task<Task>>
    : task_uses_copy_prev<Task> {};

template <typename Task, typename Policy>
struct task_prev_access_valid<catch_failure_with_policy_task<Task, Policy>>
    : task_prev_access_valid<Task> {};

template <typename Task, typename Policy>
struct task_uses_prev_access<catch_failure_with_policy_task<Task, Policy>>
    : task_uses_prev_access<Task> {};

template <typename Task, typename Policy>
struct task_uses_exclusive_prev_access<catch_failure_with_policy_task<Task, Policy>>
    : task_uses_exclusive_prev_access<Task> {};

template <typename Task, typename Policy>
struct task_uses_consume_prev<catch_failure_with_policy_task<Task, Policy>>
    : task_uses_consume_prev<Task> {};

template <typename Task, typename Policy>
struct task_uses_borrow_prev<catch_failure_with_policy_task<Task, Policy>>
    : task_uses_borrow_prev<Task> {};

template <typename Task, typename Policy>
struct task_uses_borrow_prev_mut<catch_failure_with_policy_task<Task, Policy>>
    : task_uses_borrow_prev_mut<Task> {};

template <typename Task, typename Policy>
struct task_uses_copy_prev<catch_failure_with_policy_task<Task, Policy>>
    : task_uses_copy_prev<Task> {};

template <typename Task, typename Policy>
struct task_prev_access_valid<retry_task<Task, Policy>>
    : std::bool_constant<
          task_prev_access_valid<Task>::value &&
          !task_uses_consume_prev<Task>::value> {};

template <typename Task, typename Policy>
struct task_uses_prev_access<retry_task<Task, Policy>>
    : task_uses_prev_access<Task> {};

template <typename Task, typename Policy>
struct task_uses_exclusive_prev_access<retry_task<Task, Policy>>
    : task_uses_exclusive_prev_access<Task> {};

template <typename Task, typename Policy>
struct task_uses_consume_prev<retry_task<Task, Policy>>
    : task_uses_consume_prev<Task> {};

template <typename Task, typename Policy>
struct task_uses_borrow_prev<retry_task<Task, Policy>>
    : task_uses_borrow_prev<Task> {};

template <typename Task, typename Policy>
struct task_uses_borrow_prev_mut<retry_task<Task, Policy>>
    : task_uses_borrow_prev_mut<Task> {};

template <typename Task, typename Policy>
struct task_uses_copy_prev<retry_task<Task, Policy>>
    : task_uses_copy_prev<Task> {};

template <typename Task>
inline constexpr bool task_prev_access_valid_v =
    task_prev_access_valid<std::remove_cvref_t<Task>>::value;

template <typename Task>
inline constexpr bool task_uses_prev_access_v =
    task_uses_prev_access<std::remove_cvref_t<Task>>::value;

template <typename Task>
inline constexpr bool task_uses_exclusive_prev_access_v =
    task_uses_exclusive_prev_access<std::remove_cvref_t<Task>>::value;

template <typename Task>
inline constexpr bool task_uses_consume_prev_v =
    task_uses_consume_prev<std::remove_cvref_t<Task>>::value;

template <typename Task>
inline constexpr bool task_uses_borrow_prev_v =
    task_uses_borrow_prev<std::remove_cvref_t<Task>>::value;

template <typename Task>
inline constexpr bool task_uses_borrow_prev_mut_v =
    task_uses_borrow_prev_mut<std::remove_cvref_t<Task>>::value;

template <typename Task>
inline constexpr bool task_uses_copy_prev_v =
    task_uses_copy_prev<std::remove_cvref_t<Task>>::value;

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
