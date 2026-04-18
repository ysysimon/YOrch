#pragma once

#include <type_traits>

#include "../../detail/bind/traits.hpp"
#include "../../specs.hpp"

namespace yorch::detail {

enum class prev_access_kind {
    none,
    borrow,
    borrow_mut,
    copy,
    consume,
};

template <typename F>
inline constexpr bool member_receiver_is_const_v =
    std::is_const_v<std::remove_reference_t<member_receiver_arg_t<F>>>;

template <typename Spec>
struct prev_access_spec_traits {
    using payload_type = void;

    static constexpr prev_access_kind kind = prev_access_kind::none;
    static constexpr bool is_prev_access = false;
    static constexpr bool is_exclusive = false;

    template <typename Arg>
    static constexpr bool matches_arg = false;

    template <typename F>
    static constexpr bool matches_member_receiver = false;
};

template <typename T>
struct prev_access_spec_traits<borrow_prev_t<T>> {
    using payload_type = typename borrow_prev_t<T>::type;

    static constexpr prev_access_kind kind = prev_access_kind::borrow;
    static constexpr bool is_prev_access = true;
    static constexpr bool is_exclusive = false;

    template <typename Arg>
    static constexpr bool matches_arg =
        std::is_same_v<Arg, const payload_type&>;

    template <typename F>
    static constexpr bool matches_member_receiver =
        std::is_same_v<payload_type, member_class_t<F>> &&
        member_receiver_is_const_v<F>;
};

template <typename T>
struct prev_access_spec_traits<borrow_prev_mut_t<T>> {
    using payload_type = typename borrow_prev_mut_t<T>::type;

    static constexpr prev_access_kind kind = prev_access_kind::borrow_mut;
    static constexpr bool is_prev_access = true;
    static constexpr bool is_exclusive = true;

    template <typename Arg>
    static constexpr bool matches_arg =
        std::is_same_v<Arg, payload_type&>;

    template <typename F>
    static constexpr bool matches_member_receiver =
        std::is_same_v<payload_type, member_class_t<F>>;
};

template <typename T>
struct prev_access_spec_traits<copy_prev_t<T>> {
    using payload_type = typename copy_prev_t<T>::type;

    static constexpr prev_access_kind kind = prev_access_kind::copy;
    static constexpr bool is_prev_access = true;
    static constexpr bool is_exclusive = false;

    template <typename Arg>
    static constexpr bool matches_arg =
        std::is_same_v<Arg, payload_type>;

    template <typename F>
    static constexpr bool matches_member_receiver =
        std::is_same_v<payload_type, member_class_t<F>>;
};

template <typename T>
struct prev_access_spec_traits<consume_prev_t<T>> {
    using payload_type = typename consume_prev_t<T>::type;

    static constexpr prev_access_kind kind = prev_access_kind::consume;
    static constexpr bool is_prev_access = true;
    static constexpr bool is_exclusive = true;

    template <typename Arg>
    static constexpr bool matches_arg =
        std::is_same_v<Arg, payload_type> ||
        std::is_same_v<Arg, payload_type&&>;

    template <typename F>
    static constexpr bool matches_member_receiver =
        std::is_same_v<payload_type, member_class_t<F>>;
};

template <typename Spec>
using normalized_prev_access_spec_traits =
    prev_access_spec_traits<std::remove_cvref_t<Spec>>;

template <typename Spec>
struct is_borrow_prev_spec
    : std::bool_constant<
          normalized_prev_access_spec_traits<Spec>::kind ==
          prev_access_kind::borrow> {};

template <typename Spec>
inline constexpr bool is_borrow_prev_spec_v =
    is_borrow_prev_spec<Spec>::value;

template <typename Spec>
struct is_borrow_prev_mut_spec
    : std::bool_constant<
          normalized_prev_access_spec_traits<Spec>::kind ==
          prev_access_kind::borrow_mut> {};

template <typename Spec>
inline constexpr bool is_borrow_prev_mut_spec_v =
    is_borrow_prev_mut_spec<Spec>::value;

template <typename Spec>
struct is_copy_prev_spec
    : std::bool_constant<
          normalized_prev_access_spec_traits<Spec>::kind ==
          prev_access_kind::copy> {};

template <typename Spec>
inline constexpr bool is_copy_prev_spec_v =
    is_copy_prev_spec<Spec>::value;

template <typename Spec>
struct is_consume_prev_spec
    : std::bool_constant<
          normalized_prev_access_spec_traits<Spec>::kind ==
          prev_access_kind::consume> {};

template <typename Spec>
inline constexpr bool is_consume_prev_spec_v =
    is_consume_prev_spec<Spec>::value;

template <typename Spec>
inline constexpr bool is_prev_access_spec_v =
    normalized_prev_access_spec_traits<Spec>::is_prev_access;

template <typename Spec>
inline constexpr bool is_exclusive_prev_access_spec_v =
    normalized_prev_access_spec_traits<Spec>::is_exclusive;

template <typename Spec, typename Arg>
struct prev_access_binding_valid
    : std::bool_constant<
          normalized_prev_access_spec_traits<Spec>::template matches_arg<Arg>> {};

template <typename Spec, typename Arg>
inline constexpr bool prev_access_binding_valid_v =
    prev_access_binding_valid<Spec, Arg>::value;

template <typename Spec, typename F>
struct member_receiver_prev_access_valid
    : std::bool_constant<
          normalized_prev_access_spec_traits<Spec>::template matches_member_receiver<
              std::remove_cvref_t<F>>> {};

template <typename Spec, typename F>
inline constexpr bool member_receiver_prev_access_valid_v =
    member_receiver_prev_access_valid<Spec, F>::value;

} // namespace yorch::detail
