#pragma once

#include <functional>
#include <type_traits>
#include <utility>

#include "../../resolve.hpp"
#include "traits.hpp"

namespace yorch::detail {

template <typename T>
struct borrowed_member_receiver {
    std::reference_wrapper<T> ref;
};

template <typename T>
struct copied_member_receiver {
    T value;
};

template <typename T>
struct consumed_member_receiver {
    T value;
};

template <typename T>
constexpr decltype(auto) forward_member_receiver(const borrowed_member_receiver<T>& holder) noexcept {
    return holder.ref.get();
}

template <typename T>
constexpr decltype(auto) forward_member_receiver(copied_member_receiver<T>& holder) noexcept {
    return (holder.value);
}

/**
 * Keep this rvalue overload so `invoke_member_with_receiver(...)` can accept a
 * forwarded temporary `copied_member_receiver<T>`.
 *
 * We still return `holder.value` as an lvalue on purpose: `copy_prev<T>()`
 * means "invoke the member function on a local copy", not "move the copied
 * receiver into the member call".
 */
template <typename T>
constexpr decltype(auto) forward_member_receiver
// NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
(copied_member_receiver<T>&& holder) noexcept {
    return (holder.value);
}

template <typename T>
constexpr decltype(auto) forward_member_receiver(consumed_member_receiver<T>& holder) noexcept {
    return std::move(holder.value);
}

template <typename T>
constexpr decltype(auto) forward_member_receiver(consumed_member_receiver<T>&& holder) noexcept {
    return std::move(holder.value);
}

template <typename F, typename Receiver, typename... Args>
constexpr decltype(auto) invoke_member_with_receiver(F&& func, Receiver&& receiver, Args&&... args)
    noexcept(noexcept(std::invoke(
        std::forward<F>(func),
        forward_member_receiver(std::forward<Receiver>(receiver)),
        std::forward<Args>(args)...))) {
    return std::invoke(
        std::forward<F>(func),
        forward_member_receiver(std::forward<Receiver>(receiver)),
        std::forward<Args>(args)...);
}

template <typename F, typename T, typename Ctx, typename Prev>
constexpr auto resolve_member_receiver(copy_prev_t<T> spec, exec_context<Ctx, Prev>& ec)
    noexcept(noexcept(resolve_as<member_class_t<F>>(spec, ec))) {
    using class_t = member_class_t<F>;

    static_assert(std::is_same_v<typename copy_prev_t<T>::type, class_t>,
                  "copy_prev<T>() receiver must match the member-function class type exactly");

    return copied_member_receiver<class_t> {
        resolve_as<class_t>(spec, ec)
    };
}

template <typename F, typename T, typename Ctx, typename Prev>
constexpr auto resolve_member_receiver(consume_prev_t<T> spec, exec_context<Ctx, Prev>& ec)
    noexcept(noexcept(resolve_as<member_class_t<F>>(spec, ec))) {
    using class_t = member_class_t<F>;

    static_assert(std::is_same_v<typename consume_prev_t<T>::type, class_t>,
                  "consume_prev<T>() receiver must match the member-function class type exactly");

    return consumed_member_receiver<class_t> {
        resolve_as<class_t>(spec, ec)
    };
}

template <typename F, typename Spec, typename Ctx, typename Prev>
constexpr auto resolve_member_receiver(Spec& spec, exec_context<Ctx, Prev>& ec)
    noexcept(noexcept(resolve_as<member_receiver_arg_t<F>>(spec, ec))) {
    using receiver_t = member_receiver_arg_t<F>;
    using borrowed_t = member_borrowed_receiver_t<F>;

    auto&& receiver = resolve_as<receiver_t>(spec, ec);

    return borrowed_member_receiver<borrowed_t> {
        std::reference_wrapper<borrowed_t> {receiver}
    };
}

template <typename Self, typename T>
[[nodiscard]] constexpr decltype(auto) forward_member(T& value) noexcept {
    if constexpr (std::is_lvalue_reference_v<Self&&>) {
        return (value);
    } else {
        return std::move(value);
    }
}

} // namespace yorch::detail
