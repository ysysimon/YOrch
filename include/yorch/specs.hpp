#pragma once

#include <type_traits>
#include <utility>

namespace yorch {

template <typename T>
struct value_spec {
    T stored;
};

template <auto MemberPtr>
struct from_ctx_spec {};

template <typename T>
struct from_prev_spec {};

template <typename T>
[[nodiscard]] constexpr auto value(T&& input) -> value_spec<std::decay_t<T>> {
    return {std::forward<T>(input)};
}

template <auto MemberPtr>
[[nodiscard]] constexpr auto from_ctx() noexcept -> from_ctx_spec<MemberPtr> {
    return {};
}

template <typename T>
[[nodiscard]] constexpr auto from_prev() noexcept -> from_prev_spec<T> {
    return {};
}

}  // namespace yorch
