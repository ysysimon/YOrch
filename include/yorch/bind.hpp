#pragma once

#include <tuple>
#include <type_traits>
#include <utility>

namespace yorch::detail {

template <typename F, typename... Specs>
struct bound_task {
    std::decay_t<F> callable;
    std::tuple<std::decay_t<Specs>...> specs;
};

}  // namespace yorch::detail

namespace yorch {

template <typename F, typename... Specs>
[[nodiscard]] constexpr auto bind(F&& f, Specs&&... specs)
    -> detail::bound_task<F, Specs...> {
    return {
        std::forward<F>(f),
        std::tuple<std::decay_t<Specs>...> {std::forward<Specs>(specs)...},
    };
}

}  // namespace yorch
