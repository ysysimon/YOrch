#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "metadata.hpp"

namespace yorch::detail {

template <typename Derived, typename... Nodes>
struct builder_storage_base {
    using tuple_type = std::tuple<Nodes...>;

    tuple_type nodes {};

    static constexpr std::size_t node_count = sizeof...(Nodes);
    static constexpr std::size_t max_level = detail::max_level_v<Nodes...>;

    template <std::size_t I>
    using node_type = std::tuple_element_t<I, tuple_type>;

    template <std::size_t I>
    [[nodiscard]] constexpr auto& entry() & noexcept {
        return std::get<I>(nodes);
    }

    template <std::size_t I>
    [[nodiscard]] constexpr const auto& entry() const& noexcept {
        return std::get<I>(nodes);
    }

    template <std::size_t I>
    [[nodiscard]] constexpr auto&& entry() && noexcept {
        return std::get<I>(std::move(nodes));
    }
};

} // namespace yorch::detail
