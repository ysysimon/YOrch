#pragma once

#include "builder_node_core_forward.hpp"
#include "builder_node_core_member.hpp"
#include "builder_node_core_ordinary.hpp"

namespace yorch::detail {

template <typename Derived, typename... Nodes>
struct builder_node_core
    : builder_node_core_ordinary<Derived, Nodes...>
    , builder_node_core_member<Derived, Nodes...>
    , builder_node_core_forward<Derived, Nodes...> {
    using builder_node_core_ordinary<Derived, Nodes...>::node;
    using builder_node_core_ordinary<Derived, Nodes...>::node_into;
    using builder_node_core_member<Derived, Nodes...>::node_member;
    using builder_node_core_member<Derived, Nodes...>::node_into_member;
    using builder_node_core_forward<Derived, Nodes...>::node_forward_prev;
    using builder_node_core_forward<Derived, Nodes...>::node_forward_prev_member;
};

} // namespace yorch::detail
