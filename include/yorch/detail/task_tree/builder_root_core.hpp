#pragma once

#include "builder_root_core_forward.hpp"
#include "builder_root_core_member.hpp"
#include "builder_root_core_ordinary.hpp"

namespace yorch::detail {

template <typename Derived, typename... Nodes>
struct builder_root_core
    : builder_root_core_ordinary<Derived, Nodes...>
    , builder_root_core_member<Derived, Nodes...>
    , builder_root_core_forward<Derived, Nodes...> {
    using builder_root_core_ordinary<Derived, Nodes...>::root;
    using builder_root_core_ordinary<Derived, Nodes...>::root_into;
    using builder_root_core_member<Derived, Nodes...>::root_member;
    using builder_root_core_member<Derived, Nodes...>::root_into_member;
    using builder_root_core_forward<Derived, Nodes...>::root_forward_prev;
    using builder_root_core_forward<Derived, Nodes...>::root_forward_prev_member;
};

} // namespace yorch::detail
