#pragma once

#include "../detail/task_tree/builder_node_core.hpp"
#include "../detail/task_tree/builder_node_diagnostics.hpp"
#include "../detail/task_tree/builder_root_core.hpp"
#include "../detail/task_tree/builder_root_diagnostics.hpp"
#include "../detail/task_tree/builder_storage_base.hpp"

namespace yorch {

template <typename... Nodes>
struct task_tree_builder
    : detail::builder_storage_base<task_tree_builder<Nodes...>, Nodes...>
    , detail::builder_root_core<task_tree_builder<Nodes...>, Nodes...>
    , detail::builder_node_core<task_tree_builder<Nodes...>, Nodes...>
    , detail::builder_root_diagnostics<task_tree_builder<Nodes...>, Nodes...>
    , detail::builder_node_diagnostics<task_tree_builder<Nodes...>, Nodes...> {
        
    using detail::builder_root_core<task_tree_builder<Nodes...>, Nodes...>::root;
    using detail::builder_root_core<task_tree_builder<Nodes...>, Nodes...>::root_into;
    using detail::builder_root_core<task_tree_builder<Nodes...>, Nodes...>::root_member;
    using detail::builder_root_core<task_tree_builder<Nodes...>, Nodes...>::root_into_member;
    using detail::builder_root_core<task_tree_builder<Nodes...>, Nodes...>::root_forward_prev;
    using detail::builder_root_core<task_tree_builder<Nodes...>, Nodes...>::root_forward_prev_member;

    using detail::builder_node_core<task_tree_builder<Nodes...>, Nodes...>::node;
    using detail::builder_node_core<task_tree_builder<Nodes...>, Nodes...>::node_into;
    using detail::builder_node_core<task_tree_builder<Nodes...>, Nodes...>::node_member;
    using detail::builder_node_core<task_tree_builder<Nodes...>, Nodes...>::node_into_member;
    using detail::builder_node_core<task_tree_builder<Nodes...>, Nodes...>::node_forward_prev;
    using detail::builder_node_core<task_tree_builder<Nodes...>, Nodes...>::node_forward_prev_member;

    using detail::builder_root_diagnostics<task_tree_builder<Nodes...>, Nodes...>::root;
    using detail::builder_root_diagnostics<task_tree_builder<Nodes...>, Nodes...>::root_into;
    using detail::builder_root_diagnostics<task_tree_builder<Nodes...>, Nodes...>::root_member;
    using detail::builder_root_diagnostics<task_tree_builder<Nodes...>, Nodes...>::root_into_member;
    using detail::builder_root_diagnostics<task_tree_builder<Nodes...>, Nodes...>::root_forward_prev;
    using detail::builder_root_diagnostics<task_tree_builder<Nodes...>, Nodes...>::root_forward_prev_member;

    using detail::builder_node_diagnostics<task_tree_builder<Nodes...>, Nodes...>::node;
    using detail::builder_node_diagnostics<task_tree_builder<Nodes...>, Nodes...>::node_into;
    using detail::builder_node_diagnostics<task_tree_builder<Nodes...>, Nodes...>::node_member;
    using detail::builder_node_diagnostics<task_tree_builder<Nodes...>, Nodes...>::node_into_member;
    using detail::builder_node_diagnostics<task_tree_builder<Nodes...>, Nodes...>::node_forward_prev;
    using detail::builder_node_diagnostics<task_tree_builder<Nodes...>, Nodes...>::node_forward_prev_member;
};

inline constexpr task_tree_builder<> task_tree {};

} // namespace yorch
