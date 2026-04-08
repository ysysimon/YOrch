#pragma once

#include <cstddef>
#include <type_traits>

namespace yorch::detail {

template <typename... Nodes>
struct last_node;

template <typename Node>
struct last_node<Node> {
    using type = Node;
};

template <typename Node, typename... Rest>
struct last_node<Node, Rest...> : last_node<Rest...> {};

template <typename... Nodes>
using last_node_t = typename last_node<Nodes...>::type;

template <typename... Nodes>
struct max_level;

template <>
struct max_level<> : std::integral_constant<std::size_t, 0> {};

template <typename Node>
struct max_level<Node> : std::integral_constant<std::size_t, Node::level> {};

template <typename Node, typename Next, typename... Rest>
struct max_level<Node, Next, Rest...>
    : std::integral_constant<
          std::size_t,
          (Node::level > max_level<Next, Rest...>::value
               ? Node::level
               : max_level<Next, Rest...>::value)> {};

template <typename... Nodes>
inline constexpr std::size_t max_level_v = max_level<Nodes...>::value;

template <std::size_t Level, typename... Nodes>
inline constexpr bool append_level_valid_v = [] {
    if constexpr (sizeof...(Nodes) == 0) {
        return Level == 0;
    } else {
        constexpr auto prev_level = last_node_t<Nodes...>::level;
        return Level != 0 && Level <= prev_level + 1;
    }
}();

template <std::size_t Level, typename Task>
struct task_tree_node {
    static constexpr std::size_t level = Level;
    using task_type = Task;

    Task task;
};

template <std::size_t Level, typename Task>
using task_tree_node_t = task_tree_node<Level, std::decay_t<Task>>;

} // namespace yorch::detail
