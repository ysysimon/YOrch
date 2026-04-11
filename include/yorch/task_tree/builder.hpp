#pragma once

#include <tuple>
#include <type_traits>
#include <utility>

#include "../detail/task_tree/concepts.hpp" // IWYU pragma: keep
#include "../detail/task_tree/metadata.hpp"

namespace yorch {

/**
 * @brief Immutable builder that records a static tree-shaped node sequence.
 *
 * Each chained `node<Level>(task)` call returns a new builder type whose node
 * list is extended by one compile-time level-tagged entry. The builder only
 * records the static shape; plan compilation and execution are handled later.
 *
 * @tparam Nodes Previously appended node record types.
 */
template <typename... Nodes>
struct task_tree_builder {
    using tuple_type = std::tuple<Nodes...>;

    /// Stored node records in insertion order.
    tuple_type nodes {};

    /// Number of nodes currently recorded in this builder.
    static constexpr std::size_t node_count = sizeof...(Nodes);

    /// Maximum level observed in the current node list.
    static constexpr std::size_t max_level = detail::max_level_v<Nodes...>;

    template <std::size_t I>
    using node_type = std::tuple_element_t<I, tuple_type>;

    /// Appends the unique root node onto an empty builder.
    template <typename Task>
        requires (sizeof...(Nodes) == 0)
    [[nodiscard]] constexpr auto root(Task&& task) const& {
        return node<0>(std::forward<Task>(task));
    }

    template <typename Task, typename FanoutPolicy>
        requires (sizeof...(Nodes) == 0) &&
                 detail::fanout_policy<FanoutPolicy>
    [[nodiscard]] constexpr auto root(Task&& task, FanoutPolicy&& fanout_policy) const& {
        return node<0>(
            std::forward<Task>(task),
            std::forward<FanoutPolicy>(fanout_policy));
    }

    /// Appends the unique root node onto an empty builder, moving prior state.
    template <typename Task>
        requires (sizeof...(Nodes) == 0)
    [[nodiscard]] constexpr auto root(Task&& task) && {
        return std::move(*this).template node<0>(std::forward<Task>(task));
    }

    template <typename Task, typename FanoutPolicy>
        requires (sizeof...(Nodes) == 0) &&
                 detail::fanout_policy<FanoutPolicy>
    [[nodiscard]] constexpr auto root(Task&& task, FanoutPolicy&& fanout_policy) && {
        return std::move(*this).template node<0>(
            std::forward<Task>(task),
            std::forward<FanoutPolicy>(fanout_policy));
    }

    /// Convenience syntax for `root(yorch::bind(f, specs...))`.
    template <typename F, typename... Specs>
        requires (sizeof...(Nodes) == 0) && detail::bind_signature_matches<F, Specs...>
    [[nodiscard]] constexpr auto root_bind(F&& f, Specs&&... specs) const& {
        return root(yorch::bind(std::forward<F>(f), std::forward<Specs>(specs)...));
    }

    /// Rvalue overload of `root_bind(...)`.
    template <typename F, typename... Specs>
        requires (sizeof...(Nodes) == 0) && detail::bind_signature_matches<F, Specs...>
    [[nodiscard]] constexpr auto root_bind(F&& f, Specs&&... specs) && {
        return std::move(*this).template node<0>(
            yorch::bind(std::forward<F>(f), std::forward<Specs>(specs)...));
    }

    /// Convenience syntax for `root(yorch::bind_into<T>(f, specs...))`.
    template <typename T, typename F, typename... Specs>
        requires (sizeof...(Nodes) == 0) && detail::bind_into_signature_matches<T, F, Specs...>
    [[nodiscard]] constexpr auto root_bind_into(F&& f, Specs&&... specs) const& {
        return root(yorch::bind_into<T>(std::forward<F>(f), std::forward<Specs>(specs)...));
    }

    /// Rvalue overload of `root_bind_into(...)`.
    template <typename T, typename F, typename... Specs>
        requires (sizeof...(Nodes) == 0) && detail::bind_into_signature_matches<T, F, Specs...>
    [[nodiscard]] constexpr auto root_bind_into(F&& f, Specs&&... specs) && {
        return std::move(*this).template node<0>(
            yorch::bind_into<T>(std::forward<F>(f), std::forward<Specs>(specs)...));
    }

    /// Convenience syntax for `root(yorch::catch_as_failure(yorch::bind(...)))`.
    template <typename F, typename... Specs>
        requires (sizeof...(Nodes) == 0) &&
                 detail::bind_signature_matches<F, Specs...> &&
                 (!detail::catch_policy_like<std::remove_reference_t<F>>)
    [[nodiscard]] constexpr auto root_catch_as_failure(F&& f, Specs&&... specs) const& {
        return root(yorch::catch_as_failure(
            yorch::bind(std::forward<F>(f), std::forward<Specs>(specs)...)));
    }

    /// Rvalue overload of `root_catch_as_failure(...)`.
    template <typename F, typename... Specs>
        requires (sizeof...(Nodes) == 0) &&
                 detail::bind_signature_matches<F, Specs...> &&
                 (!detail::catch_policy_like<std::remove_reference_t<F>>)
    [[nodiscard]] constexpr auto root_catch_as_failure(F&& f, Specs&&... specs) && {
        return std::move(*this).template node<0>(yorch::catch_as_failure(
            yorch::bind(std::forward<F>(f), std::forward<Specs>(specs)...)));
    }

    /// Convenience syntax for `root(yorch::catch_as_failure(yorch::bind_into<T>(...)))`.
    template <typename T, typename F, typename... Specs>
        requires (sizeof...(Nodes) == 0) &&
                 detail::bind_into_signature_matches<T, F, Specs...> &&
                 (!detail::catch_policy_like<std::remove_reference_t<F>>)
    [[nodiscard]] constexpr auto root_catch_as_failure_into(F&& f, Specs&&... specs) const& {
        return root(yorch::catch_as_failure(
            yorch::bind_into<T>(std::forward<F>(f), std::forward<Specs>(specs)...)));
    }

    /// Rvalue overload of `root_catch_as_failure_into(...)`.
    template <typename T, typename F, typename... Specs>
        requires (sizeof...(Nodes) == 0) &&
                 detail::bind_into_signature_matches<T, F, Specs...> &&
                 (!detail::catch_policy_like<std::remove_reference_t<F>>)
    [[nodiscard]] constexpr auto root_catch_as_failure_into(F&& f, Specs&&... specs) && {
        return std::move(*this).template node<0>(yorch::catch_as_failure(
            yorch::bind_into<T>(std::forward<F>(f), std::forward<Specs>(specs)...)));
    }

    /// Policy-based `catch_as_failure(...)` root sugar.
    template <typename Policy, typename F, typename... Specs>
        requires (sizeof...(Nodes) == 0) &&
                 detail::catch_policy_like<std::remove_reference_t<Policy>> &&
                 detail::bind_signature_matches<F, Specs...>
    [[nodiscard]] constexpr auto root_catch_as_failure(
        Policy&& policy,
        F&& f,
        Specs&&... specs) const& {
        return root(yorch::catch_as_failure(
            yorch::bind(std::forward<F>(f), std::forward<Specs>(specs)...),
            std::forward<Policy>(policy)));
    }

    /// Rvalue overload of policy-based `root_catch_as_failure(...)`.
    template <typename Policy, typename F, typename... Specs>
        requires (sizeof...(Nodes) == 0) &&
                 detail::catch_policy_like<std::remove_reference_t<Policy>> &&
                 detail::bind_signature_matches<F, Specs...>
    [[nodiscard]] constexpr auto root_catch_as_failure(
        Policy&& policy,
        F&& f,
        Specs&&... specs) && {
        return std::move(*this).template node<0>(yorch::catch_as_failure(
            yorch::bind(std::forward<F>(f), std::forward<Specs>(specs)...),
            std::forward<Policy>(policy)));
    }

    /// Policy-based `catch_as_failure(...)` sugar for direct-output roots.
    template <typename T, typename Policy, typename F, typename... Specs>
        requires (sizeof...(Nodes) == 0) &&
                 detail::catch_policy_like<std::remove_reference_t<Policy>> &&
                 detail::bind_into_signature_matches<T, F, Specs...>
    [[nodiscard]] constexpr auto root_catch_as_failure_into(
        Policy&& policy,
        F&& f,
        Specs&&... specs) const& {
        return root(yorch::catch_as_failure(
            yorch::bind_into<T>(std::forward<F>(f), std::forward<Specs>(specs)...),
            std::forward<Policy>(policy)));
    }

    /// Rvalue overload of policy-based `root_catch_as_failure_into(...)`.
    template <typename T, typename Policy, typename F, typename... Specs>
        requires (sizeof...(Nodes) == 0) &&
                 detail::catch_policy_like<std::remove_reference_t<Policy>> &&
                 detail::bind_into_signature_matches<T, F, Specs...>
    [[nodiscard]] constexpr auto root_catch_as_failure_into(
        Policy&& policy,
        F&& f,
        Specs&&... specs) && {
        return std::move(*this).template node<0>(yorch::catch_as_failure(
            yorch::bind_into<T>(std::forward<F>(f), std::forward<Specs>(specs)...),
            std::forward<Policy>(policy)));
    }

    /// Convenience syntax for `root(yorch::with_retry(...))`.
    template <typename Policy, typename F, typename... Specs>
        requires (sizeof...(Nodes) == 0) &&
                 retry_policy<std::decay_t<Policy>> &&
                 detail::bind_signature_matches<F, Specs...>
    [[nodiscard]] constexpr auto root_with_retry(
        Policy&& policy,
        F&& f,
        Specs&&... specs) const& {
        return root(yorch::with_retry(
            yorch::bind(std::forward<F>(f), std::forward<Specs>(specs)...),
            std::forward<Policy>(policy)));
    }

    /// Rvalue overload of `root_with_retry(...)`.
    template <typename Policy, typename F, typename... Specs>
        requires (sizeof...(Nodes) == 0) &&
                 retry_policy<std::decay_t<Policy>> &&
                 detail::bind_signature_matches<F, Specs...>
    [[nodiscard]] constexpr auto root_with_retry(
        Policy&& policy,
        F&& f,
        Specs&&... specs) && {
        return std::move(*this).template node<0>(yorch::with_retry(
            yorch::bind(std::forward<F>(f), std::forward<Specs>(specs)...),
            std::forward<Policy>(policy)));
    }

    /// Convenience syntax for `root(yorch::with_retry(yorch::bind_into<T>(...)))`.
    template <typename T, typename Policy, typename F, typename... Specs>
        requires (sizeof...(Nodes) == 0) &&
                 retry_policy<std::decay_t<Policy>> &&
                 detail::bind_into_signature_matches<T, F, Specs...>
    [[nodiscard]] constexpr auto root_with_retry_into(
        Policy&& policy,
        F&& f,
        Specs&&... specs) const& {
        return root(yorch::with_retry(
            yorch::bind_into<T>(std::forward<F>(f), std::forward<Specs>(specs)...),
            std::forward<Policy>(policy)));
    }

    /// Rvalue overload of `root_with_retry_into(...)`.
    template <typename T, typename Policy, typename F, typename... Specs>
        requires (sizeof...(Nodes) == 0) &&
                 retry_policy<std::decay_t<Policy>> &&
                 detail::bind_into_signature_matches<T, F, Specs...>
    [[nodiscard]] constexpr auto root_with_retry_into(
        Policy&& policy,
        F&& f,
        Specs&&... specs) && {
        return std::move(*this).template node<0>(yorch::with_retry(
            yorch::bind_into<T>(std::forward<F>(f), std::forward<Specs>(specs)...),
            std::forward<Policy>(policy)));
    }

    /**
     * @brief Appends a node onto an lvalue builder.
     *
     * Structural rules for this phase:
     * - the first node must be `level 0` or be introduced through `root(...)`
     * - later nodes cannot descend more than one level relative to the previous node
     * - later nodes cannot start another `level 0` root
     *
     * @tparam Level Compile-time tree depth of the new node.
     * @tparam Task Task object type.
     * @param task Task to store in the appended node record.
     * @return A new builder value containing the extra node.
     */
    template <std::size_t Level, typename Task>
        requires (detail::append_level_valid_v<Level, Nodes...>)
    [[nodiscard]] constexpr auto node(Task&& task) const& {
        using next_node_t = detail::task_tree_node_t<Level, Task>;

        return task_tree_builder<Nodes..., next_node_t> {
            std::tuple_cat(
                nodes,
                std::tuple<next_node_t> {
                    next_node_t {std::forward<Task>(task)}
                })
        };
    }

    template <std::size_t Level, typename Task, typename FanoutPolicy>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::fanout_policy<FanoutPolicy>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    [[nodiscard]] constexpr auto node(Task&& task, FanoutPolicy&& fanout_policy) const& {
        static_cast<void>(fanout_policy);
        using next_node_t = detail::task_tree_node_t<Level, Task, FanoutPolicy>;

        return task_tree_builder<Nodes..., next_node_t> {
            std::tuple_cat(
                nodes,
                std::tuple<next_node_t> {
                    next_node_t {std::forward<Task>(task)}
                })
        };
    }

    /**
     * @brief Appends a node onto an rvalue builder, moving existing nodes.
     *
     * This overload is important for chains that already contain move-only
     * task objects.
     *
     * @tparam Level Compile-time tree depth of the new node.
     * @tparam Task Task object type.
     * @param task Task to store in the appended node record.
     * @return A new builder value containing the extra node.
     */
    template <std::size_t Level, typename Task>
        requires (detail::append_level_valid_v<Level, Nodes...>)
    [[nodiscard]] constexpr auto node(Task&& task) && {
        using next_node_t = detail::task_tree_node_t<Level, Task>;

        return task_tree_builder<Nodes..., next_node_t> {
            std::tuple_cat(
                std::move(nodes),
                std::tuple<next_node_t> {
                    next_node_t {std::forward<Task>(task)}
                })
        };
    }

    template <std::size_t Level, typename Task, typename FanoutPolicy>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::fanout_policy<FanoutPolicy>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    [[nodiscard]] constexpr auto node(Task&& task, FanoutPolicy&& fanout_policy) && {
        static_cast<void>(fanout_policy);
        using next_node_t = detail::task_tree_node_t<Level, Task, FanoutPolicy>;

        return task_tree_builder<Nodes..., next_node_t> {
            std::tuple_cat(
                std::move(nodes),
                std::tuple<next_node_t> {
                    next_node_t {std::forward<Task>(task)}
                })
        };
    }

    /// Convenience syntax for `node<Level>(yorch::bind(f, specs...))`.
    template <std::size_t Level, typename F, typename... Specs>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::bind_signature_matches<F, Specs...>
    [[nodiscard]] constexpr auto node_bind(F&& f, Specs&&... specs) const& {
        return node<Level>(yorch::bind(std::forward<F>(f), std::forward<Specs>(specs)...));
    }

    /// Rvalue overload of `node_bind(...)`.
    template <std::size_t Level, typename F, typename... Specs>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::bind_signature_matches<F, Specs...>
    [[nodiscard]] constexpr auto node_bind(F&& f, Specs&&... specs) && {
        return std::move(*this).template node<Level>(
            yorch::bind(std::forward<F>(f), std::forward<Specs>(specs)...));
    }

    /// Convenience syntax for `node<Level>(yorch::bind_into<T>(f, specs...))`.
    template <std::size_t Level, typename T, typename F, typename... Specs>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::bind_into_signature_matches<T, F, Specs...>
    [[nodiscard]] constexpr auto node_bind_into(F&& f, Specs&&... specs) const& {
        return node<Level>(yorch::bind_into<T>(std::forward<F>(f), std::forward<Specs>(specs)...));
    }

    /// Rvalue overload of `node_bind_into(...)`.
    template <std::size_t Level, typename T, typename F, typename... Specs>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::bind_into_signature_matches<T, F, Specs...>
    [[nodiscard]] constexpr auto node_bind_into(F&& f, Specs&&... specs) && {
        return std::move(*this).template node<Level>(
            yorch::bind_into<T>(std::forward<F>(f), std::forward<Specs>(specs)...));
    }

    /// Convenience syntax for `node<Level>(yorch::catch_as_failure(yorch::bind(...)))`.
    template <std::size_t Level, typename F, typename... Specs>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::bind_signature_matches<F, Specs...> &&
                 (!detail::catch_policy_like<std::remove_reference_t<F>>)
    [[nodiscard]] constexpr auto node_catch_as_failure(F&& f, Specs&&... specs) const& {
        return node<Level>(yorch::catch_as_failure(
            yorch::bind(std::forward<F>(f), std::forward<Specs>(specs)...)));
    }

    /// Rvalue overload of `node_catch_as_failure(...)`.
    template <std::size_t Level, typename F, typename... Specs>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::bind_signature_matches<F, Specs...> &&
                 (!detail::catch_policy_like<std::remove_reference_t<F>>)
    [[nodiscard]] constexpr auto node_catch_as_failure(F&& f, Specs&&... specs) && {
        return std::move(*this).template node<Level>(yorch::catch_as_failure(
            yorch::bind(std::forward<F>(f), std::forward<Specs>(specs)...)));
    }

    /// Convenience syntax for `node<Level>(yorch::catch_as_failure(yorch::bind_into<T>(...)))`.
    template <std::size_t Level, typename T, typename F, typename... Specs>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::bind_into_signature_matches<T, F, Specs...> &&
                 (!detail::catch_policy_like<std::remove_reference_t<F>>)
    [[nodiscard]] constexpr auto node_catch_as_failure_into(F&& f, Specs&&... specs) const& {
        return node<Level>(yorch::catch_as_failure(
            yorch::bind_into<T>(std::forward<F>(f), std::forward<Specs>(specs)...)));
    }

    /// Rvalue overload of `node_catch_as_failure_into(...)`.
    template <std::size_t Level, typename T, typename F, typename... Specs>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::bind_into_signature_matches<T, F, Specs...> &&
                 (!detail::catch_policy_like<std::remove_reference_t<F>>)
    [[nodiscard]] constexpr auto node_catch_as_failure_into(F&& f, Specs&&... specs) && {
        return std::move(*this).template node<Level>(yorch::catch_as_failure(
            yorch::bind_into<T>(std::forward<F>(f), std::forward<Specs>(specs)...)));
    }

    /// Policy-based `catch_as_failure(...)` node sugar.
    template <std::size_t Level, typename Policy, typename F, typename... Specs>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::catch_policy_like<std::remove_reference_t<Policy>> &&
                 detail::bind_signature_matches<F, Specs...>
    [[nodiscard]] constexpr auto node_catch_as_failure(
        Policy&& policy,
        F&& f,
        Specs&&... specs) const& {
        return node<Level>(yorch::catch_as_failure(
            yorch::bind(std::forward<F>(f), std::forward<Specs>(specs)...),
            std::forward<Policy>(policy)));
    }

    /// Rvalue overload of policy-based `node_catch_as_failure(...)`.
    template <std::size_t Level, typename Policy, typename F, typename... Specs>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::catch_policy_like<std::remove_reference_t<Policy>> &&
                 detail::bind_signature_matches<F, Specs...>
    [[nodiscard]] constexpr auto node_catch_as_failure(
        Policy&& policy,
        F&& f,
        Specs&&... specs) && {
        return std::move(*this).template node<Level>(yorch::catch_as_failure(
            yorch::bind(std::forward<F>(f), std::forward<Specs>(specs)...),
            std::forward<Policy>(policy)));
    }

    /// Policy-based `catch_as_failure(...)` sugar for direct-output nodes.
    template <std::size_t Level, typename T, typename Policy, typename F, typename... Specs>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::catch_policy_like<std::remove_reference_t<Policy>> &&
                 detail::bind_into_signature_matches<T, F, Specs...>
    [[nodiscard]] constexpr auto node_catch_as_failure_into(
        Policy&& policy,
        F&& f,
        Specs&&... specs) const& {
        return node<Level>(yorch::catch_as_failure(
            yorch::bind_into<T>(std::forward<F>(f), std::forward<Specs>(specs)...),
            std::forward<Policy>(policy)));
    }

    /// Rvalue overload of policy-based `node_catch_as_failure_into(...)`.
    template <std::size_t Level, typename T, typename Policy, typename F, typename... Specs>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::catch_policy_like<std::remove_reference_t<Policy>> &&
                 detail::bind_into_signature_matches<T, F, Specs...>
    [[nodiscard]] constexpr auto node_catch_as_failure_into(
        Policy&& policy,
        F&& f,
        Specs&&... specs) && {
        return std::move(*this).template node<Level>(yorch::catch_as_failure(
            yorch::bind_into<T>(std::forward<F>(f), std::forward<Specs>(specs)...),
            std::forward<Policy>(policy)));
    }

    /// Convenience syntax for `node<Level>(yorch::with_retry(...))`.
    template <std::size_t Level, typename Policy, typename F, typename... Specs>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 retry_policy<std::decay_t<Policy>> &&
                 detail::bind_signature_matches<F, Specs...>
    [[nodiscard]] constexpr auto node_with_retry(
        Policy&& policy,
        F&& f,
        Specs&&... specs) const& {
        return node<Level>(yorch::with_retry(
            yorch::bind(std::forward<F>(f), std::forward<Specs>(specs)...),
            std::forward<Policy>(policy)));
    }

    /// Rvalue overload of `node_with_retry(...)`.
    template <std::size_t Level, typename Policy, typename F, typename... Specs>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 retry_policy<std::decay_t<Policy>> &&
                 detail::bind_signature_matches<F, Specs...>
    [[nodiscard]] constexpr auto node_with_retry(
        Policy&& policy,
        F&& f,
        Specs&&... specs) && {
        return std::move(*this).template node<Level>(yorch::with_retry(
            yorch::bind(std::forward<F>(f), std::forward<Specs>(specs)...),
            std::forward<Policy>(policy)));
    }

    /// Convenience syntax for `node<Level>(yorch::with_retry(yorch::bind_into<T>(...)))`.
    template <std::size_t Level, typename T, typename Policy, typename F, typename... Specs>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 retry_policy<std::decay_t<Policy>> &&
                 detail::bind_into_signature_matches<T, F, Specs...>
    [[nodiscard]] constexpr auto node_with_retry_into(
        Policy&& policy,
        F&& f,
        Specs&&... specs) const& {
        return node<Level>(yorch::with_retry(
            yorch::bind_into<T>(std::forward<F>(f), std::forward<Specs>(specs)...),
            std::forward<Policy>(policy)));
    }

    /// Rvalue overload of `node_with_retry_into(...)`.
    template <std::size_t Level, typename T, typename Policy, typename F, typename... Specs>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 retry_policy<std::decay_t<Policy>> &&
                 detail::bind_into_signature_matches<T, F, Specs...>
    [[nodiscard]] constexpr auto node_with_retry_into(
        Policy&& policy,
        F&& f,
        Specs&&... specs) && {
        return std::move(*this).template node<Level>(yorch::with_retry(
            yorch::bind_into<T>(std::forward<F>(f), std::forward<Specs>(specs)...),
            std::forward<Policy>(policy)));
    }

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

/// @brief Empty task tree builder value used as the starting point for `root(...)`.
inline constexpr task_tree_builder<> task_tree {};

} // namespace yorch
