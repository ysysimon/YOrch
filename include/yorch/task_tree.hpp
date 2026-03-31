#pragma once

#include <exception>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "bind.hpp"
#include "task_adapters.hpp"

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

/**
 * @brief Reports whether `F` exposes callable signature metadata understood by
 * `bind(...)`.
 *
 * The task-tree convenience APIs reuse the same signature introspection path
 * as `yorch::bind(...)`. Before those helpers can forward a callable into
 * `bind(...)`, they first need to know whether `function_traits<F>` is
 * well-formed and publishes an `arity` member. This concept acts as that
 * early structural gate, so unsupported callables are rejected during overload
 * selection rather than later from deeper template instantiations.
 *
 * @tparam F Candidate callable type.
 */
template <typename F>
concept bind_callable =
    requires {
        { function_traits<std::remove_cvref_t<F>>::arity } -> std::convertible_to<std::size_t>;
    };

/**
 * @brief Reports whether `F` and `Specs...` can form a valid `bind(...)`
 * invocation.
 *
 * In addition to requiring a callable signature that `bind(...)` can inspect,
 * the convenience APIs also need the same arity check enforced by
 * `yorch::bind(...)`: there must be exactly one spec per function parameter.
 * Encoding that rule here keeps the builder overload set narrow and produces
 * earlier diagnostics when users pass the wrong number of specs.
 *
 * @tparam F Candidate callable type.
 * @tparam Specs Candidate binding spec types.
 */
template <typename F, typename... Specs>
concept bind_signature_matches =
    bind_callable<F> &&
    function_traits<std::remove_cvref_t<F>>::arity == sizeof...(Specs);

/**
 * @brief Reports whether `Policy` looks like a `catch_as_failure(...)`
 * fallback policy.
 *
 * The single-layer task-tree sugar provides both
 * `catch_as_failure(callable, specs...)` and
 * `catch_as_failure(policy, callable, specs...)` forms. This concept helps the
 * overload set distinguish those two cases by checking for the core policy
 * contract used by the underlying adapter: the object must be invocable as
 * `policy(std::exception_ptr)` and that call itself must be `noexcept`.
 *
 * @tparam Policy Candidate fallback policy type.
 */
template <typename Policy>
concept catch_policy_like =
    requires(Policy& policy, std::exception_ptr ep) {
        { policy(ep) } noexcept;
    };

} // namespace yorch::detail

namespace yorch {

/**
 * @brief Immutable builder that records a static tree-shaped node sequence.
 *
 * Each chained `node<Level>(task)` call returns a new builder type whose node
 * list is extended by one compile-time level-tagged entry. This now only
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

    /**
     * @brief Appends the unique root node onto an empty lvalue builder.
     *
     * This is explicit syntax sugar for `node<0>(...)` so the public DSL can
     * state the single-root tree intent more directly without changing the
     * internal representation.
     *
     * @tparam Task Task object type.
     * @param task Task to store as the root node.
     * @return A new builder whose first entry is a `level 0` node.
     */
    template <typename Task>
        requires (sizeof...(Nodes) == 0)
    [[nodiscard]] constexpr auto root(Task&& task) const& {
        return node<0>(std::forward<Task>(task));
    }

    /**
     * @brief Appends the unique root node onto an empty rvalue builder.
     *
     * @tparam Task Task object type.
     * @param task Task to store as the root node.
     * @return A new builder whose first entry is a `level 0` node.
     */
    template <typename Task>
        requires (sizeof...(Nodes) == 0)
    [[nodiscard]] constexpr auto root(Task&& task) && {
        return std::move(*this).template node<0>(std::forward<Task>(task));
    }

    /**
     * @brief Binds a callable and appends it as the root node.
     *
     * This is convenience syntax for `root(yorch::bind(f, specs...))`.
     *
     * @tparam F Callable type.
     * @tparam Specs Binding spec types.
     * @param f Callable to bind.
     * @param specs Binding specs in call order.
     * @return A new builder whose root stores the bound task.
     */
    template <typename F, typename... Specs>
        requires (sizeof...(Nodes) == 0) && detail::bind_signature_matches<F, Specs...>
    [[nodiscard]] constexpr auto root_bind(F&& f, Specs&&... specs) const& {
        return root(yorch::bind(std::forward<F>(f), std::forward<Specs>(specs)...));
    }

    /**
     * @brief Binds a callable and appends it as the root node, moving prior state.
     *
     * This is convenience syntax for `root(yorch::bind(f, specs...))`.
     *
     * @tparam F Callable type.
     * @tparam Specs Binding spec types.
     * @param f Callable to bind.
     * @param specs Binding specs in call order.
     * @return A new builder whose root stores the bound task.
     */
    template <typename F, typename... Specs>
        requires (sizeof...(Nodes) == 0) && detail::bind_signature_matches<F, Specs...>
    [[nodiscard]] constexpr auto root_bind(F&& f, Specs&&... specs) && {
        return std::move(*this).template node<0>(
            yorch::bind(std::forward<F>(f), std::forward<Specs>(specs)...));
    }

    /**
     * @brief Appends a root node wrapped in default `catch_as_failure(...)`.
     *
     * This is convenience syntax for
     * `root(yorch::catch_as_failure(yorch::bind(f, specs...)))`.
     *
     * @tparam F Callable type.
     * @tparam Specs Binding spec types.
     * @param f Callable to bind and wrap.
     * @param specs Binding specs in call order.
     * @return A new builder whose root stores the adapted bound task.
     */
    template <typename F, typename... Specs>
        requires (sizeof...(Nodes) == 0) &&
                 detail::bind_signature_matches<F, Specs...> &&
                 (!detail::catch_policy_like<std::remove_reference_t<F>>)
    [[nodiscard]] constexpr auto root_catch_as_failure(F&& f, Specs&&... specs) const& {
        return root(yorch::catch_as_failure(
            yorch::bind(std::forward<F>(f), std::forward<Specs>(specs)...)));
    }

    /**
     * @brief Appends a root node wrapped in default `catch_as_failure(...)`, moving prior state.
     *
     * This is convenience syntax for
     * `root(yorch::catch_as_failure(yorch::bind(f, specs...)))`.
     *
     * @tparam F Callable type.
     * @tparam Specs Binding spec types.
     * @param f Callable to bind and wrap.
     * @param specs Binding specs in call order.
     * @return A new builder whose root stores the adapted bound task.
     */
    template <typename F, typename... Specs>
        requires (sizeof...(Nodes) == 0) &&
                 detail::bind_signature_matches<F, Specs...> &&
                 (!detail::catch_policy_like<std::remove_reference_t<F>>)
    [[nodiscard]] constexpr auto root_catch_as_failure(F&& f, Specs&&... specs) && {
        return std::move(*this).template node<0>(yorch::catch_as_failure(
            yorch::bind(std::forward<F>(f), std::forward<Specs>(specs)...)));
    }

    /**
     * @brief Appends a root node wrapped in policy-based `catch_as_failure(...)`.
     *
     * This is convenience syntax for
     * `root(yorch::catch_as_failure(yorch::bind(f, specs...), policy))`.
     *
     * @tparam Policy Exception fallback policy type.
     * @tparam F Callable type.
     * @tparam Specs Binding spec types.
     * @param policy Fallback policy invoked on exception.
     * @param f Callable to bind and wrap.
     * @param specs Binding specs in call order.
     * @return A new builder whose root stores the adapted bound task.
     */
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

    /**
     * @brief Appends a root node wrapped in policy-based `catch_as_failure(...)`, moving prior state.
     *
     * This is convenience syntax for
     * `root(yorch::catch_as_failure(yorch::bind(f, specs...), policy))`.
     *
     * @tparam Policy Exception fallback policy type.
     * @tparam F Callable type.
     * @tparam Specs Binding spec types.
     * @param policy Fallback policy invoked on exception.
     * @param f Callable to bind and wrap.
     * @param specs Binding specs in call order.
     * @return A new builder whose root stores the adapted bound task.
     */
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

    /**
     * @brief Appends a root node wrapped in `with_retry(...)`.
     *
     * This is convenience syntax for
     * `root(yorch::with_retry(yorch::bind(f, specs...), policy))`.
     *
     * @tparam Policy Retry policy type.
     * @tparam F Callable type.
     * @tparam Specs Binding spec types.
     * @param policy Retry policy controlling repeated execution.
     * @param f Callable to bind and wrap.
     * @param specs Binding specs in call order.
     * @return A new builder whose root stores the adapted bound task.
     */
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

    /**
     * @brief Appends a root node wrapped in `with_retry(...)`, moving prior state.
     *
     * This is convenience syntax for
     * `root(yorch::with_retry(yorch::bind(f, specs...), policy))`.
     *
     * @tparam Policy Retry policy type.
     * @tparam F Callable type.
     * @tparam Specs Binding spec types.
     * @param policy Retry policy controlling repeated execution.
     * @param f Callable to bind and wrap.
     * @param specs Binding specs in call order.
     * @return A new builder whose root stores the adapted bound task.
     */
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

    /**
     * @brief Binds a callable and appends it as a node at `Level`.
     *
     * This is convenience syntax for `node<Level>(yorch::bind(f, specs...))`.
     *
     * @tparam Level Compile-time tree depth of the new node.
     * @tparam F Callable type.
     * @tparam Specs Binding spec types.
     * @param f Callable to bind.
     * @param specs Binding specs in call order.
     * @return A new builder containing the appended bound task.
     */
    template <std::size_t Level, typename F, typename... Specs>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::bind_signature_matches<F, Specs...>
    [[nodiscard]] constexpr auto node_bind(F&& f, Specs&&... specs) const& {
        return node<Level>(yorch::bind(std::forward<F>(f), std::forward<Specs>(specs)...));
    }

    /**
     * @brief Binds a callable and appends it as a node at `Level`, moving prior state.
     *
     * This is convenience syntax for `node<Level>(yorch::bind(f, specs...))`.
     *
     * @tparam Level Compile-time tree depth of the new node.
     * @tparam F Callable type.
     * @tparam Specs Binding spec types.
     * @param f Callable to bind.
     * @param specs Binding specs in call order.
     * @return A new builder containing the appended bound task.
     */
    template <std::size_t Level, typename F, typename... Specs>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::bind_signature_matches<F, Specs...>
    [[nodiscard]] constexpr auto node_bind(F&& f, Specs&&... specs) && {
        return std::move(*this).template node<Level>(
            yorch::bind(std::forward<F>(f), std::forward<Specs>(specs)...));
    }

    /**
     * @brief Appends a node at `Level` wrapped in default `catch_as_failure(...)`.
     *
     * This is convenience syntax for
     * `node<Level>(yorch::catch_as_failure(yorch::bind(f, specs...)))`.
     *
     * @tparam Level Compile-time tree depth of the new node.
     * @tparam F Callable type.
     * @tparam Specs Binding spec types.
     * @param f Callable to bind and wrap.
     * @param specs Binding specs in call order.
     * @return A new builder containing the appended adapted bound task.
     */
    template <std::size_t Level, typename F, typename... Specs>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::bind_signature_matches<F, Specs...> &&
                 (!detail::catch_policy_like<std::remove_reference_t<F>>)
    [[nodiscard]] constexpr auto node_catch_as_failure(F&& f, Specs&&... specs) const& {
        return node<Level>(yorch::catch_as_failure(
            yorch::bind(std::forward<F>(f), std::forward<Specs>(specs)...)));
    }

    /**
     * @brief Appends a node at `Level` wrapped in default `catch_as_failure(...)`, moving prior state.
     *
     * This is convenience syntax for
     * `node<Level>(yorch::catch_as_failure(yorch::bind(f, specs...)))`.
     *
     * @tparam Level Compile-time tree depth of the new node.
     * @tparam F Callable type.
     * @tparam Specs Binding spec types.
     * @param f Callable to bind and wrap.
     * @param specs Binding specs in call order.
     * @return A new builder containing the appended adapted bound task.
     */
    template <std::size_t Level, typename F, typename... Specs>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::bind_signature_matches<F, Specs...> &&
                 (!detail::catch_policy_like<std::remove_reference_t<F>>)
    [[nodiscard]] constexpr auto node_catch_as_failure(F&& f, Specs&&... specs) && {
        return std::move(*this).template node<Level>(yorch::catch_as_failure(
            yorch::bind(std::forward<F>(f), std::forward<Specs>(specs)...)));
    }

    /**
     * @brief Appends a node at `Level` wrapped in policy-based `catch_as_failure(...)`.
     *
     * This is convenience syntax for
     * `node<Level>(yorch::catch_as_failure(yorch::bind(f, specs...), policy))`.
     *
     * @tparam Level Compile-time tree depth of the new node.
     * @tparam Policy Exception fallback policy type.
     * @tparam F Callable type.
     * @tparam Specs Binding spec types.
     * @param policy Fallback policy invoked on exception.
     * @param f Callable to bind and wrap.
     * @param specs Binding specs in call order.
     * @return A new builder containing the appended adapted bound task.
     */
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

    /**
     * @brief Appends a node at `Level` wrapped in policy-based `catch_as_failure(...)`, moving prior state.
     *
     * This is convenience syntax for
     * `node<Level>(yorch::catch_as_failure(yorch::bind(f, specs...), policy))`.
     *
     * @tparam Level Compile-time tree depth of the new node.
     * @tparam Policy Exception fallback policy type.
     * @tparam F Callable type.
     * @tparam Specs Binding spec types.
     * @param policy Fallback policy invoked on exception.
     * @param f Callable to bind and wrap.
     * @param specs Binding specs in call order.
     * @return A new builder containing the appended adapted bound task.
     */
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

    /**
     * @brief Appends a node at `Level` wrapped in `with_retry(...)`.
     *
     * This is convenience syntax for
     * `node<Level>(yorch::with_retry(yorch::bind(f, specs...), policy))`.
     *
     * @tparam Level Compile-time tree depth of the new node.
     * @tparam Policy Retry policy type.
     * @tparam F Callable type.
     * @tparam Specs Binding spec types.
     * @param policy Retry policy controlling repeated execution.
     * @param f Callable to bind and wrap.
     * @param specs Binding specs in call order.
     * @return A new builder containing the appended adapted bound task.
     */
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

    /**
     * @brief Appends a node at `Level` wrapped in `with_retry(...)`, moving prior state.
     *
     * This is convenience syntax for
     * `node<Level>(yorch::with_retry(yorch::bind(f, specs...), policy))`.
     *
     * @tparam Level Compile-time tree depth of the new node.
     * @tparam Policy Retry policy type.
     * @tparam F Callable type.
     * @tparam Specs Binding spec types.
     * @param policy Retry policy controlling repeated execution.
     * @param f Callable to bind and wrap.
     * @param specs Binding specs in call order.
     * @return A new builder containing the appended adapted bound task.
     */
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
