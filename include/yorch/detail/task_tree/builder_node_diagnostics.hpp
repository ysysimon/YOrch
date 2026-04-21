#pragma once

#include <tuple>

#include "concepts.hpp" // IWYU pragma: keep
#include "metadata.hpp"

namespace yorch::detail {

template <typename Derived, typename... Nodes>
struct builder_node_diagnostics {
    template <std::size_t Level, typename Task>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_task_object_argument<Task>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void node(Task&&) const& {
        static_assert(
            detail::always_false_v<Task>,
            "yorch::task_tree.node<Level>(...) received a direct-output task; use node_into<Level>(...) instead.");
    }

    template <std::size_t Level, typename Task, typename FanoutPolicy>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_task_object_argument<Task> &&
                 detail::fanout_policy<FanoutPolicy>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void node(Task&&, FanoutPolicy&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<Task, FanoutPolicy>>,
            "yorch::task_tree.node<Level>(...) received a direct-output task; use node_into<Level>(...) instead.");
    }

    template <std::size_t Level, typename F>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_callable_task_argument<F>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void node(F&&) const& {
        static_assert(
            detail::always_false_v<F>,
            "yorch::task_tree.node<Level>(...) received a direct-output callable whose last parameter is yorch::direct_out<T>; use node_into<Level>(...) instead.");
    }

    template <std::size_t Level, typename F, typename FanoutPolicyOrChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_callable_task_argument<F> &&
                 detail::fanout_policy_or_chain<FanoutPolicyOrChain>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void node(F&&, FanoutPolicyOrChain&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<F, FanoutPolicyOrChain>>,
            "yorch::task_tree.node<Level>(...) received a direct-output callable whose last parameter is yorch::direct_out<T>; use node_into<Level>(...) instead.");
    }

    template <std::size_t Level, typename F, typename FanoutPolicy, typename AdapterChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_callable_task_argument<F> &&
                 detail::fanout_policy<FanoutPolicy> &&
                 detail::adapter_chain_like<AdapterChain>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void node(F&&, FanoutPolicy&&, AdapterChain&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<F, FanoutPolicy, AdapterChain>>,
            "yorch::task_tree.node<Level>(...) received a direct-output callable whose last parameter is yorch::direct_out<T>; use node_into<Level>(...) instead.");
    }

    template <std::size_t Level, typename Task>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_task_object_argument<Task>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void node_into(Task&&) const& {
        static_assert(
            detail::always_false_v<Task>,
            "yorch::task_tree.node_into<Level>(...) received a non-direct-output task; use node<Level>(...) instead.");
    }

    template <std::size_t Level, typename Task, typename FanoutPolicy>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_task_object_argument<Task> &&
                 detail::fanout_policy<FanoutPolicy>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void node_into(Task&&, FanoutPolicy&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<Task, FanoutPolicy>>,
            "yorch::task_tree.node_into<Level>(...) received a non-direct-output task; use node<Level>(...) instead.");
    }

    template <std::size_t Level, typename F>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_callable_task_argument<F>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void node_into(F&&) const& {
        static_assert(
            detail::always_false_v<F>,
            "yorch::task_tree.node_into<Level>(...) requires a callable whose last parameter is yorch::direct_out<T>; use node<Level>(...) for ordinary tasks.");
    }

    template <std::size_t Level, typename F, typename FanoutPolicyOrChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_callable_task_argument<F> &&
                 detail::fanout_policy_or_chain<FanoutPolicyOrChain>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void node_into(F&&, FanoutPolicyOrChain&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<F, FanoutPolicyOrChain>>,
            "yorch::task_tree.node_into<Level>(...) requires a callable whose last parameter is yorch::direct_out<T>; use node<Level>(...) for ordinary tasks.");
    }

    template <std::size_t Level, typename F, typename FanoutPolicy, typename AdapterChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_callable_task_argument<F> &&
                 detail::fanout_policy<FanoutPolicy> &&
                 detail::adapter_chain_like<AdapterChain>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void node_into(F&&, FanoutPolicy&&, AdapterChain&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<F, FanoutPolicy, AdapterChain>>,
            "yorch::task_tree.node_into<Level>(...) requires a callable whose last parameter is yorch::direct_out<T>; use node<Level>(...) for ordinary tasks.");
    }

    template <std::size_t Level, typename Task>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::task_object_argument<Task>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void node_forward_prev(Task&&) const& {
        static_assert(
            detail::always_false_v<Task>,
            "yorch::task_tree.node_forward_prev<Level>(...) only accepts a callable; pass prebuilt tasks directly to node<Level>(...) instead.");
    }

    template <std::size_t Level, typename Task, typename FanoutPolicyOrChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::task_object_argument<Task> &&
                 detail::fanout_policy_or_chain<FanoutPolicyOrChain>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void node_forward_prev(Task&&, FanoutPolicyOrChain&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<Task, FanoutPolicyOrChain>>,
            "yorch::task_tree.node_forward_prev<Level>(...) only accepts a callable; pass prebuilt tasks directly to node<Level>(...) instead.");
    }

    template <std::size_t Level, typename F>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_callable_task_argument<F>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void node_forward_prev(F&&) const& {
        static_assert(
            detail::always_false_v<F>,
            "yorch::task_tree.node_forward_prev<Level>(...) does not accept callables with yorch::direct_out<T>; use node_into<Level>(...) for direct-output materialization.");
    }

    template <std::size_t Level, typename F, typename FanoutPolicyOrChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_callable_task_argument<F> &&
                 detail::fanout_policy_or_chain<FanoutPolicyOrChain>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void node_forward_prev(F&&, FanoutPolicyOrChain&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<F, FanoutPolicyOrChain>>,
            "yorch::task_tree.node_forward_prev<Level>(...) does not accept callables with yorch::direct_out<T>; use node_into<Level>(...) for direct-output materialization.");
    }

    template <std::size_t Level, typename F, typename FanoutPolicy, typename AdapterChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_callable_task_argument<F> &&
                 detail::fanout_policy<FanoutPolicy> &&
                 detail::adapter_chain_like<AdapterChain>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void node_forward_prev(F&&, FanoutPolicy&&, AdapterChain&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<F, FanoutPolicy, AdapterChain>>,
            "yorch::task_tree.node_forward_prev<Level>(...) does not accept callables with yorch::direct_out<T>; use node_into<Level>(...) for direct-output materialization.");
    }

    template <std::size_t Level, typename F>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_member_callable_task_argument<F>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void node_forward_prev(F&&) const& {
        static_assert(
            detail::always_false_v<F>,
            "yorch::task_tree.node_forward_prev<Level>(...) does not accept member function pointers; use node_forward_prev_member<Level>(...) instead.");
    }

    template <std::size_t Level, typename F>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_member_callable_task_argument<F>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void node_forward_prev(F&&) const& {
        static_assert(
            detail::always_false_v<F>,
            "yorch::task_tree.node_forward_prev<Level>(...) does not accept direct-output member functions; use node_into_member<Level>(...) instead.");
    }

    template <std::size_t Level, typename F, typename FanoutPolicyOrChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_member_callable_task_argument<F> &&
                 detail::fanout_policy_or_chain<FanoutPolicyOrChain>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void node_forward_prev(F&&, FanoutPolicyOrChain&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<F, FanoutPolicyOrChain>>,
            "yorch::task_tree.node_forward_prev<Level>(...) does not accept member function pointers; use node_forward_prev_member<Level>(...) instead.");
    }

    template <std::size_t Level, typename F, typename FanoutPolicyOrChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_member_callable_task_argument<F> &&
                 detail::fanout_policy_or_chain<FanoutPolicyOrChain>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void node_forward_prev(F&&, FanoutPolicyOrChain&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<F, FanoutPolicyOrChain>>,
            "yorch::task_tree.node_forward_prev<Level>(...) does not accept direct-output member functions; use node_into_member<Level>(...) instead.");
    }

    template <std::size_t Level, typename F>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_member_callable_task_argument<F>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void node_forward_prev_member(F&&) const& {
        static_assert(
            detail::always_false_v<F>,
            "yorch::task_tree.node_forward_prev_member<Level>(...) requires an explicit receiver binding as its second argument.");
    }

    template <std::size_t Level, typename F>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_member_callable_task_argument<F>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void node_forward_prev_member(F&&) const& {
        static_assert(
            detail::always_false_v<F>,
            "yorch::task_tree.node_forward_prev_member<Level>(...) requires an explicit receiver binding as its second argument; use node_into_member<Level>(...) for direct-output member functions.");
    }

    template <std::size_t Level, typename F, typename FanoutPolicyOrChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_member_callable_task_argument<F> &&
                 detail::fanout_policy_or_chain<FanoutPolicyOrChain>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void node_forward_prev_member(F&&, FanoutPolicyOrChain&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<F, FanoutPolicyOrChain>>,
            "yorch::task_tree.node_forward_prev_member<Level>(...) requires an explicit receiver binding as its second argument; pass fanout policies or adapters only after the receiver.");
    }

    template <std::size_t Level, typename F, typename FanoutPolicyOrChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_member_callable_task_argument<F> &&
                 detail::fanout_policy_or_chain<FanoutPolicyOrChain>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void node_forward_prev_member(F&&, FanoutPolicyOrChain&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<F, FanoutPolicyOrChain>>,
            "yorch::task_tree.node_forward_prev_member<Level>(...) requires an explicit receiver binding as its second argument; use node_into_member<Level>(...) for direct-output member functions.");
    }

    template <std::size_t Level, typename F, typename ReceiverSpec>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_member_callable_task_argument<F>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void node_forward_prev_member(F&&, ReceiverSpec&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<F, ReceiverSpec>>,
            "yorch::task_tree.node_forward_prev_member<Level>(...) received a direct-output member function whose last parameter is yorch::direct_out<T>; use node_into_member<Level>(...) instead.");
    }

    template <std::size_t Level, typename F, typename ReceiverSpec, typename FanoutPolicyOrChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_member_callable_task_argument<F> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>) &&
                 detail::fanout_policy_or_chain<FanoutPolicyOrChain>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void node_forward_prev_member(F&&, ReceiverSpec&&, FanoutPolicyOrChain&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<F, ReceiverSpec, FanoutPolicyOrChain>>,
            "yorch::task_tree.node_forward_prev_member<Level>(...) received a direct-output member function whose last parameter is yorch::direct_out<T>; use node_into_member<Level>(...) instead.");
    }

    template <std::size_t Level, typename F>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_member_callable_task_argument<F>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void node_member(F&&) const& {
        static_assert(
            detail::always_false_v<F>,
            "yorch::task_tree.node_member<Level>(...) requires an explicit receiver binding as its second argument.");
    }

    template <std::size_t Level, typename F>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_member_callable_task_argument<F>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void node_member(F&&) const& {
        static_assert(
            detail::always_false_v<F>,
            "yorch::task_tree.node_member<Level>(...) requires an explicit receiver binding as its second argument; use node_into_member<Level>(...) for direct-output member functions.");
    }

    template <std::size_t Level, typename F, typename FanoutPolicyOrChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_member_callable_task_argument<F> &&
                 detail::fanout_policy_or_chain<FanoutPolicyOrChain>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void node_member(F&&, FanoutPolicyOrChain&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<F, FanoutPolicyOrChain>>,
            "yorch::task_tree.node_member<Level>(...) requires an explicit receiver binding as its second argument; pass fanout policies or adapters only after the receiver.");
    }

    template <std::size_t Level, typename F, typename FanoutPolicyOrChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_member_callable_task_argument<F> &&
                 detail::fanout_policy_or_chain<FanoutPolicyOrChain>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void node_member(F&&, FanoutPolicyOrChain&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<F, FanoutPolicyOrChain>>,
            "yorch::task_tree.node_member<Level>(...) requires an explicit receiver binding as its second argument; use node_into_member<Level>(...) for direct-output member functions.");
    }

    template <std::size_t Level, typename F, typename ReceiverSpec>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_member_callable_task_argument<F>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void node_member(F&&, ReceiverSpec&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<F, ReceiverSpec>>,
            "yorch::task_tree.node_member<Level>(...) received a direct-output member function whose last parameter is yorch::direct_out<T>; use node_into_member<Level>(...) instead.");
    }

    template <std::size_t Level, typename F, typename ReceiverSpec, typename FanoutPolicyOrChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_member_callable_task_argument<F> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>) &&
                 detail::fanout_policy_or_chain<FanoutPolicyOrChain>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void node_member(F&&, ReceiverSpec&&, FanoutPolicyOrChain&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<F, ReceiverSpec, FanoutPolicyOrChain>>,
            "yorch::task_tree.node_member<Level>(...) received a direct-output member function whose last parameter is yorch::direct_out<T>; use node_into_member<Level>(...) instead.");
    }

    template <std::size_t Level, typename F>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_member_callable_task_argument<F>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void node_into_member(F&&) const& {
        static_assert(
            detail::always_false_v<F>,
            "yorch::task_tree.node_into_member<Level>(...) requires an explicit receiver binding as its second argument.");
    }

    template <std::size_t Level, typename F>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_member_callable_task_argument<F>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void node_into_member(F&&) const& {
        static_assert(
            detail::always_false_v<F>,
            "yorch::task_tree.node_into_member<Level>(...) requires an explicit receiver binding as its second argument; use node_member<Level>(...) for ordinary member functions.");
    }

    template <std::size_t Level, typename F, typename FanoutPolicyOrChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_member_callable_task_argument<F> &&
                 detail::fanout_policy_or_chain<FanoutPolicyOrChain>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void node_into_member(F&&, FanoutPolicyOrChain&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<F, FanoutPolicyOrChain>>,
            "yorch::task_tree.node_into_member<Level>(...) requires an explicit receiver binding as its second argument; pass fanout policies or adapters only after the receiver.");
    }

    template <std::size_t Level, typename F, typename FanoutPolicyOrChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_member_callable_task_argument<F> &&
                 detail::fanout_policy_or_chain<FanoutPolicyOrChain>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void node_into_member(F&&, FanoutPolicyOrChain&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<F, FanoutPolicyOrChain>>,
            "yorch::task_tree.node_into_member<Level>(...) requires an explicit receiver binding as its second argument; use node_member<Level>(...) for ordinary member functions.");
    }

    template <std::size_t Level, typename F, typename ReceiverSpec>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_member_callable_task_argument<F>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void node_into_member(F&&, ReceiverSpec&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<F, ReceiverSpec>>,
            "yorch::task_tree.node_into_member<Level>(...) requires a member function whose last parameter is yorch::direct_out<T>; use node_member<Level>(...) for ordinary member functions.");
    }

    template <std::size_t Level, typename F, typename ReceiverSpec, typename FanoutPolicyOrChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_member_callable_task_argument<F> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>) &&
                 detail::fanout_policy_or_chain<FanoutPolicyOrChain>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void node_into_member(F&&, ReceiverSpec&&, FanoutPolicyOrChain&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<F, ReceiverSpec, FanoutPolicyOrChain>>,
            "yorch::task_tree.node_into_member<Level>(...) requires a member function whose last parameter is yorch::direct_out<T>; use node_member<Level>(...) for ordinary member functions.");
    }
};

} // namespace yorch::detail
