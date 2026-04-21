#pragma once

#include <tuple>

#include "concepts.hpp" // IWYU pragma: keep

namespace yorch::detail {

template <typename Derived, typename... Nodes>
struct builder_root_diagnostics {
    template <typename Task>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_task_object_argument<Task>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void root(Task&&) const& {
        static_assert(
            detail::always_false_v<Task>,
            "yorch::task_tree.root(...) received a direct-output task; use root_into(...) instead.");
    }

    template <typename Task, typename FanoutPolicy>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_task_object_argument<Task> &&
                 detail::fanout_policy<FanoutPolicy>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void root(Task&&, FanoutPolicy&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<Task, FanoutPolicy>>,
            "yorch::task_tree.root(...) received a direct-output task; use root_into(...) instead.");
    }

    template <typename F>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_callable_task_argument<F>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void root(F&&) const& {
        static_assert(
            detail::always_false_v<F>,
            "yorch::task_tree.root(...) received a direct-output callable whose last parameter is yorch::direct_out<T>; use root_into(...) instead.");
    }

    template <typename F, typename FanoutPolicyOrChain>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_callable_task_argument<F> &&
                 detail::fanout_policy_or_chain<FanoutPolicyOrChain>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void root(F&&, FanoutPolicyOrChain&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<F, FanoutPolicyOrChain>>,
            "yorch::task_tree.root(...) received a direct-output callable whose last parameter is yorch::direct_out<T>; use root_into(...) instead.");
    }

    template <typename F, typename FanoutPolicy, typename AdapterChain>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_callable_task_argument<F> &&
                 detail::fanout_policy<FanoutPolicy> &&
                 detail::adapter_chain_like<AdapterChain>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void root(F&&, FanoutPolicy&&, AdapterChain&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<F, FanoutPolicy, AdapterChain>>,
            "yorch::task_tree.root(...) received a direct-output callable whose last parameter is yorch::direct_out<T>; use root_into(...) instead.");
    }

    template <typename Task>
        requires (sizeof...(Nodes) == 0) &&
                 detail::ordinary_task_object_argument<Task>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void root_into(Task&&) const& {
        static_assert(
            detail::always_false_v<Task>,
            "yorch::task_tree.root_into(...) received a non-direct-output task; use root(...) instead.");
    }

    template <typename Task, typename FanoutPolicy>
        requires (sizeof...(Nodes) == 0) &&
                 detail::ordinary_task_object_argument<Task> &&
                 detail::fanout_policy<FanoutPolicy>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void root_into(Task&&, FanoutPolicy&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<Task, FanoutPolicy>>,
            "yorch::task_tree.root_into(...) received a non-direct-output task; use root(...) instead.");
    }

    template <typename F>
        requires (sizeof...(Nodes) == 0) &&
                 detail::ordinary_callable_task_argument<F>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void root_into(F&&) const& {
        static_assert(
            detail::always_false_v<F>,
            "yorch::task_tree.root_into(...) requires a callable whose last parameter is yorch::direct_out<T>; use root(...) for ordinary tasks.");
    }

    template <typename F, typename FanoutPolicyOrChain>
        requires (sizeof...(Nodes) == 0) &&
                 detail::ordinary_callable_task_argument<F> &&
                 detail::fanout_policy_or_chain<FanoutPolicyOrChain>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void root_into(F&&, FanoutPolicyOrChain&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<F, FanoutPolicyOrChain>>,
            "yorch::task_tree.root_into(...) requires a callable whose last parameter is yorch::direct_out<T>; use root(...) for ordinary tasks.");
    }

    template <typename F, typename FanoutPolicy, typename AdapterChain>
        requires (sizeof...(Nodes) == 0) &&
                 detail::ordinary_callable_task_argument<F> &&
                 detail::fanout_policy<FanoutPolicy> &&
                 detail::adapter_chain_like<AdapterChain>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void root_into(F&&, FanoutPolicy&&, AdapterChain&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<F, FanoutPolicy, AdapterChain>>,
            "yorch::task_tree.root_into(...) requires a callable whose last parameter is yorch::direct_out<T>; use root(...) for ordinary tasks.");
    }

    template <typename Task>
        requires (sizeof...(Nodes) == 0) &&
                 detail::task_object_argument<Task>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void root_forward_prev(Task&&) const& {
        static_assert(
            detail::always_false_v<Task>,
            "yorch::task_tree.root_forward_prev(...) only accepts a callable; pass prebuilt tasks directly to root/node instead.");
    }

    template <typename Task, typename FanoutPolicyOrChain>
        requires (sizeof...(Nodes) == 0) &&
                 detail::task_object_argument<Task> &&
                 detail::fanout_policy_or_chain<FanoutPolicyOrChain>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void root_forward_prev(Task&&, FanoutPolicyOrChain&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<Task, FanoutPolicyOrChain>>,
            "yorch::task_tree.root_forward_prev(...) only accepts a callable; pass prebuilt tasks directly to root/node instead.");
    }

    template <typename F>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_callable_task_argument<F>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void root_forward_prev(F&&) const& {
        static_assert(
            detail::always_false_v<F>,
            "yorch::task_tree.root_forward_prev(...) does not accept callables with yorch::direct_out<T>; use root_into(...) for direct-output materialization.");
    }

    template <typename F, typename FanoutPolicyOrChain>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_callable_task_argument<F> &&
                 detail::fanout_policy_or_chain<FanoutPolicyOrChain>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void root_forward_prev(F&&, FanoutPolicyOrChain&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<F, FanoutPolicyOrChain>>,
            "yorch::task_tree.root_forward_prev(...) does not accept callables with yorch::direct_out<T>; use root_into(...) for direct-output materialization.");
    }

    template <typename F, typename FanoutPolicy, typename AdapterChain>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_callable_task_argument<F> &&
                 detail::fanout_policy<FanoutPolicy> &&
                 detail::adapter_chain_like<AdapterChain>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void root_forward_prev(F&&, FanoutPolicy&&, AdapterChain&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<F, FanoutPolicy, AdapterChain>>,
            "yorch::task_tree.root_forward_prev(...) does not accept callables with yorch::direct_out<T>; use root_into(...) for direct-output materialization.");
    }

    template <typename F>
        requires (sizeof...(Nodes) == 0) &&
                 detail::ordinary_member_callable_task_argument<F>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void root_forward_prev(F&&) const& {
        static_assert(
            detail::always_false_v<F>,
            "yorch::task_tree.root_forward_prev(...) does not accept member function pointers; use root_forward_prev_member(...) instead.");
    }

    template <typename F>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_member_callable_task_argument<F>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void root_forward_prev(F&&) const& {
        static_assert(
            detail::always_false_v<F>,
            "yorch::task_tree.root_forward_prev(...) does not accept direct-output member functions; use root_into_member(...) instead.");
    }

    template <typename F, typename FanoutPolicyOrChain>
        requires (sizeof...(Nodes) == 0) &&
                 detail::ordinary_member_callable_task_argument<F> &&
                 detail::fanout_policy_or_chain<FanoutPolicyOrChain>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void root_forward_prev(F&&, FanoutPolicyOrChain&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<F, FanoutPolicyOrChain>>,
            "yorch::task_tree.root_forward_prev(...) does not accept member function pointers; use root_forward_prev_member(...) instead.");
    }

    template <typename F, typename FanoutPolicyOrChain>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_member_callable_task_argument<F> &&
                 detail::fanout_policy_or_chain<FanoutPolicyOrChain>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void root_forward_prev(F&&, FanoutPolicyOrChain&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<F, FanoutPolicyOrChain>>,
            "yorch::task_tree.root_forward_prev(...) does not accept direct-output member functions; use root_into_member(...) instead.");
    }

    template <typename F>
        requires (sizeof...(Nodes) == 0) &&
                 detail::ordinary_member_callable_task_argument<F>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void root_forward_prev_member(F&&) const& {
        static_assert(
            detail::always_false_v<F>,
            "yorch::task_tree.root_forward_prev_member(...) requires an explicit receiver binding as its second argument.");
    }

    template <typename F>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_member_callable_task_argument<F>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void root_forward_prev_member(F&&) const& {
        static_assert(
            detail::always_false_v<F>,
            "yorch::task_tree.root_forward_prev_member(...) requires an explicit receiver binding as its second argument; use root_into_member(...) for direct-output member functions.");
    }

    template <typename F, typename FanoutPolicyOrChain>
        requires (sizeof...(Nodes) == 0) &&
                 detail::ordinary_member_callable_task_argument<F> &&
                 detail::fanout_policy_or_chain<FanoutPolicyOrChain>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void root_forward_prev_member(F&&, FanoutPolicyOrChain&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<F, FanoutPolicyOrChain>>,
            "yorch::task_tree.root_forward_prev_member(...) requires an explicit receiver binding as its second argument; pass fanout policies or adapters only after the receiver.");
    }

    template <typename F, typename FanoutPolicyOrChain>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_member_callable_task_argument<F> &&
                 detail::fanout_policy_or_chain<FanoutPolicyOrChain>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void root_forward_prev_member(F&&, FanoutPolicyOrChain&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<F, FanoutPolicyOrChain>>,
            "yorch::task_tree.root_forward_prev_member(...) requires an explicit receiver binding as its second argument; use root_into_member(...) for direct-output member functions.");
    }

    template <typename F, typename ReceiverSpec>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_member_callable_task_argument<F>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void root_forward_prev_member(F&&, ReceiverSpec&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<F, ReceiverSpec>>,
            "yorch::task_tree.root_forward_prev_member(...) received a direct-output member function whose last parameter is yorch::direct_out<T>; use root_into_member(...) instead.");
    }

    template <typename F, typename ReceiverSpec, typename FanoutPolicyOrChain>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_member_callable_task_argument<F> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>) &&
                 detail::fanout_policy_or_chain<FanoutPolicyOrChain>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void root_forward_prev_member(F&&, ReceiverSpec&&, FanoutPolicyOrChain&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<F, ReceiverSpec, FanoutPolicyOrChain>>,
            "yorch::task_tree.root_forward_prev_member(...) received a direct-output member function whose last parameter is yorch::direct_out<T>; use root_into_member(...) instead.");
    }

    template <typename F>
        requires (sizeof...(Nodes) == 0) &&
                 detail::ordinary_member_callable_task_argument<F>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void root_member(F&&) const& {
        static_assert(
            detail::always_false_v<F>,
            "yorch::task_tree.root_member(...) requires an explicit receiver binding as its second argument.");
    }

    template <typename F>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_member_callable_task_argument<F>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void root_member(F&&) const& {
        static_assert(
            detail::always_false_v<F>,
            "yorch::task_tree.root_member(...) requires an explicit receiver binding as its second argument; use root_into_member(...) for direct-output member functions.");
    }

    template <typename F, typename FanoutPolicyOrChain>
        requires (sizeof...(Nodes) == 0) &&
                 detail::ordinary_member_callable_task_argument<F> &&
                 detail::fanout_policy_or_chain<FanoutPolicyOrChain>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void root_member(F&&, FanoutPolicyOrChain&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<F, FanoutPolicyOrChain>>,
            "yorch::task_tree.root_member(...) requires an explicit receiver binding as its second argument; pass fanout policies or adapters only after the receiver.");
    }

    template <typename F, typename FanoutPolicyOrChain>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_member_callable_task_argument<F> &&
                 detail::fanout_policy_or_chain<FanoutPolicyOrChain>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void root_member(F&&, FanoutPolicyOrChain&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<F, FanoutPolicyOrChain>>,
            "yorch::task_tree.root_member(...) requires an explicit receiver binding as its second argument; use root_into_member(...) for direct-output member functions.");
    }

    template <typename F, typename ReceiverSpec>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_member_callable_task_argument<F>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void root_member(F&&, ReceiverSpec&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<F, ReceiverSpec>>,
            "yorch::task_tree.root_member(...) received a direct-output member function whose last parameter is yorch::direct_out<T>; use root_into_member(...) instead.");
    }

    template <typename F, typename ReceiverSpec, typename FanoutPolicyOrChain>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_member_callable_task_argument<F> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>) &&
                 detail::fanout_policy_or_chain<FanoutPolicyOrChain>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void root_member(F&&, ReceiverSpec&&, FanoutPolicyOrChain&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<F, ReceiverSpec, FanoutPolicyOrChain>>,
            "yorch::task_tree.root_member(...) received a direct-output member function whose last parameter is yorch::direct_out<T>; use root_into_member(...) instead.");
    }

    template <typename F>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_member_callable_task_argument<F>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void root_into_member(F&&) const& {
        static_assert(
            detail::always_false_v<F>,
            "yorch::task_tree.root_into_member(...) requires an explicit receiver binding as its second argument.");
    }

    template <typename F>
        requires (sizeof...(Nodes) == 0) &&
                 detail::ordinary_member_callable_task_argument<F>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void root_into_member(F&&) const& {
        static_assert(
            detail::always_false_v<F>,
            "yorch::task_tree.root_into_member(...) requires an explicit receiver binding as its second argument; use root_member(...) for ordinary member functions.");
    }

    template <typename F, typename FanoutPolicyOrChain>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_member_callable_task_argument<F> &&
                 detail::fanout_policy_or_chain<FanoutPolicyOrChain>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void root_into_member(F&&, FanoutPolicyOrChain&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<F, FanoutPolicyOrChain>>,
            "yorch::task_tree.root_into_member(...) requires an explicit receiver binding as its second argument; pass fanout policies or adapters only after the receiver.");
    }

    template <typename F, typename FanoutPolicyOrChain>
        requires (sizeof...(Nodes) == 0) &&
                 detail::ordinary_member_callable_task_argument<F> &&
                 detail::fanout_policy_or_chain<FanoutPolicyOrChain>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void root_into_member(F&&, FanoutPolicyOrChain&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<F, FanoutPolicyOrChain>>,
            "yorch::task_tree.root_into_member(...) requires an explicit receiver binding as its second argument; use root_member(...) for ordinary member functions.");
    }

    template <typename F, typename ReceiverSpec>
        requires (sizeof...(Nodes) == 0) &&
                 detail::ordinary_member_callable_task_argument<F>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void root_into_member(F&&, ReceiverSpec&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<F, ReceiverSpec>>,
            "yorch::task_tree.root_into_member(...) requires a member function whose last parameter is yorch::direct_out<T>; use root_member(...) for ordinary member functions.");
    }

    template <typename F, typename ReceiverSpec, typename FanoutPolicyOrChain>
        requires (sizeof...(Nodes) == 0) &&
                 detail::ordinary_member_callable_task_argument<F> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>) &&
                 detail::fanout_policy_or_chain<FanoutPolicyOrChain>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    constexpr void root_into_member(F&&, ReceiverSpec&&, FanoutPolicyOrChain&&) const& {
        static_assert(
            detail::always_false_v<std::tuple<F, ReceiverSpec, FanoutPolicyOrChain>>,
            "yorch::task_tree.root_into_member(...) requires a member function whose last parameter is yorch::direct_out<T>; use root_member(...) for ordinary member functions.");
    }
};

} // namespace yorch::detail
