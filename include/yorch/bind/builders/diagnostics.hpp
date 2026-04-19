#pragma once

#include <tuple>
#include "core.hpp" // IWYU pragma: keep

namespace yorch {

template <typename F>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
constexpr void task(F&&)
    requires detail::inferable_direct_output_callable<F> {
    static_assert(
        detail::always_false_v<F>,
        "yorch::task(...) received a direct-output callable whose last parameter is yorch::direct_out<T>; use yorch::task_into(...) instead.");
}

template <typename F>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
constexpr void task(F&&)
    requires detail::member_bind_callable<F> {
    static_assert(
        detail::always_false_v<F>,
        "yorch::task(...) does not accept member function pointers; use bind_member(...) first.");
}

template <typename Task>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
constexpr void task(Task&&)
    requires detail::bind_task_object<Task> &&
             detail::task_uses_direct_output_protocol_v<Task> {
    static_assert(
        detail::always_false_v<Task>,
        "yorch::task(...) does not accept prebuilt direct-output tasks; pass them directly to root_into(...) or node_into<Level>(...) instead.");
}

template <typename F, typename AdapterChain>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
constexpr void task(F&&, AdapterChain&&)
    requires detail::inferable_direct_output_callable<F> &&
             detail::adapter_chain_like<AdapterChain> {
    static_assert(
        detail::always_false_v<std::tuple<F, AdapterChain>>,
        "yorch::task(...) received a direct-output callable whose last parameter is yorch::direct_out<T>; use yorch::task_into(...) instead.");
}

template <typename F, typename AdapterChain>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
constexpr void task(F&&, AdapterChain&&)
    requires detail::member_bind_callable<F> &&
             detail::adapter_chain_like<AdapterChain> {
    static_assert(
        detail::always_false_v<std::tuple<F, AdapterChain>>,
        "yorch::task(...) does not accept member function pointers; use bind_member(...) first.");
}

template <typename Task>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
constexpr void task_into(Task&&)
    requires detail::bind_task_object<Task> {
    static_assert(
        detail::always_false_v<Task>,
        "yorch::task_into(...) only accepts a callable whose last parameter is yorch::direct_out<T>; pass prebuilt tasks directly to root/node instead.");
}

template <typename Task, typename AdapterChain>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
constexpr void task_into(Task&&, AdapterChain&&)
    requires detail::bind_task_object<Task> &&
             detail::adapter_chain_like<AdapterChain> {
    static_assert(
        detail::always_false_v<std::tuple<Task, AdapterChain>>,
        "yorch::task_into(...) only accepts a callable whose last parameter is yorch::direct_out<T>; pass prebuilt tasks directly to root/node instead.");
}

template <typename F>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
constexpr void task_into(F&&)
    requires detail::ordinary_bind_callable<F> {
    static_assert(
        detail::always_false_v<F>,
        "yorch::task_into(...) requires a callable whose last parameter is yorch::direct_out<T>; use yorch::task(...) for ordinary tasks.");
}

template <typename Task>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
constexpr void task_forward_prev(Task&&)
    requires detail::bind_task_object<Task> {
    static_assert(
        detail::always_false_v<Task>,
        "yorch::task_forward_prev(...) only accepts a callable; pass prebuilt tasks directly to root/node instead.");
}

template <typename Task, typename AdapterChain>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
constexpr void task_forward_prev(Task&&, AdapterChain&&)
    requires detail::bind_task_object<Task> &&
             detail::adapter_chain_like<AdapterChain> {
    static_assert(
        detail::always_false_v<std::tuple<Task, AdapterChain>>,
        "yorch::task_forward_prev(...) only accepts a callable; pass prebuilt tasks directly to root/node instead.");
}

template <typename F>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
constexpr void task_forward_prev(F&&)
    requires detail::inferable_direct_output_callable<F> {
    static_assert(
        detail::always_false_v<F>,
        "yorch::task_forward_prev(...) does not accept callables with yorch::direct_out<T>; use yorch::task_into(...) for direct-output materialization.");
}

template <typename F, typename AdapterChain>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
constexpr void task_forward_prev(F&&, AdapterChain&&)
    requires detail::inferable_direct_output_callable<F> &&
             detail::adapter_chain_like<AdapterChain> {
    static_assert(
        detail::always_false_v<std::tuple<F, AdapterChain>>,
        "yorch::task_forward_prev(...) does not accept callables with yorch::direct_out<T>; use yorch::task_into(...) for direct-output materialization.");
}

template <typename F>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
constexpr void task_forward_prev(F&&)
    requires detail::member_bind_callable<F> {
    static_assert(
        detail::always_false_v<F>,
        "yorch::task_forward_prev(...) does not support member function pointers in v1.");
}

template <typename F, typename AdapterChain>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
constexpr void task_forward_prev(F&&, AdapterChain&&)
    requires detail::member_bind_callable<F> &&
             detail::adapter_chain_like<AdapterChain> {
    static_assert(
        detail::always_false_v<std::tuple<F, AdapterChain>>,
        "yorch::task_forward_prev(...) does not support member function pointers in v1.");
}

template <typename F>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
constexpr void task_member(F&&)
    requires detail::ordinary_member_bind_callable<F> {
    static_assert(
        detail::always_false_v<F>,
        "yorch::task_member(...) requires an explicit receiver binding as its second argument.");
}

template <typename F, typename AdapterChain>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
constexpr void task_member(F&&, AdapterChain&&)
    requires detail::ordinary_member_bind_callable<F> &&
             detail::adapter_chain_like<AdapterChain> {
    static_assert(
        detail::always_false_v<std::tuple<F, AdapterChain>>,
        "yorch::task_member(...) requires an explicit receiver binding as its second argument; pass adapters(...) as the third argument.");
}

template <typename F>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
constexpr void task_member(F&&)
    requires detail::inferable_direct_output_member_callable<F> {
    static_assert(
        detail::always_false_v<F>,
        "yorch::task_member(...) requires an explicit receiver binding as its second argument; use yorch::task_into_member(...) for direct-output member functions.");
}

template <typename F, typename AdapterChain>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
constexpr void task_member(F&&, AdapterChain&&)
    requires detail::inferable_direct_output_member_callable<F> &&
             detail::adapter_chain_like<AdapterChain> {
    static_assert(
        detail::always_false_v<std::tuple<F, AdapterChain>>,
        "yorch::task_member(...) requires an explicit receiver binding as its second argument; use yorch::task_into_member(...) for direct-output member functions.");
}

template <typename F, typename ReceiverSpec>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
constexpr void task_member(F&&, ReceiverSpec&&)
    requires detail::inferable_direct_output_member_callable<F> {
    static_assert(
        detail::always_false_v<std::tuple<F, ReceiverSpec>>,
        "yorch::task_member(...) received a direct-output member function whose last parameter is yorch::direct_out<T>; use yorch::task_into_member(...) instead.");
}

template <typename F, typename ReceiverSpec, typename AdapterChain>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
constexpr void task_member(F&&, ReceiverSpec&&, AdapterChain&&)
    requires detail::inferable_direct_output_member_callable<F> &&
             detail::adapter_chain_like<AdapterChain> {
    static_assert(
        detail::always_false_v<std::tuple<F, ReceiverSpec, AdapterChain>>,
        "yorch::task_member(...) received a direct-output member function whose last parameter is yorch::direct_out<T>; use yorch::task_into_member(...) instead.");
}

template <typename F>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
constexpr void task_into_member(F&&)
    requires detail::inferable_direct_output_member_callable<F> {
    static_assert(
        detail::always_false_v<F>,
        "yorch::task_into_member(...) requires an explicit receiver binding as its second argument.");
}

template <typename F, typename AdapterChain>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
constexpr void task_into_member(F&&, AdapterChain&&)
    requires detail::inferable_direct_output_member_callable<F> &&
             detail::adapter_chain_like<AdapterChain> {
    static_assert(
        detail::always_false_v<std::tuple<F, AdapterChain>>,
        "yorch::task_into_member(...) requires an explicit receiver binding as its second argument; pass adapters(...) as the third argument.");
}

template <typename F>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
constexpr void task_into_member(F&&)
    requires detail::ordinary_member_bind_callable<F> {
    static_assert(
        detail::always_false_v<F>,
        "yorch::task_into_member(...) requires an explicit receiver binding as its second argument; use yorch::task_member(...) for ordinary member functions.");
}

template <typename F, typename AdapterChain>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
constexpr void task_into_member(F&&, AdapterChain&&)
    requires detail::ordinary_member_bind_callable<F> &&
             detail::adapter_chain_like<AdapterChain> {
    static_assert(
        detail::always_false_v<std::tuple<F, AdapterChain>>,
        "yorch::task_into_member(...) requires an explicit receiver binding as its second argument; use yorch::task_member(...) for ordinary member functions.");
}

template <typename F, typename ReceiverSpec>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
constexpr void task_into_member(F&&, ReceiverSpec&&)
    requires detail::ordinary_member_bind_callable<F> {
    static_assert(
        detail::always_false_v<std::tuple<F, ReceiverSpec>>,
        "yorch::task_into_member(...) requires a member function whose last parameter is yorch::direct_out<T>; use yorch::task_member(...) for ordinary member functions.");
}

template <typename F, typename ReceiverSpec, typename AdapterChain>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
constexpr void task_into_member(F&&, ReceiverSpec&&, AdapterChain&&)
    requires detail::ordinary_member_bind_callable<F> &&
             detail::adapter_chain_like<AdapterChain> {
    static_assert(
        detail::always_false_v<std::tuple<F, ReceiverSpec, AdapterChain>>,
        "yorch::task_into_member(...) requires a member function whose last parameter is yorch::direct_out<T>; use yorch::task_member(...) for ordinary member functions.");
}

template <typename F>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
constexpr void task_into(F&&)
    requires detail::member_bind_callable<F> {
    static_assert(
        detail::always_false_v<F>,
        "yorch::task_into(...) does not accept member function pointers; use bind_into_member(...) first.");
}

template <typename F, typename AdapterChain>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
constexpr void task_into(F&&, AdapterChain&&)
    requires detail::ordinary_bind_callable<F> &&
             detail::adapter_chain_like<AdapterChain> {
    static_assert(
        detail::always_false_v<std::tuple<F, AdapterChain>>,
        "yorch::task_into(...) requires a callable whose last parameter is yorch::direct_out<T>; use yorch::task(...) for ordinary tasks.");
}

template <typename F, typename AdapterChain>
// NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
constexpr void task_into(F&&, AdapterChain&&)
    requires detail::member_bind_callable<F> &&
             detail::adapter_chain_like<AdapterChain> {
    static_assert(
        detail::always_false_v<std::tuple<F, AdapterChain>>,
        "yorch::task_into(...) does not accept member function pointers; use bind_into_member(...) first.");
}

} // namespace yorch
