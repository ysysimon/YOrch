#pragma once

#include <tuple>
#include <type_traits>
#include <utility>

#include "../detail/bind/traits.hpp"
#include "adapters.hpp"
#include "tasks.hpp"

namespace yorch {

template <typename F, typename AdapterChain>
struct task_into_binder;

template <typename F, typename AdapterChain>
struct task_binder {
    F func;
    AdapterChain adapter_specs;

    template <typename... Specs>
        requires detail::bind_signature_matches<F, Specs...>
    constexpr auto operator()(Specs&&... specs) const& {
        return yorch::apply_adapters(
            yorch::bind(func, std::forward<Specs>(specs)...),
            adapter_specs);
    }

    template <typename... Specs>
        requires detail::bind_signature_matches<F, Specs...>
    constexpr auto operator()(Specs&&... specs) && {
        return yorch::apply_adapters(
            yorch::bind(std::move(func), std::forward<Specs>(specs)...),
            std::move(adapter_specs));
    }
};

template <typename F, typename AdapterChain>
struct task_into_binder {
    F func;
    AdapterChain adapter_specs;
    using output_type =
        detail::direct_out_payload_t<detail::last_arg_t<F>>;

    template <typename... Specs>
        requires detail::inferred_bind_into_signature_matches<F, Specs...>
    constexpr auto operator()(Specs&&... specs) const& {
        return yorch::apply_adapters(
            yorch::bind_into<output_type>(func, std::forward<Specs>(specs)...),
            adapter_specs);
    }

    template <typename... Specs>
        requires detail::inferred_bind_into_signature_matches<F, Specs...>
    constexpr auto operator()(Specs&&... specs) && {
        return yorch::apply_adapters(
            yorch::bind_into<output_type>(std::move(func), std::forward<Specs>(specs)...),
            std::move(adapter_specs));
    }
};

template <typename F>
constexpr auto task(F&& f)
    requires detail::ordinary_bind_callable<F> {
    return task_binder<std::decay_t<F>, adapter_chain<>> {
        std::forward<F>(f),
        {}
    };
}

template <typename F, typename AdapterChain>
constexpr auto task(F&& f, AdapterChain&& adapter_specs)
    requires detail::ordinary_bind_callable<F> &&
             detail::adapter_chain_like<AdapterChain> {
    return task_binder<
        std::decay_t<F>,
        std::decay_t<AdapterChain>
    > {
        std::forward<F>(f),
        std::forward<AdapterChain>(adapter_specs)
    };
}

template <typename F>
constexpr auto task_into(F&& f)
    requires detail::inferable_direct_output_callable<F> {
    return task_into_binder<std::decay_t<F>, adapter_chain<>> {
        std::forward<F>(f),
        {}
    };
}

template <typename F, typename AdapterChain>
constexpr auto task_into(F&& f, AdapterChain&& adapter_specs)
    requires detail::inferable_direct_output_callable<F> &&
             detail::adapter_chain_like<AdapterChain> {
    return task_into_binder<
        std::decay_t<F>,
        std::decay_t<AdapterChain>
    > {
        std::forward<F>(f),
        std::forward<AdapterChain>(adapter_specs)
    };
}

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
             detail::has_declared_task_output_v<Task> {
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
