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
struct task_forward_prev_binder;

template <typename F, typename ReceiverSpec, typename AdapterChain>
struct task_member_receiver_binder;

template <typename F, typename ReceiverSpec, typename AdapterChain>
struct task_into_member_receiver_binder;

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

template <typename F, typename AdapterChain>
struct task_forward_prev_binder {
    F func;
    AdapterChain adapter_specs;

    template <typename... Specs>
        requires detail::inferred_forward_prev_signature_matches<F, Specs...>
    constexpr auto operator()(Specs&&... specs) const& {
        using output_type = detail::forward_prev_unique_prev_payload_t<std::decay_t<Specs>...>;

        static_assert(!std::is_void_v<output_type>,
                      "yorch::task_forward_prev(...) requires exactly one prev-access binding");

        return yorch::apply_adapters(
            yorch::bind_forward_prev<output_type>(func, std::forward<Specs>(specs)...),
            adapter_specs);
    }

    template <typename... Specs>
        requires detail::inferred_forward_prev_signature_matches<F, Specs...>
    constexpr auto operator()(Specs&&... specs) && {
        using output_type = detail::forward_prev_unique_prev_payload_t<std::decay_t<Specs>...>;

        static_assert(!std::is_void_v<output_type>,
                      "yorch::task_forward_prev(...) requires exactly one prev-access binding");

        return yorch::apply_adapters(
            yorch::bind_forward_prev<output_type>(std::move(func), std::forward<Specs>(specs)...),
            std::move(adapter_specs));
    }
};

template <typename F, typename ReceiverSpec, typename AdapterChain>
struct task_member_receiver_binder {
    F func;
    ReceiverSpec receiver_spec;
    AdapterChain adapter_specs;

    template <typename... Specs>
        requires detail::member_bound_signature_matches<F, Specs...>
    constexpr auto operator()(Specs&&... specs) const& {
        return yorch::apply_adapters(
            yorch::bind_member(func, receiver_spec, std::forward<Specs>(specs)...),
            adapter_specs);
    }

    template <typename... Specs>
        requires detail::member_bound_signature_matches<F, Specs...>
    constexpr auto operator()(Specs&&... specs) && {
        return yorch::apply_adapters(
            yorch::bind_member(std::move(func), std::move(receiver_spec), std::forward<Specs>(specs)...),
            std::move(adapter_specs));
    }
};

template <typename F, typename ReceiverSpec, typename AdapterChain>
struct task_into_member_receiver_binder {
    F func;
    ReceiverSpec receiver_spec;
    AdapterChain adapter_specs;
    using output_type =
        detail::direct_out_payload_t<detail::member_last_arg_t<F>>;

    template <typename... Specs>
        requires detail::inferred_member_bound_into_signature_matches<F, Specs...>
    constexpr auto operator()(Specs&&... specs) const& {
        return yorch::apply_adapters(
            yorch::bind_into_member<output_type>(
                func,
                receiver_spec,
                std::forward<Specs>(specs)...),
            adapter_specs);
    }

    template <typename... Specs>
        requires detail::inferred_member_bound_into_signature_matches<F, Specs...>
    constexpr auto operator()(Specs&&... specs) && {
        return yorch::apply_adapters(
            yorch::bind_into_member<output_type>(
                std::move(func),
                std::move(receiver_spec),
                std::forward<Specs>(specs)...),
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

template <typename F, typename ReceiverSpec>
constexpr auto task_member(F&& f, ReceiverSpec&& receiver_spec)
    requires detail::ordinary_member_bind_callable<F> &&
             detail::member_receiver_bindable<F, ReceiverSpec> &&
             (!detail::adapter_chain_like<ReceiverSpec>) {
    return task_member_receiver_binder<
        std::decay_t<F>,
        std::decay_t<ReceiverSpec>,
        adapter_chain<>
    > {
        std::forward<F>(f),
        std::forward<ReceiverSpec>(receiver_spec),
        {}
    };
}

template <typename F, typename ReceiverSpec, typename AdapterChain>
constexpr auto task_member(F&& f, ReceiverSpec&& receiver_spec, AdapterChain&& adapter_specs)
    requires detail::ordinary_member_bind_callable<F> &&
             detail::member_receiver_bindable<F, ReceiverSpec> &&
             (!detail::adapter_chain_like<ReceiverSpec>) &&
             detail::adapter_chain_like<AdapterChain> {
    return task_member_receiver_binder<
        std::decay_t<F>,
        std::decay_t<ReceiverSpec>,
        std::decay_t<AdapterChain>
    > {
        std::forward<F>(f),
        std::forward<ReceiverSpec>(receiver_spec),
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
constexpr auto task_forward_prev(F&& f)
    requires detail::inferable_forward_prev_callable<F> {
    return task_forward_prev_binder<std::decay_t<F>, adapter_chain<>> {
        std::forward<F>(f),
        {}
    };
}

template <typename F, typename AdapterChain>
constexpr auto task_forward_prev(F&& f, AdapterChain&& adapter_specs)
    requires detail::inferable_forward_prev_callable<F> &&
             detail::adapter_chain_like<AdapterChain> {
    return task_forward_prev_binder<
        std::decay_t<F>,
        std::decay_t<AdapterChain>
    > {
        std::forward<F>(f),
        std::forward<AdapterChain>(adapter_specs)
    };
}

template <typename F, typename ReceiverSpec>
constexpr auto task_into_member(F&& f, ReceiverSpec&& receiver_spec)
    requires detail::inferable_direct_output_member_callable<F> &&
             detail::member_receiver_bindable<F, ReceiverSpec> &&
             (!detail::adapter_chain_like<ReceiverSpec>) {
    return task_into_member_receiver_binder<
        std::decay_t<F>,
        std::decay_t<ReceiverSpec>,
        adapter_chain<>
    > {
        std::forward<F>(f),
        std::forward<ReceiverSpec>(receiver_spec),
        {}
    };
}

template <typename F, typename ReceiverSpec, typename AdapterChain>
constexpr auto task_into_member(F&& f, ReceiverSpec&& receiver_spec, AdapterChain&& adapter_specs)
    requires detail::inferable_direct_output_member_callable<F> &&
             detail::member_receiver_bindable<F, ReceiverSpec> &&
             (!detail::adapter_chain_like<ReceiverSpec>) &&
             detail::adapter_chain_like<AdapterChain> {
    return task_into_member_receiver_binder<
        std::decay_t<F>,
        std::decay_t<ReceiverSpec>,
        std::decay_t<AdapterChain>
    > {
        std::forward<F>(f),
        std::forward<ReceiverSpec>(receiver_spec),
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
