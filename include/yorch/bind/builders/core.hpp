#pragma once

#include <type_traits>
#include <utility>

#include "../../detail/bind/tasks.hpp"
#include "../../detail/bind/traits.hpp"
#include "../adapters.hpp"

namespace yorch {

template <typename F, typename AdapterChain>
struct task_into_binder;

template <typename F, typename AdapterChain>
struct task_forward_prev_binder;

template <typename F, typename ReceiverSpec, typename AdapterChain>
struct task_member_receiver_binder;

template <typename F, typename ReceiverSpec, typename AdapterChain>
struct task_forward_prev_member_receiver_binder;

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
struct task_forward_prev_member_receiver_binder {
    F func;
    ReceiverSpec receiver_spec;
    AdapterChain adapter_specs;

    template <typename... Specs>
        requires detail::member_bound_signature_matches<F, Specs...>
    constexpr auto operator()(Specs&&... specs) const& {
        using output_type =
            detail::forward_prev_unique_prev_payload_t<
                std::decay_t<ReceiverSpec>,
                std::decay_t<Specs>...>;

        static_assert(
            !std::is_void_v<output_type>,
            "yorch::task_forward_prev_member(...) requires exactly one prev-access binding across receiver and member-function parameters");

        return yorch::apply_adapters(
            yorch::bind_forward_prev_member<output_type>(
                func,
                receiver_spec,
                std::forward<Specs>(specs)...),
            adapter_specs);
    }

    template <typename... Specs>
        requires detail::member_bound_signature_matches<F, Specs...>
    constexpr auto operator()(Specs&&... specs) && {
        using output_type =
            detail::forward_prev_unique_prev_payload_t<
                std::decay_t<ReceiverSpec>,
                std::decay_t<Specs>...>;

        static_assert(
            !std::is_void_v<output_type>,
            "yorch::task_forward_prev_member(...) requires exactly one prev-access binding across receiver and member-function parameters");

        return yorch::apply_adapters(
            yorch::bind_forward_prev_member<output_type>(
                std::move(func),
                std::move(receiver_spec),
                std::forward<Specs>(specs)...),
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
constexpr auto task_forward_prev_member(F&& f, ReceiverSpec&& receiver_spec)
    requires detail::ordinary_member_bind_callable<F> &&
             detail::member_receiver_bindable<F, ReceiverSpec> &&
             (!detail::adapter_chain_like<ReceiverSpec>) {
    return task_forward_prev_member_receiver_binder<
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
constexpr auto task_forward_prev_member(F&& f, ReceiverSpec&& receiver_spec, AdapterChain&& adapter_specs)
    requires detail::ordinary_member_bind_callable<F> &&
             detail::member_receiver_bindable<F, ReceiverSpec> &&
             (!detail::adapter_chain_like<ReceiverSpec>) &&
             detail::adapter_chain_like<AdapterChain> {
    return task_forward_prev_member_receiver_binder<
        std::decay_t<F>,
        std::decay_t<ReceiverSpec>,
        std::decay_t<AdapterChain>
    > {
        std::forward<F>(f),
        std::forward<ReceiverSpec>(receiver_spec),
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

} // namespace yorch
