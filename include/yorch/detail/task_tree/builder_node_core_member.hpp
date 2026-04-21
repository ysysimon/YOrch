#pragma once

#include <utility>

#include "concepts.hpp" // IWYU pragma: keep
#include "metadata.hpp"
#include "node_binder.hpp"

namespace yorch::detail {

template <typename Derived, typename... Nodes>
struct builder_node_core_member {
    template <std::size_t Level, typename F, typename ReceiverSpec>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>)
    [[nodiscard]] constexpr auto node_member(F&& f, ReceiverSpec&& receiver_spec) const& {
        return detail::tree_node_member_receiver_binder<
            const Derived*,
            Level,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            detail::no_fanout_policy_tag,
            adapter_chain<>
        > {
            static_cast<const Derived*>(this),
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            {},
            {}
        };
    }

    template <std::size_t Level, typename F, typename ReceiverSpec, typename FanoutPolicy>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>) &&
                 detail::fanout_policy<FanoutPolicy>
    [[nodiscard]] constexpr auto node_member(F&& f, ReceiverSpec&& receiver_spec, FanoutPolicy&& fanout_policy) const& {
        return detail::tree_node_member_receiver_binder<
            const Derived*,
            Level,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            std::decay_t<FanoutPolicy>,
            adapter_chain<>
        > {
            static_cast<const Derived*>(this),
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            std::forward<FanoutPolicy>(fanout_policy),
            {}
        };
    }

    template <std::size_t Level, typename F, typename ReceiverSpec, typename AdapterChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>) &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto node_member(F&& f, ReceiverSpec&& receiver_spec, AdapterChain&& adapter_specs) const& {
        return detail::tree_node_member_receiver_binder<
            const Derived*,
            Level,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            detail::no_fanout_policy_tag,
            std::decay_t<AdapterChain>
        > {
            static_cast<const Derived*>(this),
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            {},
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <std::size_t Level, typename F, typename ReceiverSpec, typename FanoutPolicy, typename AdapterChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>) &&
                 detail::fanout_policy<FanoutPolicy> &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto node_member(F&& f, ReceiverSpec&& receiver_spec, FanoutPolicy&& fanout_policy, AdapterChain&& adapter_specs) const& {
        return detail::tree_node_member_receiver_binder<
            const Derived*,
            Level,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            std::decay_t<FanoutPolicy>,
            std::decay_t<AdapterChain>
        > {
            static_cast<const Derived*>(this),
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            std::forward<FanoutPolicy>(fanout_policy),
            std::forward<AdapterChain>(adapter_specs)
        };
    }
    template <std::size_t Level, typename F, typename ReceiverSpec>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>)
    [[nodiscard]] constexpr auto node_into_member(F&& f, ReceiverSpec&& receiver_spec) const& {
        return detail::tree_node_into_member_receiver_binder<
            const Derived*,
            Level,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            detail::no_fanout_policy_tag,
            adapter_chain<>
        > {
            static_cast<const Derived*>(this),
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            {},
            {}
        };
    }

    template <std::size_t Level, typename F, typename ReceiverSpec, typename FanoutPolicy>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>) &&
                 detail::fanout_policy<FanoutPolicy>
    [[nodiscard]] constexpr auto node_into_member(F&& f, ReceiverSpec&& receiver_spec, FanoutPolicy&& fanout_policy) const& {
        return detail::tree_node_into_member_receiver_binder<
            const Derived*,
            Level,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            std::decay_t<FanoutPolicy>,
            adapter_chain<>
        > {
            static_cast<const Derived*>(this),
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            std::forward<FanoutPolicy>(fanout_policy),
            {}
        };
    }

    template <std::size_t Level, typename F, typename ReceiverSpec, typename AdapterChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>) &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto node_into_member(F&& f, ReceiverSpec&& receiver_spec, AdapterChain&& adapter_specs) const& {
        return detail::tree_node_into_member_receiver_binder<
            const Derived*,
            Level,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            detail::no_fanout_policy_tag,
            std::decay_t<AdapterChain>
        > {
            static_cast<const Derived*>(this),
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            {},
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <std::size_t Level, typename F, typename ReceiverSpec, typename FanoutPolicy, typename AdapterChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>) &&
                 detail::fanout_policy<FanoutPolicy> &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto node_into_member(F&& f, ReceiverSpec&& receiver_spec, FanoutPolicy&& fanout_policy, AdapterChain&& adapter_specs) const& {
        return detail::tree_node_into_member_receiver_binder<
            const Derived*,
            Level,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            std::decay_t<FanoutPolicy>,
            std::decay_t<AdapterChain>
        > {
            static_cast<const Derived*>(this),
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            std::forward<FanoutPolicy>(fanout_policy),
            std::forward<AdapterChain>(adapter_specs)
        };
    }
    template <std::size_t Level, typename F, typename ReceiverSpec>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>)
    [[nodiscard]] constexpr auto node_member(F&& f, ReceiverSpec&& receiver_spec) && {
        return detail::tree_node_member_receiver_binder<
            Derived,
            Level,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            detail::no_fanout_policy_tag,
            adapter_chain<>
        > {
            static_cast<Derived&&>(*this),
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            {},
            {}
        };
    }

    template <std::size_t Level, typename F, typename ReceiverSpec, typename FanoutPolicy>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>) &&
                 detail::fanout_policy<FanoutPolicy>
    [[nodiscard]] constexpr auto node_member(F&& f, ReceiverSpec&& receiver_spec, FanoutPolicy&& fanout_policy) && {
        return detail::tree_node_member_receiver_binder<
            Derived,
            Level,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            std::decay_t<FanoutPolicy>,
            adapter_chain<>
        > {
            static_cast<Derived&&>(*this),
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            std::forward<FanoutPolicy>(fanout_policy),
            {}
        };
    }

    template <std::size_t Level, typename F, typename ReceiverSpec, typename AdapterChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>) &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto node_member(F&& f, ReceiverSpec&& receiver_spec, AdapterChain&& adapter_specs) && {
        return detail::tree_node_member_receiver_binder<
            Derived,
            Level,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            detail::no_fanout_policy_tag,
            std::decay_t<AdapterChain>
        > {
            static_cast<Derived&&>(*this),
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            {},
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <std::size_t Level, typename F, typename ReceiverSpec, typename FanoutPolicy, typename AdapterChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>) &&
                 detail::fanout_policy<FanoutPolicy> &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto node_member(F&& f, ReceiverSpec&& receiver_spec, FanoutPolicy&& fanout_policy, AdapterChain&& adapter_specs) && {
        return detail::tree_node_member_receiver_binder<
            Derived,
            Level,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            std::decay_t<FanoutPolicy>,
            std::decay_t<AdapterChain>
        > {
            static_cast<Derived&&>(*this),
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            std::forward<FanoutPolicy>(fanout_policy),
            std::forward<AdapterChain>(adapter_specs)
        };
    }
    template <std::size_t Level, typename F, typename ReceiverSpec>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>)
    [[nodiscard]] constexpr auto node_into_member(F&& f, ReceiverSpec&& receiver_spec) && {
        return detail::tree_node_into_member_receiver_binder<
            Derived,
            Level,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            detail::no_fanout_policy_tag,
            adapter_chain<>
        > {
            static_cast<Derived&&>(*this),
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            {},
            {}
        };
    }

    template <std::size_t Level, typename F, typename ReceiverSpec, typename FanoutPolicy>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>) &&
                 detail::fanout_policy<FanoutPolicy>
    [[nodiscard]] constexpr auto node_into_member(F&& f, ReceiverSpec&& receiver_spec, FanoutPolicy&& fanout_policy) && {
        return detail::tree_node_into_member_receiver_binder<
            Derived,
            Level,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            std::decay_t<FanoutPolicy>,
            adapter_chain<>
        > {
            static_cast<Derived&&>(*this),
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            std::forward<FanoutPolicy>(fanout_policy),
            {}
        };
    }

    template <std::size_t Level, typename F, typename ReceiverSpec, typename AdapterChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>) &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto node_into_member(F&& f, ReceiverSpec&& receiver_spec, AdapterChain&& adapter_specs) && {
        return detail::tree_node_into_member_receiver_binder<
            Derived,
            Level,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            detail::no_fanout_policy_tag,
            std::decay_t<AdapterChain>
        > {
            static_cast<Derived&&>(*this),
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            {},
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <std::size_t Level, typename F, typename ReceiverSpec, typename FanoutPolicy, typename AdapterChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>) &&
                 detail::fanout_policy<FanoutPolicy> &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto node_into_member(F&& f, ReceiverSpec&& receiver_spec, FanoutPolicy&& fanout_policy, AdapterChain&& adapter_specs) && {
        return detail::tree_node_into_member_receiver_binder<
            Derived,
            Level,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            std::decay_t<FanoutPolicy>,
            std::decay_t<AdapterChain>
        > {
            static_cast<Derived&&>(*this),
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            std::forward<FanoutPolicy>(fanout_policy),
            std::forward<AdapterChain>(adapter_specs)
        };
    }
};

} // namespace yorch::detail
