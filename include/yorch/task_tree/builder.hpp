#pragma once

#include <tuple>
#include <type_traits>
#include <utility>

#include "../detail/task_tree/concepts.hpp" // IWYU pragma: keep
#include "../detail/task_tree/metadata.hpp"
#include "../detail/task_tree/node_binder.hpp"

namespace yorch {

template <typename... Nodes>
struct task_tree_builder {
    using tuple_type = std::tuple<Nodes...>;

    tuple_type nodes {};

    static constexpr std::size_t node_count = sizeof...(Nodes);
    static constexpr std::size_t max_level = detail::max_level_v<Nodes...>;

    template <std::size_t I>
    using node_type = std::tuple_element_t<I, tuple_type>;

    template <typename Task>
        requires (sizeof...(Nodes) == 0) &&
                 detail::ordinary_task_object_argument<Task>
    [[nodiscard]] constexpr auto root(Task&& task) const& {
        return node<0>(std::forward<Task>(task));
    }

    template <typename Task, typename FanoutPolicy>
        requires (sizeof...(Nodes) == 0) &&
                 detail::ordinary_task_object_argument<Task> &&
                 detail::fanout_policy<FanoutPolicy>
    [[nodiscard]] constexpr auto root(Task&& task, FanoutPolicy&& fanout_policy) const& {
        return node<0>(std::forward<Task>(task), std::forward<FanoutPolicy>(fanout_policy));
    }

    template <typename F>
        requires (sizeof...(Nodes) == 0) &&
                 detail::ordinary_callable_task_argument<F>
    [[nodiscard]] constexpr auto root(F&& f) const& {
        return detail::tree_node_binder<
            const task_tree_builder*,
            0,
            std::decay_t<F>,
            detail::no_fanout_policy_tag,
            adapter_chain<>
        > {
            this,
            std::forward<F>(f),
            {},
            {}
        };
    }

    template <typename F, typename FanoutPolicy>
        requires (sizeof...(Nodes) == 0) &&
                 detail::ordinary_callable_task_argument<F> &&
                 detail::fanout_policy<FanoutPolicy>
    [[nodiscard]] constexpr auto root(F&& f, FanoutPolicy&& fanout_policy) const& {
        return detail::tree_node_binder<
            const task_tree_builder*,
            0,
            std::decay_t<F>,
            std::decay_t<FanoutPolicy>,
            adapter_chain<>
        > {
            this,
            std::forward<F>(f),
            std::forward<FanoutPolicy>(fanout_policy),
            {}
        };
    }

    template <typename F, typename AdapterChain>
        requires (sizeof...(Nodes) == 0) &&
                 detail::ordinary_callable_task_argument<F> &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto root(F&& f, AdapterChain&& adapter_specs) const& {
        return detail::tree_node_binder<
            const task_tree_builder*,
            0,
            std::decay_t<F>,
            detail::no_fanout_policy_tag,
            std::decay_t<AdapterChain>
        > {
            this,
            std::forward<F>(f),
            {},
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <typename F, typename FanoutPolicy, typename AdapterChain>
        requires (sizeof...(Nodes) == 0) &&
                 detail::ordinary_callable_task_argument<F> &&
                 detail::fanout_policy<FanoutPolicy> &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto root(F&& f, FanoutPolicy&& fanout_policy, AdapterChain&& adapter_specs) const& {
        return detail::tree_node_binder<
            const task_tree_builder*,
            0,
            std::decay_t<F>,
            std::decay_t<FanoutPolicy>,
            std::decay_t<AdapterChain>
        > {
            this,
            std::forward<F>(f),
            std::forward<FanoutPolicy>(fanout_policy),
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <typename F, typename ReceiverSpec>
        requires (sizeof...(Nodes) == 0) &&
                 detail::ordinary_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>)
    [[nodiscard]] constexpr auto root_member(F&& f, ReceiverSpec&& receiver_spec) const& {
        return detail::tree_node_member_receiver_binder<
            const task_tree_builder*,
            0,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            detail::no_fanout_policy_tag,
            adapter_chain<>
        > {
            this,
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            {},
            {}
        };
    }

    template <typename F, typename ReceiverSpec, typename FanoutPolicy>
        requires (sizeof...(Nodes) == 0) &&
                 detail::ordinary_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>) &&
                 detail::fanout_policy<FanoutPolicy>
    [[nodiscard]] constexpr auto root_member(F&& f, ReceiverSpec&& receiver_spec, FanoutPolicy&& fanout_policy) const& {
        return detail::tree_node_member_receiver_binder<
            const task_tree_builder*,
            0,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            std::decay_t<FanoutPolicy>,
            adapter_chain<>
        > {
            this,
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            std::forward<FanoutPolicy>(fanout_policy),
            {}
        };
    }

    template <typename F, typename ReceiverSpec, typename AdapterChain>
        requires (sizeof...(Nodes) == 0) &&
                 detail::ordinary_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>) &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto root_member(F&& f, ReceiverSpec&& receiver_spec, AdapterChain&& adapter_specs) const& {
        return detail::tree_node_member_receiver_binder<
            const task_tree_builder*,
            0,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            detail::no_fanout_policy_tag,
            std::decay_t<AdapterChain>
        > {
            this,
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            {},
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <typename F, typename ReceiverSpec, typename FanoutPolicy, typename AdapterChain>
        requires (sizeof...(Nodes) == 0) &&
                 detail::ordinary_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>) &&
                 detail::fanout_policy<FanoutPolicy> &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto root_member(F&& f, ReceiverSpec&& receiver_spec, FanoutPolicy&& fanout_policy, AdapterChain&& adapter_specs) const& {
        return detail::tree_node_member_receiver_binder<
            const task_tree_builder*,
            0,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            std::decay_t<FanoutPolicy>,
            std::decay_t<AdapterChain>
        > {
            this,
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            std::forward<FanoutPolicy>(fanout_policy),
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <typename Task>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_task_object_argument<Task>
    [[nodiscard]] constexpr auto root_into(Task&& task) const& {
        return node_into<0>(std::forward<Task>(task));
    }

    template <typename Task, typename FanoutPolicy>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_task_object_argument<Task> &&
                 detail::fanout_policy<FanoutPolicy>
    [[nodiscard]] constexpr auto root_into(Task&& task, FanoutPolicy&& fanout_policy) const& {
        return node_into<0>(std::forward<Task>(task), std::forward<FanoutPolicy>(fanout_policy));
    }

    template <typename F>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_callable_task_argument<F>
    [[nodiscard]] constexpr auto root_into(F&& f) const& {
        return detail::tree_node_into_binder<
            const task_tree_builder*,
            0,
            std::decay_t<F>,
            detail::no_fanout_policy_tag,
            adapter_chain<>
        > {
            this,
            std::forward<F>(f),
            {},
            {}
        };
    }

    template <typename F, typename FanoutPolicy>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_callable_task_argument<F> &&
                 detail::fanout_policy<FanoutPolicy>
    [[nodiscard]] constexpr auto root_into(F&& f, FanoutPolicy&& fanout_policy) const& {
        return detail::tree_node_into_binder<
            const task_tree_builder*,
            0,
            std::decay_t<F>,
            std::decay_t<FanoutPolicy>,
            adapter_chain<>
        > {
            this,
            std::forward<F>(f),
            std::forward<FanoutPolicy>(fanout_policy),
            {}
        };
    }

    template <typename F, typename AdapterChain>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_callable_task_argument<F> &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto root_into(F&& f, AdapterChain&& adapter_specs) const& {
        return detail::tree_node_into_binder<
            const task_tree_builder*,
            0,
            std::decay_t<F>,
            detail::no_fanout_policy_tag,
            std::decay_t<AdapterChain>
        > {
            this,
            std::forward<F>(f),
            {},
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <typename F, typename FanoutPolicy, typename AdapterChain>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_callable_task_argument<F> &&
                 detail::fanout_policy<FanoutPolicy> &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto root_into(F&& f, FanoutPolicy&& fanout_policy, AdapterChain&& adapter_specs) const& {
        return detail::tree_node_into_binder<
            const task_tree_builder*,
            0,
            std::decay_t<F>,
            std::decay_t<FanoutPolicy>,
            std::decay_t<AdapterChain>
        > {
            this,
            std::forward<F>(f),
            std::forward<FanoutPolicy>(fanout_policy),
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <typename F>
        requires (sizeof...(Nodes) == 0) &&
                 detail::forward_prev_callable_task_argument<F>
    [[nodiscard]] constexpr auto root_forward_prev(F&& f) const& {
        return detail::tree_node_forward_prev_binder<
            const task_tree_builder*,
            0,
            std::decay_t<F>,
            detail::no_fanout_policy_tag,
            adapter_chain<>
        > {
            this,
            std::forward<F>(f),
            {},
            {}
        };
    }

    template <typename F, typename FanoutPolicy>
        requires (sizeof...(Nodes) == 0) &&
                 detail::forward_prev_callable_task_argument<F> &&
                 detail::fanout_policy<FanoutPolicy>
    [[nodiscard]] constexpr auto root_forward_prev(F&& f, FanoutPolicy&& fanout_policy) const& {
        return detail::tree_node_forward_prev_binder<
            const task_tree_builder*,
            0,
            std::decay_t<F>,
            std::decay_t<FanoutPolicy>,
            adapter_chain<>
        > {
            this,
            std::forward<F>(f),
            std::forward<FanoutPolicy>(fanout_policy),
            {}
        };
    }

    template <typename F, typename AdapterChain>
        requires (sizeof...(Nodes) == 0) &&
                 detail::forward_prev_callable_task_argument<F> &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto root_forward_prev(F&& f, AdapterChain&& adapter_specs) const& {
        return detail::tree_node_forward_prev_binder<
            const task_tree_builder*,
            0,
            std::decay_t<F>,
            detail::no_fanout_policy_tag,
            std::decay_t<AdapterChain>
        > {
            this,
            std::forward<F>(f),
            {},
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <typename F, typename FanoutPolicy, typename AdapterChain>
        requires (sizeof...(Nodes) == 0) &&
                 detail::forward_prev_callable_task_argument<F> &&
                 detail::fanout_policy<FanoutPolicy> &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto root_forward_prev(F&& f, FanoutPolicy&& fanout_policy, AdapterChain&& adapter_specs) const& {
        return detail::tree_node_forward_prev_binder<
            const task_tree_builder*,
            0,
            std::decay_t<F>,
            std::decay_t<FanoutPolicy>,
            std::decay_t<AdapterChain>
        > {
            this,
            std::forward<F>(f),
            std::forward<FanoutPolicy>(fanout_policy),
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <typename F, typename ReceiverSpec>
        requires (sizeof...(Nodes) == 0) &&
                 detail::forward_prev_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>)
    [[nodiscard]] constexpr auto root_forward_prev_member(F&& f, ReceiverSpec&& receiver_spec) const& {
        return detail::tree_node_forward_prev_member_receiver_binder<
            const task_tree_builder*,
            0,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            detail::no_fanout_policy_tag,
            adapter_chain<>
        > {
            this,
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            {},
            {}
        };
    }

    template <typename F, typename ReceiverSpec, typename FanoutPolicy>
        requires (sizeof...(Nodes) == 0) &&
                 detail::forward_prev_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>) &&
                 detail::fanout_policy<FanoutPolicy>
    [[nodiscard]] constexpr auto root_forward_prev_member(F&& f, ReceiverSpec&& receiver_spec, FanoutPolicy&& fanout_policy) const& {
        return detail::tree_node_forward_prev_member_receiver_binder<
            const task_tree_builder*,
            0,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            std::decay_t<FanoutPolicy>,
            adapter_chain<>
        > {
            this,
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            std::forward<FanoutPolicy>(fanout_policy),
            {}
        };
    }

    template <typename F, typename ReceiverSpec, typename AdapterChain>
        requires (sizeof...(Nodes) == 0) &&
                 detail::forward_prev_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>) &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto root_forward_prev_member(F&& f, ReceiverSpec&& receiver_spec, AdapterChain&& adapter_specs) const& {
        return detail::tree_node_forward_prev_member_receiver_binder<
            const task_tree_builder*,
            0,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            detail::no_fanout_policy_tag,
            std::decay_t<AdapterChain>
        > {
            this,
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            {},
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <typename F, typename ReceiverSpec, typename FanoutPolicy, typename AdapterChain>
        requires (sizeof...(Nodes) == 0) &&
                 detail::forward_prev_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>) &&
                 detail::fanout_policy<FanoutPolicy> &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto root_forward_prev_member(F&& f, ReceiverSpec&& receiver_spec, FanoutPolicy&& fanout_policy, AdapterChain&& adapter_specs) const& {
        return detail::tree_node_forward_prev_member_receiver_binder<
            const task_tree_builder*,
            0,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            std::decay_t<FanoutPolicy>,
            std::decay_t<AdapterChain>
        > {
            this,
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            std::forward<FanoutPolicy>(fanout_policy),
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <typename F>
        requires (sizeof...(Nodes) == 0) &&
                 detail::forward_prev_callable_task_argument<F>
    [[nodiscard]] constexpr auto root_forward_prev(F&& f) && {
        return detail::tree_node_forward_prev_binder<
            task_tree_builder,
            0,
            std::decay_t<F>,
            detail::no_fanout_policy_tag,
            adapter_chain<>
        > {
            std::move(*this),
            std::forward<F>(f),
            {},
            {}
        };
    }

    template <typename F, typename FanoutPolicy>
        requires (sizeof...(Nodes) == 0) &&
                 detail::forward_prev_callable_task_argument<F> &&
                 detail::fanout_policy<FanoutPolicy>
    [[nodiscard]] constexpr auto root_forward_prev(F&& f, FanoutPolicy&& fanout_policy) && {
        return detail::tree_node_forward_prev_binder<
            task_tree_builder,
            0,
            std::decay_t<F>,
            std::decay_t<FanoutPolicy>,
            adapter_chain<>
        > {
            std::move(*this),
            std::forward<F>(f),
            std::forward<FanoutPolicy>(fanout_policy),
            {}
        };
    }

    template <typename F, typename AdapterChain>
        requires (sizeof...(Nodes) == 0) &&
                 detail::forward_prev_callable_task_argument<F> &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto root_forward_prev(F&& f, AdapterChain&& adapter_specs) && {
        return detail::tree_node_forward_prev_binder<
            task_tree_builder,
            0,
            std::decay_t<F>,
            detail::no_fanout_policy_tag,
            std::decay_t<AdapterChain>
        > {
            std::move(*this),
            std::forward<F>(f),
            {},
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <typename F, typename FanoutPolicy, typename AdapterChain>
        requires (sizeof...(Nodes) == 0) &&
                 detail::forward_prev_callable_task_argument<F> &&
                 detail::fanout_policy<FanoutPolicy> &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto root_forward_prev(F&& f, FanoutPolicy&& fanout_policy, AdapterChain&& adapter_specs) && {
        return detail::tree_node_forward_prev_binder<
            task_tree_builder,
            0,
            std::decay_t<F>,
            std::decay_t<FanoutPolicy>,
            std::decay_t<AdapterChain>
        > {
            std::move(*this),
            std::forward<F>(f),
            std::forward<FanoutPolicy>(fanout_policy),
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <typename F, typename ReceiverSpec>
        requires (sizeof...(Nodes) == 0) &&
                 detail::forward_prev_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>)
    [[nodiscard]] constexpr auto root_forward_prev_member(F&& f, ReceiverSpec&& receiver_spec) && {
        return detail::tree_node_forward_prev_member_receiver_binder<
            task_tree_builder,
            0,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            detail::no_fanout_policy_tag,
            adapter_chain<>
        > {
            std::move(*this),
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            {},
            {}
        };
    }

    template <typename F, typename ReceiverSpec, typename FanoutPolicy>
        requires (sizeof...(Nodes) == 0) &&
                 detail::forward_prev_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>) &&
                 detail::fanout_policy<FanoutPolicy>
    [[nodiscard]] constexpr auto root_forward_prev_member(F&& f, ReceiverSpec&& receiver_spec, FanoutPolicy&& fanout_policy) && {
        return detail::tree_node_forward_prev_member_receiver_binder<
            task_tree_builder,
            0,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            std::decay_t<FanoutPolicy>,
            adapter_chain<>
        > {
            std::move(*this),
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            std::forward<FanoutPolicy>(fanout_policy),
            {}
        };
    }

    template <typename F, typename ReceiverSpec, typename AdapterChain>
        requires (sizeof...(Nodes) == 0) &&
                 detail::forward_prev_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>) &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto root_forward_prev_member(F&& f, ReceiverSpec&& receiver_spec, AdapterChain&& adapter_specs) && {
        return detail::tree_node_forward_prev_member_receiver_binder<
            task_tree_builder,
            0,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            detail::no_fanout_policy_tag,
            std::decay_t<AdapterChain>
        > {
            std::move(*this),
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            {},
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <typename F, typename ReceiverSpec, typename FanoutPolicy, typename AdapterChain>
        requires (sizeof...(Nodes) == 0) &&
                 detail::forward_prev_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>) &&
                 detail::fanout_policy<FanoutPolicy> &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto root_forward_prev_member(F&& f, ReceiverSpec&& receiver_spec, FanoutPolicy&& fanout_policy, AdapterChain&& adapter_specs) && {
        return detail::tree_node_forward_prev_member_receiver_binder<
            task_tree_builder,
            0,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            std::decay_t<FanoutPolicy>,
            std::decay_t<AdapterChain>
        > {
            std::move(*this),
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            std::forward<FanoutPolicy>(fanout_policy),
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <typename F, typename ReceiverSpec>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>)
    [[nodiscard]] constexpr auto root_into_member(F&& f, ReceiverSpec&& receiver_spec) const& {
        return detail::tree_node_into_member_receiver_binder<
            const task_tree_builder*,
            0,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            detail::no_fanout_policy_tag,
            adapter_chain<>
        > {
            this,
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            {},
            {}
        };
    }

    template <typename F, typename ReceiverSpec, typename FanoutPolicy>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>) &&
                 detail::fanout_policy<FanoutPolicy>
    [[nodiscard]] constexpr auto root_into_member(F&& f, ReceiverSpec&& receiver_spec, FanoutPolicy&& fanout_policy) const& {
        return detail::tree_node_into_member_receiver_binder<
            const task_tree_builder*,
            0,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            std::decay_t<FanoutPolicy>,
            adapter_chain<>
        > {
            this,
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            std::forward<FanoutPolicy>(fanout_policy),
            {}
        };
    }

    template <typename F, typename ReceiverSpec, typename AdapterChain>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>) &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto root_into_member(F&& f, ReceiverSpec&& receiver_spec, AdapterChain&& adapter_specs) const& {
        return detail::tree_node_into_member_receiver_binder<
            const task_tree_builder*,
            0,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            detail::no_fanout_policy_tag,
            std::decay_t<AdapterChain>
        > {
            this,
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            {},
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <typename F, typename ReceiverSpec, typename FanoutPolicy, typename AdapterChain>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>) &&
                 detail::fanout_policy<FanoutPolicy> &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto root_into_member(F&& f, ReceiverSpec&& receiver_spec, FanoutPolicy&& fanout_policy, AdapterChain&& adapter_specs) const& {
        return detail::tree_node_into_member_receiver_binder<
            const task_tree_builder*,
            0,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            std::decay_t<FanoutPolicy>,
            std::decay_t<AdapterChain>
        > {
            this,
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            std::forward<FanoutPolicy>(fanout_policy),
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <typename Task>
        requires (sizeof...(Nodes) == 0) &&
                 detail::ordinary_task_object_argument<Task>
    [[nodiscard]] constexpr auto root(Task&& task) && {
        return std::move(*this).template node<0>(std::forward<Task>(task));
    }

    template <typename Task, typename FanoutPolicy>
        requires (sizeof...(Nodes) == 0) &&
                 detail::ordinary_task_object_argument<Task> &&
                 detail::fanout_policy<FanoutPolicy>
    [[nodiscard]] constexpr auto root(Task&& task, FanoutPolicy&& fanout_policy) && {
        return std::move(*this).template node<0>(std::forward<Task>(task), std::forward<FanoutPolicy>(fanout_policy));
    }

    template <typename F>
        requires (sizeof...(Nodes) == 0) &&
                 detail::ordinary_callable_task_argument<F>
    [[nodiscard]] constexpr auto root(F&& f) && {
        return detail::tree_node_binder<
            task_tree_builder,
            0,
            std::decay_t<F>,
            detail::no_fanout_policy_tag,
            adapter_chain<>
        > {
            std::move(*this),
            std::forward<F>(f),
            {},
            {}
        };
    }

    template <typename F, typename FanoutPolicy>
        requires (sizeof...(Nodes) == 0) &&
                 detail::ordinary_callable_task_argument<F> &&
                 detail::fanout_policy<FanoutPolicy>
    [[nodiscard]] constexpr auto root(F&& f, FanoutPolicy&& fanout_policy) && {
        return detail::tree_node_binder<
            task_tree_builder,
            0,
            std::decay_t<F>,
            std::decay_t<FanoutPolicy>,
            adapter_chain<>
        > {
            std::move(*this),
            std::forward<F>(f),
            std::forward<FanoutPolicy>(fanout_policy),
            {}
        };
    }

    template <typename F, typename AdapterChain>
        requires (sizeof...(Nodes) == 0) &&
                 detail::ordinary_callable_task_argument<F> &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto root(F&& f, AdapterChain&& adapter_specs) && {
        return detail::tree_node_binder<
            task_tree_builder,
            0,
            std::decay_t<F>,
            detail::no_fanout_policy_tag,
            std::decay_t<AdapterChain>
        > {
            std::move(*this),
            std::forward<F>(f),
            {},
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <typename F, typename FanoutPolicy, typename AdapterChain>
        requires (sizeof...(Nodes) == 0) &&
                 detail::ordinary_callable_task_argument<F> &&
                 detail::fanout_policy<FanoutPolicy> &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto root(F&& f, FanoutPolicy&& fanout_policy, AdapterChain&& adapter_specs) && {
        return detail::tree_node_binder<
            task_tree_builder,
            0,
            std::decay_t<F>,
            std::decay_t<FanoutPolicy>,
            std::decay_t<AdapterChain>
        > {
            std::move(*this),
            std::forward<F>(f),
            std::forward<FanoutPolicy>(fanout_policy),
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <typename F, typename ReceiverSpec>
        requires (sizeof...(Nodes) == 0) &&
                 detail::ordinary_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>)
    [[nodiscard]] constexpr auto root_member(F&& f, ReceiverSpec&& receiver_spec) && {
        return detail::tree_node_member_receiver_binder<
            task_tree_builder,
            0,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            detail::no_fanout_policy_tag,
            adapter_chain<>
        > {
            std::move(*this),
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            {},
            {}
        };
    }

    template <typename F, typename ReceiverSpec, typename FanoutPolicy>
        requires (sizeof...(Nodes) == 0) &&
                 detail::ordinary_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>) &&
                 detail::fanout_policy<FanoutPolicy>
    [[nodiscard]] constexpr auto root_member(F&& f, ReceiverSpec&& receiver_spec, FanoutPolicy&& fanout_policy) && {
        return detail::tree_node_member_receiver_binder<
            task_tree_builder,
            0,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            std::decay_t<FanoutPolicy>,
            adapter_chain<>
        > {
            std::move(*this),
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            std::forward<FanoutPolicy>(fanout_policy),
            {}
        };
    }

    template <typename F, typename ReceiverSpec, typename AdapterChain>
        requires (sizeof...(Nodes) == 0) &&
                 detail::ordinary_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>) &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto root_member(F&& f, ReceiverSpec&& receiver_spec, AdapterChain&& adapter_specs) && {
        return detail::tree_node_member_receiver_binder<
            task_tree_builder,
            0,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            detail::no_fanout_policy_tag,
            std::decay_t<AdapterChain>
        > {
            std::move(*this),
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            {},
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <typename F, typename ReceiverSpec, typename FanoutPolicy, typename AdapterChain>
        requires (sizeof...(Nodes) == 0) &&
                 detail::ordinary_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>) &&
                 detail::fanout_policy<FanoutPolicy> &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto root_member(F&& f, ReceiverSpec&& receiver_spec, FanoutPolicy&& fanout_policy, AdapterChain&& adapter_specs) && {
        return detail::tree_node_member_receiver_binder<
            task_tree_builder,
            0,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            std::decay_t<FanoutPolicy>,
            std::decay_t<AdapterChain>
        > {
            std::move(*this),
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            std::forward<FanoutPolicy>(fanout_policy),
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <typename Task>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_task_object_argument<Task>
    [[nodiscard]] constexpr auto root_into(Task&& task) && {
        return std::move(*this).template node_into<0>(std::forward<Task>(task));
    }

    template <typename Task, typename FanoutPolicy>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_task_object_argument<Task> &&
                 detail::fanout_policy<FanoutPolicy>
    [[nodiscard]] constexpr auto root_into(Task&& task, FanoutPolicy&& fanout_policy) && {
        return std::move(*this).template node_into<0>(std::forward<Task>(task), std::forward<FanoutPolicy>(fanout_policy));
    }

    template <typename F>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_callable_task_argument<F>
    [[nodiscard]] constexpr auto root_into(F&& f) && {
        return detail::tree_node_into_binder<
            task_tree_builder,
            0,
            std::decay_t<F>,
            detail::no_fanout_policy_tag,
            adapter_chain<>
        > {
            std::move(*this),
            std::forward<F>(f),
            {},
            {}
        };
    }

    template <typename F, typename FanoutPolicy>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_callable_task_argument<F> &&
                 detail::fanout_policy<FanoutPolicy>
    [[nodiscard]] constexpr auto root_into(F&& f, FanoutPolicy&& fanout_policy) && {
        return detail::tree_node_into_binder<
            task_tree_builder,
            0,
            std::decay_t<F>,
            std::decay_t<FanoutPolicy>,
            adapter_chain<>
        > {
            std::move(*this),
            std::forward<F>(f),
            std::forward<FanoutPolicy>(fanout_policy),
            {}
        };
    }

    template <typename F, typename AdapterChain>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_callable_task_argument<F> &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto root_into(F&& f, AdapterChain&& adapter_specs) && {
        return detail::tree_node_into_binder<
            task_tree_builder,
            0,
            std::decay_t<F>,
            detail::no_fanout_policy_tag,
            std::decay_t<AdapterChain>
        > {
            std::move(*this),
            std::forward<F>(f),
            {},
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <typename F, typename FanoutPolicy, typename AdapterChain>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_callable_task_argument<F> &&
                 detail::fanout_policy<FanoutPolicy> &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto root_into(F&& f, FanoutPolicy&& fanout_policy, AdapterChain&& adapter_specs) && {
        return detail::tree_node_into_binder<
            task_tree_builder,
            0,
            std::decay_t<F>,
            std::decay_t<FanoutPolicy>,
            std::decay_t<AdapterChain>
        > {
            std::move(*this),
            std::forward<F>(f),
            std::forward<FanoutPolicy>(fanout_policy),
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <typename F, typename ReceiverSpec>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>)
    [[nodiscard]] constexpr auto root_into_member(F&& f, ReceiverSpec&& receiver_spec) && {
        return detail::tree_node_into_member_receiver_binder<
            task_tree_builder,
            0,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            detail::no_fanout_policy_tag,
            adapter_chain<>
        > {
            std::move(*this),
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            {},
            {}
        };
    }

    template <typename F, typename ReceiverSpec, typename FanoutPolicy>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>) &&
                 detail::fanout_policy<FanoutPolicy>
    [[nodiscard]] constexpr auto root_into_member(F&& f, ReceiverSpec&& receiver_spec, FanoutPolicy&& fanout_policy) && {
        return detail::tree_node_into_member_receiver_binder<
            task_tree_builder,
            0,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            std::decay_t<FanoutPolicy>,
            adapter_chain<>
        > {
            std::move(*this),
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            std::forward<FanoutPolicy>(fanout_policy),
            {}
        };
    }

    template <typename F, typename ReceiverSpec, typename AdapterChain>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>) &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto root_into_member(F&& f, ReceiverSpec&& receiver_spec, AdapterChain&& adapter_specs) && {
        return detail::tree_node_into_member_receiver_binder<
            task_tree_builder,
            0,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            detail::no_fanout_policy_tag,
            std::decay_t<AdapterChain>
        > {
            std::move(*this),
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            {},
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <typename F, typename ReceiverSpec, typename FanoutPolicy, typename AdapterChain>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>) &&
                 detail::fanout_policy<FanoutPolicy> &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto root_into_member(F&& f, ReceiverSpec&& receiver_spec, FanoutPolicy&& fanout_policy, AdapterChain&& adapter_specs) && {
        return detail::tree_node_into_member_receiver_binder<
            task_tree_builder,
            0,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            std::decay_t<FanoutPolicy>,
            std::decay_t<AdapterChain>
        > {
            std::move(*this),
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            std::forward<FanoutPolicy>(fanout_policy),
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <std::size_t Level, typename Task>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_task_object_argument<Task>
    [[nodiscard]] constexpr auto node(Task&& task) const& {
        using next_node_t = detail::task_tree_node_t<Level, Task>;
        return task_tree_builder<Nodes..., next_node_t> {
            std::tuple_cat(nodes, std::tuple<next_node_t> {next_node_t {std::forward<Task>(task)}})
        };
    }

    template <std::size_t Level, typename Task, typename FanoutPolicy>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_task_object_argument<Task> &&
                 detail::fanout_policy<FanoutPolicy>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    [[nodiscard]] constexpr auto node(Task&& task, FanoutPolicy&&) const& {
        using next_node_t = detail::task_tree_node_t<Level, Task, FanoutPolicy>;
        return task_tree_builder<Nodes..., next_node_t> {
            std::tuple_cat(nodes, std::tuple<next_node_t> {next_node_t {std::forward<Task>(task)}})
        };
    }

    template <std::size_t Level, typename Task>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_task_object_argument<Task>
    [[nodiscard]] constexpr auto node_into(Task&& task) const& {
        using next_node_t = detail::task_tree_node_t<Level, Task>;
        return task_tree_builder<Nodes..., next_node_t> {
            std::tuple_cat(nodes, std::tuple<next_node_t> {next_node_t {std::forward<Task>(task)}})
        };
    }

    template <std::size_t Level, typename Task, typename FanoutPolicy>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_task_object_argument<Task> &&
                 detail::fanout_policy<FanoutPolicy>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    [[nodiscard]] constexpr auto node_into(Task&& task, FanoutPolicy&&) const& {
        using next_node_t = detail::task_tree_node_t<Level, Task, FanoutPolicy>;
        return task_tree_builder<Nodes..., next_node_t> {
            std::tuple_cat(nodes, std::tuple<next_node_t> {next_node_t {std::forward<Task>(task)}})
        };
    }

    template <std::size_t Level, typename F>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_callable_task_argument<F>
    [[nodiscard]] constexpr auto node(F&& f) const& {
        return detail::tree_node_binder<
            const task_tree_builder*,
            Level,
            std::decay_t<F>,
            detail::no_fanout_policy_tag,
            adapter_chain<>
        > {
            this,
            std::forward<F>(f),
            {},
            {}
        };
    }

    template <std::size_t Level, typename F, typename FanoutPolicy>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_callable_task_argument<F> &&
                 detail::fanout_policy<FanoutPolicy>
    [[nodiscard]] constexpr auto node(F&& f, FanoutPolicy&& fanout_policy) const& {
        return detail::tree_node_binder<
            const task_tree_builder*,
            Level,
            std::decay_t<F>,
            std::decay_t<FanoutPolicy>,
            adapter_chain<>
        > {
            this,
            std::forward<F>(f),
            std::forward<FanoutPolicy>(fanout_policy),
            {}
        };
    }

    template <std::size_t Level, typename F, typename AdapterChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_callable_task_argument<F> &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto node(F&& f, AdapterChain&& adapter_specs) const& {
        return detail::tree_node_binder<
            const task_tree_builder*,
            Level,
            std::decay_t<F>,
            detail::no_fanout_policy_tag,
            std::decay_t<AdapterChain>
        > {
            this,
            std::forward<F>(f),
            {},
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <std::size_t Level, typename F, typename FanoutPolicy, typename AdapterChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_callable_task_argument<F> &&
                 detail::fanout_policy<FanoutPolicy> &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto node(F&& f, FanoutPolicy&& fanout_policy, AdapterChain&& adapter_specs) const& {
        return detail::tree_node_binder<
            const task_tree_builder*,
            Level,
            std::decay_t<F>,
            std::decay_t<FanoutPolicy>,
            std::decay_t<AdapterChain>
        > {
            this,
            std::forward<F>(f),
            std::forward<FanoutPolicy>(fanout_policy),
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <std::size_t Level, typename F, typename ReceiverSpec>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>)
    [[nodiscard]] constexpr auto node_member(F&& f, ReceiverSpec&& receiver_spec) const& {
        return detail::tree_node_member_receiver_binder<
            const task_tree_builder*,
            Level,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            detail::no_fanout_policy_tag,
            adapter_chain<>
        > {
            this,
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
            const task_tree_builder*,
            Level,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            std::decay_t<FanoutPolicy>,
            adapter_chain<>
        > {
            this,
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
            const task_tree_builder*,
            Level,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            detail::no_fanout_policy_tag,
            std::decay_t<AdapterChain>
        > {
            this,
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
            const task_tree_builder*,
            Level,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            std::decay_t<FanoutPolicy>,
            std::decay_t<AdapterChain>
        > {
            this,
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            std::forward<FanoutPolicy>(fanout_policy),
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <std::size_t Level, typename F>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_callable_task_argument<F>
    [[nodiscard]] constexpr auto node_into(F&& f) const& {
        return detail::tree_node_into_binder<
            const task_tree_builder*,
            Level,
            std::decay_t<F>,
            detail::no_fanout_policy_tag,
            adapter_chain<>
        > {
            this,
            std::forward<F>(f),
            {},
            {}
        };
    }

    template <std::size_t Level, typename F, typename FanoutPolicy>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_callable_task_argument<F> &&
                 detail::fanout_policy<FanoutPolicy>
    [[nodiscard]] constexpr auto node_into(F&& f, FanoutPolicy&& fanout_policy) const& {
        return detail::tree_node_into_binder<
            const task_tree_builder*,
            Level,
            std::decay_t<F>,
            std::decay_t<FanoutPolicy>,
            adapter_chain<>
        > {
            this,
            std::forward<F>(f),
            std::forward<FanoutPolicy>(fanout_policy),
            {}
        };
    }

    template <std::size_t Level, typename F, typename AdapterChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_callable_task_argument<F> &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto node_into(F&& f, AdapterChain&& adapter_specs) const& {
        return detail::tree_node_into_binder<
            const task_tree_builder*,
            Level,
            std::decay_t<F>,
            detail::no_fanout_policy_tag,
            std::decay_t<AdapterChain>
        > {
            this,
            std::forward<F>(f),
            {},
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <std::size_t Level, typename F, typename FanoutPolicy, typename AdapterChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_callable_task_argument<F> &&
                 detail::fanout_policy<FanoutPolicy> &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto node_into(F&& f, FanoutPolicy&& fanout_policy, AdapterChain&& adapter_specs) const& {
        return detail::tree_node_into_binder<
            const task_tree_builder*,
            Level,
            std::decay_t<F>,
            std::decay_t<FanoutPolicy>,
            std::decay_t<AdapterChain>
        > {
            this,
            std::forward<F>(f),
            std::forward<FanoutPolicy>(fanout_policy),
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <std::size_t Level, typename F>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::forward_prev_callable_task_argument<F>
    [[nodiscard]] constexpr auto node_forward_prev(F&& f) const& {
        return detail::tree_node_forward_prev_binder<
            const task_tree_builder*,
            Level,
            std::decay_t<F>,
            detail::no_fanout_policy_tag,
            adapter_chain<>
        > {
            this,
            std::forward<F>(f),
            {},
            {}
        };
    }

    template <std::size_t Level, typename F, typename FanoutPolicy>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::forward_prev_callable_task_argument<F> &&
                 detail::fanout_policy<FanoutPolicy>
    [[nodiscard]] constexpr auto node_forward_prev(F&& f, FanoutPolicy&& fanout_policy) const& {
        return detail::tree_node_forward_prev_binder<
            const task_tree_builder*,
            Level,
            std::decay_t<F>,
            std::decay_t<FanoutPolicy>,
            adapter_chain<>
        > {
            this,
            std::forward<F>(f),
            std::forward<FanoutPolicy>(fanout_policy),
            {}
        };
    }

    template <std::size_t Level, typename F, typename AdapterChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::forward_prev_callable_task_argument<F> &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto node_forward_prev(F&& f, AdapterChain&& adapter_specs) const& {
        return detail::tree_node_forward_prev_binder<
            const task_tree_builder*,
            Level,
            std::decay_t<F>,
            detail::no_fanout_policy_tag,
            std::decay_t<AdapterChain>
        > {
            this,
            std::forward<F>(f),
            {},
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <std::size_t Level, typename F, typename FanoutPolicy, typename AdapterChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::forward_prev_callable_task_argument<F> &&
                 detail::fanout_policy<FanoutPolicy> &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto node_forward_prev(F&& f, FanoutPolicy&& fanout_policy, AdapterChain&& adapter_specs) const& {
        return detail::tree_node_forward_prev_binder<
            const task_tree_builder*,
            Level,
            std::decay_t<F>,
            std::decay_t<FanoutPolicy>,
            std::decay_t<AdapterChain>
        > {
            this,
            std::forward<F>(f),
            std::forward<FanoutPolicy>(fanout_policy),
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <std::size_t Level, typename F, typename ReceiverSpec>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::forward_prev_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>)
    [[nodiscard]] constexpr auto node_forward_prev_member(F&& f, ReceiverSpec&& receiver_spec) const& {
        return detail::tree_node_forward_prev_member_receiver_binder<
            const task_tree_builder*,
            Level,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            detail::no_fanout_policy_tag,
            adapter_chain<>
        > {
            this,
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            {},
            {}
        };
    }

    template <std::size_t Level, typename F, typename ReceiverSpec, typename FanoutPolicy>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::forward_prev_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>) &&
                 detail::fanout_policy<FanoutPolicy>
    [[nodiscard]] constexpr auto node_forward_prev_member(F&& f, ReceiverSpec&& receiver_spec, FanoutPolicy&& fanout_policy) const& {
        return detail::tree_node_forward_prev_member_receiver_binder<
            const task_tree_builder*,
            Level,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            std::decay_t<FanoutPolicy>,
            adapter_chain<>
        > {
            this,
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            std::forward<FanoutPolicy>(fanout_policy),
            {}
        };
    }

    template <std::size_t Level, typename F, typename ReceiverSpec, typename AdapterChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::forward_prev_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>) &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto node_forward_prev_member(F&& f, ReceiverSpec&& receiver_spec, AdapterChain&& adapter_specs) const& {
        return detail::tree_node_forward_prev_member_receiver_binder<
            const task_tree_builder*,
            Level,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            detail::no_fanout_policy_tag,
            std::decay_t<AdapterChain>
        > {
            this,
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            {},
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <std::size_t Level, typename F, typename ReceiverSpec, typename FanoutPolicy, typename AdapterChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::forward_prev_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>) &&
                 detail::fanout_policy<FanoutPolicy> &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto node_forward_prev_member(F&& f, ReceiverSpec&& receiver_spec, FanoutPolicy&& fanout_policy, AdapterChain&& adapter_specs) const& {
        return detail::tree_node_forward_prev_member_receiver_binder<
            const task_tree_builder*,
            Level,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            std::decay_t<FanoutPolicy>,
            std::decay_t<AdapterChain>
        > {
            this,
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            std::forward<FanoutPolicy>(fanout_policy),
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <std::size_t Level, typename F>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::forward_prev_callable_task_argument<F>
    [[nodiscard]] constexpr auto node_forward_prev(F&& f) && {
        return detail::tree_node_forward_prev_binder<
            task_tree_builder,
            Level,
            std::decay_t<F>,
            detail::no_fanout_policy_tag,
            adapter_chain<>
        > {
            std::move(*this),
            std::forward<F>(f),
            {},
            {}
        };
    }

    template <std::size_t Level, typename F, typename FanoutPolicy>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::forward_prev_callable_task_argument<F> &&
                 detail::fanout_policy<FanoutPolicy>
    [[nodiscard]] constexpr auto node_forward_prev(F&& f, FanoutPolicy&& fanout_policy) && {
        return detail::tree_node_forward_prev_binder<
            task_tree_builder,
            Level,
            std::decay_t<F>,
            std::decay_t<FanoutPolicy>,
            adapter_chain<>
        > {
            std::move(*this),
            std::forward<F>(f),
            std::forward<FanoutPolicy>(fanout_policy),
            {}
        };
    }

    template <std::size_t Level, typename F, typename AdapterChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::forward_prev_callable_task_argument<F> &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto node_forward_prev(F&& f, AdapterChain&& adapter_specs) && {
        return detail::tree_node_forward_prev_binder<
            task_tree_builder,
            Level,
            std::decay_t<F>,
            detail::no_fanout_policy_tag,
            std::decay_t<AdapterChain>
        > {
            std::move(*this),
            std::forward<F>(f),
            {},
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <std::size_t Level, typename F, typename FanoutPolicy, typename AdapterChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::forward_prev_callable_task_argument<F> &&
                 detail::fanout_policy<FanoutPolicy> &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto node_forward_prev(F&& f, FanoutPolicy&& fanout_policy, AdapterChain&& adapter_specs) && {
        return detail::tree_node_forward_prev_binder<
            task_tree_builder,
            Level,
            std::decay_t<F>,
            std::decay_t<FanoutPolicy>,
            std::decay_t<AdapterChain>
        > {
            std::move(*this),
            std::forward<F>(f),
            std::forward<FanoutPolicy>(fanout_policy),
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <std::size_t Level, typename F, typename ReceiverSpec>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::forward_prev_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>)
    [[nodiscard]] constexpr auto node_forward_prev_member(F&& f, ReceiverSpec&& receiver_spec) && {
        return detail::tree_node_forward_prev_member_receiver_binder<
            task_tree_builder,
            Level,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            detail::no_fanout_policy_tag,
            adapter_chain<>
        > {
            std::move(*this),
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            {},
            {}
        };
    }

    template <std::size_t Level, typename F, typename ReceiverSpec, typename FanoutPolicy>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::forward_prev_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>) &&
                 detail::fanout_policy<FanoutPolicy>
    [[nodiscard]] constexpr auto node_forward_prev_member(F&& f, ReceiverSpec&& receiver_spec, FanoutPolicy&& fanout_policy) && {
        return detail::tree_node_forward_prev_member_receiver_binder<
            task_tree_builder,
            Level,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            std::decay_t<FanoutPolicy>,
            adapter_chain<>
        > {
            std::move(*this),
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            std::forward<FanoutPolicy>(fanout_policy),
            {}
        };
    }

    template <std::size_t Level, typename F, typename ReceiverSpec, typename AdapterChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::forward_prev_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>) &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto node_forward_prev_member(F&& f, ReceiverSpec&& receiver_spec, AdapterChain&& adapter_specs) && {
        return detail::tree_node_forward_prev_member_receiver_binder<
            task_tree_builder,
            Level,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            detail::no_fanout_policy_tag,
            std::decay_t<AdapterChain>
        > {
            std::move(*this),
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            {},
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <std::size_t Level, typename F, typename ReceiverSpec, typename FanoutPolicy, typename AdapterChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::forward_prev_member_callable_task_argument<F> &&
                 detail::member_receiver_bindable<F, ReceiverSpec> &&
                 (!detail::fanout_policy_or_chain<ReceiverSpec>) &&
                 detail::fanout_policy<FanoutPolicy> &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto node_forward_prev_member(F&& f, ReceiverSpec&& receiver_spec, FanoutPolicy&& fanout_policy, AdapterChain&& adapter_specs) && {
        return detail::tree_node_forward_prev_member_receiver_binder<
            task_tree_builder,
            Level,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            std::decay_t<FanoutPolicy>,
            std::decay_t<AdapterChain>
        > {
            std::move(*this),
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
            const task_tree_builder*,
            Level,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            detail::no_fanout_policy_tag,
            adapter_chain<>
        > {
            this,
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
            const task_tree_builder*,
            Level,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            std::decay_t<FanoutPolicy>,
            adapter_chain<>
        > {
            this,
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
            const task_tree_builder*,
            Level,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            detail::no_fanout_policy_tag,
            std::decay_t<AdapterChain>
        > {
            this,
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
            const task_tree_builder*,
            Level,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            std::decay_t<FanoutPolicy>,
            std::decay_t<AdapterChain>
        > {
            this,
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            std::forward<FanoutPolicy>(fanout_policy),
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <std::size_t Level, typename Task>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_task_object_argument<Task>
    [[nodiscard]] constexpr auto node(Task&& task) && {
        using next_node_t = detail::task_tree_node_t<Level, Task>;
        return task_tree_builder<Nodes..., next_node_t> {
            std::tuple_cat(std::move(nodes), std::tuple<next_node_t> {next_node_t {std::forward<Task>(task)}})
        };
    }

    template <std::size_t Level, typename Task, typename FanoutPolicy>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_task_object_argument<Task> &&
                 detail::fanout_policy<FanoutPolicy>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    [[nodiscard]] constexpr auto node(Task&& task, FanoutPolicy&&) && {
        using next_node_t = detail::task_tree_node_t<Level, Task, FanoutPolicy>;
        return task_tree_builder<Nodes..., next_node_t> {
            std::tuple_cat(std::move(nodes), std::tuple<next_node_t> {next_node_t {std::forward<Task>(task)}})
        };
    }

    template <std::size_t Level, typename Task>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_task_object_argument<Task>
    [[nodiscard]] constexpr auto node_into(Task&& task) && {
        using next_node_t = detail::task_tree_node_t<Level, Task>;
        return task_tree_builder<Nodes..., next_node_t> {
            std::tuple_cat(std::move(nodes), std::tuple<next_node_t> {next_node_t {std::forward<Task>(task)}})
        };
    }

    template <std::size_t Level, typename Task, typename FanoutPolicy>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_task_object_argument<Task> &&
                 detail::fanout_policy<FanoutPolicy>
                 // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    [[nodiscard]] constexpr auto node_into(Task&& task, FanoutPolicy&&) && {
        using next_node_t = detail::task_tree_node_t<Level, Task, FanoutPolicy>;
        return task_tree_builder<Nodes..., next_node_t> {
            std::tuple_cat(std::move(nodes), std::tuple<next_node_t> {next_node_t {std::forward<Task>(task)}})
        };
    }

    template <std::size_t Level, typename F>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_callable_task_argument<F>
    [[nodiscard]] constexpr auto node(F&& f) && {
        return detail::tree_node_binder<
            task_tree_builder,
            Level,
            std::decay_t<F>,
            detail::no_fanout_policy_tag,
            adapter_chain<>
        > {
            std::move(*this),
            std::forward<F>(f),
            {},
            {}
        };
    }

    template <std::size_t Level, typename F, typename FanoutPolicy>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_callable_task_argument<F> &&
                 detail::fanout_policy<FanoutPolicy>
    [[nodiscard]] constexpr auto node(F&& f, FanoutPolicy&& fanout_policy) && {
        return detail::tree_node_binder<
            task_tree_builder,
            Level,
            std::decay_t<F>,
            std::decay_t<FanoutPolicy>,
            adapter_chain<>
        > {
            std::move(*this),
            std::forward<F>(f),
            std::forward<FanoutPolicy>(fanout_policy),
            {}
        };
    }

    template <std::size_t Level, typename F, typename AdapterChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_callable_task_argument<F> &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto node(F&& f, AdapterChain&& adapter_specs) && {
        return detail::tree_node_binder<
            task_tree_builder,
            Level,
            std::decay_t<F>,
            detail::no_fanout_policy_tag,
            std::decay_t<AdapterChain>
        > {
            std::move(*this),
            std::forward<F>(f),
            {},
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <std::size_t Level, typename F, typename FanoutPolicy, typename AdapterChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_callable_task_argument<F> &&
                 detail::fanout_policy<FanoutPolicy> &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto node(F&& f, FanoutPolicy&& fanout_policy, AdapterChain&& adapter_specs) && {
        return detail::tree_node_binder<
            task_tree_builder,
            Level,
            std::decay_t<F>,
            std::decay_t<FanoutPolicy>,
            std::decay_t<AdapterChain>
        > {
            std::move(*this),
            std::forward<F>(f),
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
            task_tree_builder,
            Level,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            detail::no_fanout_policy_tag,
            adapter_chain<>
        > {
            std::move(*this),
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
            task_tree_builder,
            Level,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            std::decay_t<FanoutPolicy>,
            adapter_chain<>
        > {
            std::move(*this),
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
            task_tree_builder,
            Level,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            detail::no_fanout_policy_tag,
            std::decay_t<AdapterChain>
        > {
            std::move(*this),
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
            task_tree_builder,
            Level,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            std::decay_t<FanoutPolicy>,
            std::decay_t<AdapterChain>
        > {
            std::move(*this),
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            std::forward<FanoutPolicy>(fanout_policy),
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <std::size_t Level, typename F>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_callable_task_argument<F>
    [[nodiscard]] constexpr auto node_into(F&& f) && {
        return detail::tree_node_into_binder<
            task_tree_builder,
            Level,
            std::decay_t<F>,
            detail::no_fanout_policy_tag,
            adapter_chain<>
        > {
            std::move(*this),
            std::forward<F>(f),
            {},
            {}
        };
    }

    template <std::size_t Level, typename F, typename FanoutPolicy>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_callable_task_argument<F> &&
                 detail::fanout_policy<FanoutPolicy>
    [[nodiscard]] constexpr auto node_into(F&& f, FanoutPolicy&& fanout_policy) && {
        return detail::tree_node_into_binder<
            task_tree_builder,
            Level,
            std::decay_t<F>,
            std::decay_t<FanoutPolicy>,
            adapter_chain<>
        > {
            std::move(*this),
            std::forward<F>(f),
            std::forward<FanoutPolicy>(fanout_policy),
            {}
        };
    }

    template <std::size_t Level, typename F, typename AdapterChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_callable_task_argument<F> &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto node_into(F&& f, AdapterChain&& adapter_specs) && {
        return detail::tree_node_into_binder<
            task_tree_builder,
            Level,
            std::decay_t<F>,
            detail::no_fanout_policy_tag,
            std::decay_t<AdapterChain>
        > {
            std::move(*this),
            std::forward<F>(f),
            {},
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <std::size_t Level, typename F, typename FanoutPolicy, typename AdapterChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_callable_task_argument<F> &&
                 detail::fanout_policy<FanoutPolicy> &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto node_into(F&& f, FanoutPolicy&& fanout_policy, AdapterChain&& adapter_specs) && {
        return detail::tree_node_into_binder<
            task_tree_builder,
            Level,
            std::decay_t<F>,
            std::decay_t<FanoutPolicy>,
            std::decay_t<AdapterChain>
        > {
            std::move(*this),
            std::forward<F>(f),
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
            task_tree_builder,
            Level,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            detail::no_fanout_policy_tag,
            adapter_chain<>
        > {
            std::move(*this),
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
            task_tree_builder,
            Level,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            std::decay_t<FanoutPolicy>,
            adapter_chain<>
        > {
            std::move(*this),
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
            task_tree_builder,
            Level,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            detail::no_fanout_policy_tag,
            std::decay_t<AdapterChain>
        > {
            std::move(*this),
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
            task_tree_builder,
            Level,
            std::decay_t<F>,
            std::decay_t<ReceiverSpec>,
            std::decay_t<FanoutPolicy>,
            std::decay_t<AdapterChain>
        > {
            std::move(*this),
            std::forward<F>(f),
            std::forward<ReceiverSpec>(receiver_spec),
            std::forward<FanoutPolicy>(fanout_policy),
            std::forward<AdapterChain>(adapter_specs)
        };
    }

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

inline constexpr task_tree_builder<> task_tree {};

} // namespace yorch
