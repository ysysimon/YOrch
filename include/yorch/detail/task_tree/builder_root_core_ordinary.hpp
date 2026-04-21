#pragma once

#include <utility>

#include "concepts.hpp" // IWYU pragma: keep
#include "node_binder.hpp"

namespace yorch::detail {

template <typename Derived, typename... Nodes>
struct builder_root_core_ordinary {
    template <typename Task>
        requires (sizeof...(Nodes) == 0) &&
                 detail::ordinary_task_object_argument<Task>
    [[nodiscard]] constexpr auto root(Task&& task) const& {
        return static_cast<const Derived&>(*this).template node<0>(std::forward<Task>(task));
    }

    template <typename Task, typename FanoutPolicy>
        requires (sizeof...(Nodes) == 0) &&
                 detail::ordinary_task_object_argument<Task> &&
                 detail::fanout_policy<FanoutPolicy>
    [[nodiscard]] constexpr auto root(Task&& task, FanoutPolicy&& fanout_policy) const& {
        return static_cast<const Derived&>(*this).template node<0>(std::forward<Task>(task), std::forward<FanoutPolicy>(fanout_policy));
    }

    template <typename F>
        requires (sizeof...(Nodes) == 0) &&
                 detail::ordinary_callable_task_argument<F>
    [[nodiscard]] constexpr auto root(F&& f) const& {
        return detail::tree_node_binder<
            const Derived*,
            0,
            std::decay_t<F>,
            detail::no_fanout_policy_tag,
            adapter_chain<>
        > {
            static_cast<const Derived*>(this),
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
            const Derived*,
            0,
            std::decay_t<F>,
            std::decay_t<FanoutPolicy>,
            adapter_chain<>
        > {
            static_cast<const Derived*>(this),
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
            const Derived*,
            0,
            std::decay_t<F>,
            detail::no_fanout_policy_tag,
            std::decay_t<AdapterChain>
        > {
            static_cast<const Derived*>(this),
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
            const Derived*,
            0,
            std::decay_t<F>,
            std::decay_t<FanoutPolicy>,
            std::decay_t<AdapterChain>
        > {
            static_cast<const Derived*>(this),
            std::forward<F>(f),
            std::forward<FanoutPolicy>(fanout_policy),
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <typename Task>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_task_object_argument<Task>
    [[nodiscard]] constexpr auto root_into(Task&& task) const& {
        return static_cast<const Derived&>(*this).template node_into<0>(std::forward<Task>(task));
    }

    template <typename Task, typename FanoutPolicy>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_task_object_argument<Task> &&
                 detail::fanout_policy<FanoutPolicy>
    [[nodiscard]] constexpr auto root_into(Task&& task, FanoutPolicy&& fanout_policy) const& {
        return static_cast<const Derived&>(*this).template node_into<0>(std::forward<Task>(task), std::forward<FanoutPolicy>(fanout_policy));
    }

    template <typename F>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_callable_task_argument<F>
    [[nodiscard]] constexpr auto root_into(F&& f) const& {
        return detail::tree_node_into_binder<
            const Derived*,
            0,
            std::decay_t<F>,
            detail::no_fanout_policy_tag,
            adapter_chain<>
        > {
            static_cast<const Derived*>(this),
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
            const Derived*,
            0,
            std::decay_t<F>,
            std::decay_t<FanoutPolicy>,
            adapter_chain<>
        > {
            static_cast<const Derived*>(this),
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
            const Derived*,
            0,
            std::decay_t<F>,
            detail::no_fanout_policy_tag,
            std::decay_t<AdapterChain>
        > {
            static_cast<const Derived*>(this),
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
            const Derived*,
            0,
            std::decay_t<F>,
            std::decay_t<FanoutPolicy>,
            std::decay_t<AdapterChain>
        > {
            static_cast<const Derived*>(this),
            std::forward<F>(f),
            std::forward<FanoutPolicy>(fanout_policy),
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <typename Task>
        requires (sizeof...(Nodes) == 0) &&
                 detail::ordinary_task_object_argument<Task>
    [[nodiscard]] constexpr auto root(Task&& task) && {
        return static_cast<Derived&&>(*this).template node<0>(std::forward<Task>(task));
    }

    template <typename Task, typename FanoutPolicy>
        requires (sizeof...(Nodes) == 0) &&
                 detail::ordinary_task_object_argument<Task> &&
                 detail::fanout_policy<FanoutPolicy>
    [[nodiscard]] constexpr auto root(Task&& task, FanoutPolicy&& fanout_policy) && {
        return static_cast<Derived&&>(*this).template node<0>(std::forward<Task>(task), std::forward<FanoutPolicy>(fanout_policy));
    }

    template <typename F>
        requires (sizeof...(Nodes) == 0) &&
                 detail::ordinary_callable_task_argument<F>
    [[nodiscard]] constexpr auto root(F&& f) && {
        return detail::tree_node_binder<
            Derived,
            0,
            std::decay_t<F>,
            detail::no_fanout_policy_tag,
            adapter_chain<>
        > {
            static_cast<Derived&&>(*this),
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
            Derived,
            0,
            std::decay_t<F>,
            std::decay_t<FanoutPolicy>,
            adapter_chain<>
        > {
            static_cast<Derived&&>(*this),
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
            Derived,
            0,
            std::decay_t<F>,
            detail::no_fanout_policy_tag,
            std::decay_t<AdapterChain>
        > {
            static_cast<Derived&&>(*this),
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
            Derived,
            0,
            std::decay_t<F>,
            std::decay_t<FanoutPolicy>,
            std::decay_t<AdapterChain>
        > {
            static_cast<Derived&&>(*this),
            std::forward<F>(f),
            std::forward<FanoutPolicy>(fanout_policy),
            std::forward<AdapterChain>(adapter_specs)
        };
    }

    template <typename Task>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_task_object_argument<Task>
    [[nodiscard]] constexpr auto root_into(Task&& task) && {
        return static_cast<Derived&&>(*this).template node_into<0>(std::forward<Task>(task));
    }

    template <typename Task, typename FanoutPolicy>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_task_object_argument<Task> &&
                 detail::fanout_policy<FanoutPolicy>
    [[nodiscard]] constexpr auto root_into(Task&& task, FanoutPolicy&& fanout_policy) && {
        return static_cast<Derived&&>(*this).template node_into<0>(std::forward<Task>(task), std::forward<FanoutPolicy>(fanout_policy));
    }

    template <typename F>
        requires (sizeof...(Nodes) == 0) &&
                 detail::direct_output_callable_task_argument<F>
    [[nodiscard]] constexpr auto root_into(F&& f) && {
        return detail::tree_node_into_binder<
            Derived,
            0,
            std::decay_t<F>,
            detail::no_fanout_policy_tag,
            adapter_chain<>
        > {
            static_cast<Derived&&>(*this),
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
            Derived,
            0,
            std::decay_t<F>,
            std::decay_t<FanoutPolicy>,
            adapter_chain<>
        > {
            static_cast<Derived&&>(*this),
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
            Derived,
            0,
            std::decay_t<F>,
            detail::no_fanout_policy_tag,
            std::decay_t<AdapterChain>
        > {
            static_cast<Derived&&>(*this),
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
            Derived,
            0,
            std::decay_t<F>,
            std::decay_t<FanoutPolicy>,
            std::decay_t<AdapterChain>
        > {
            static_cast<Derived&&>(*this),
            std::forward<F>(f),
            std::forward<FanoutPolicy>(fanout_policy),
            std::forward<AdapterChain>(adapter_specs)
        };
    }
};

} // namespace yorch::detail
