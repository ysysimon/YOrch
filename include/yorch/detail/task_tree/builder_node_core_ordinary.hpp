#pragma once

#include <tuple>
#include <utility>

#include "concepts.hpp" // IWYU pragma: keep
#include "metadata.hpp"
#include "node_binder.hpp"

namespace yorch {
template <typename... Nodes>
struct task_tree_builder;
}

namespace yorch::detail {

template <typename Derived, typename... Nodes>
struct builder_node_core_ordinary {
    template <std::size_t Level, typename Task>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_task_object_argument<Task>
    [[nodiscard]] constexpr auto node(Task&& task) const& {
        using next_node_t = detail::task_tree_node_t<Level, Task>;
        return task_tree_builder<Nodes..., next_node_t> {
            std::tuple_cat(static_cast<const Derived&>(*this).nodes, std::tuple<next_node_t> {next_node_t {std::forward<Task>(task)}})
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
            std::tuple_cat(static_cast<const Derived&>(*this).nodes, std::tuple<next_node_t> {next_node_t {std::forward<Task>(task)}})
        };
    }

    template <std::size_t Level, typename Task>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_task_object_argument<Task>
    [[nodiscard]] constexpr auto node_into(Task&& task) const& {
        using next_node_t = detail::task_tree_node_t<Level, Task>;
        return task_tree_builder<Nodes..., next_node_t> {
            std::tuple_cat(static_cast<const Derived&>(*this).nodes, std::tuple<next_node_t> {next_node_t {std::forward<Task>(task)}})
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
            std::tuple_cat(static_cast<const Derived&>(*this).nodes, std::tuple<next_node_t> {next_node_t {std::forward<Task>(task)}})
        };
    }

    template <std::size_t Level, typename F>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_callable_task_argument<F>
    [[nodiscard]] constexpr auto node(F&& f) const& {
        return detail::tree_node_binder<
            const Derived*,
            Level,
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

    template <std::size_t Level, typename F, typename FanoutPolicy>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_callable_task_argument<F> &&
                 detail::fanout_policy<FanoutPolicy>
    [[nodiscard]] constexpr auto node(F&& f, FanoutPolicy&& fanout_policy) const& {
        return detail::tree_node_binder<
            const Derived*,
            Level,
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

    template <std::size_t Level, typename F, typename AdapterChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_callable_task_argument<F> &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto node(F&& f, AdapterChain&& adapter_specs) const& {
        return detail::tree_node_binder<
            const Derived*,
            Level,
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

    template <std::size_t Level, typename F, typename FanoutPolicy, typename AdapterChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_callable_task_argument<F> &&
                 detail::fanout_policy<FanoutPolicy> &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto node(F&& f, FanoutPolicy&& fanout_policy, AdapterChain&& adapter_specs) const& {
        return detail::tree_node_binder<
            const Derived*,
            Level,
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
    template <std::size_t Level, typename F>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_callable_task_argument<F>
    [[nodiscard]] constexpr auto node_into(F&& f) const& {
        return detail::tree_node_into_binder<
            const Derived*,
            Level,
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

    template <std::size_t Level, typename F, typename FanoutPolicy>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_callable_task_argument<F> &&
                 detail::fanout_policy<FanoutPolicy>
    [[nodiscard]] constexpr auto node_into(F&& f, FanoutPolicy&& fanout_policy) const& {
        return detail::tree_node_into_binder<
            const Derived*,
            Level,
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

    template <std::size_t Level, typename F, typename AdapterChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_callable_task_argument<F> &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto node_into(F&& f, AdapterChain&& adapter_specs) const& {
        return detail::tree_node_into_binder<
            const Derived*,
            Level,
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

    template <std::size_t Level, typename F, typename FanoutPolicy, typename AdapterChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_callable_task_argument<F> &&
                 detail::fanout_policy<FanoutPolicy> &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto node_into(F&& f, FanoutPolicy&& fanout_policy, AdapterChain&& adapter_specs) const& {
        return detail::tree_node_into_binder<
            const Derived*,
            Level,
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
    template <std::size_t Level, typename Task>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_task_object_argument<Task>
    [[nodiscard]] constexpr auto node(Task&& task) && {
        using next_node_t = detail::task_tree_node_t<Level, Task>;
        return task_tree_builder<Nodes..., next_node_t> {
            std::tuple_cat(std::move(static_cast<Derived&>(*this).nodes), std::tuple<next_node_t> {next_node_t {std::forward<Task>(task)}})
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
            std::tuple_cat(std::move(static_cast<Derived&>(*this).nodes), std::tuple<next_node_t> {next_node_t {std::forward<Task>(task)}})
        };
    }

    template <std::size_t Level, typename Task>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_task_object_argument<Task>
    [[nodiscard]] constexpr auto node_into(Task&& task) && {
        using next_node_t = detail::task_tree_node_t<Level, Task>;
        return task_tree_builder<Nodes..., next_node_t> {
            std::tuple_cat(std::move(static_cast<Derived&>(*this).nodes), std::tuple<next_node_t> {next_node_t {std::forward<Task>(task)}})
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
            std::tuple_cat(std::move(static_cast<Derived&>(*this).nodes), std::tuple<next_node_t> {next_node_t {std::forward<Task>(task)}})
        };
    }

    template <std::size_t Level, typename F>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_callable_task_argument<F>
    [[nodiscard]] constexpr auto node(F&& f) && {
        return detail::tree_node_binder<
            Derived,
            Level,
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

    template <std::size_t Level, typename F, typename FanoutPolicy>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_callable_task_argument<F> &&
                 detail::fanout_policy<FanoutPolicy>
    [[nodiscard]] constexpr auto node(F&& f, FanoutPolicy&& fanout_policy) && {
        return detail::tree_node_binder<
            Derived,
            Level,
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

    template <std::size_t Level, typename F, typename AdapterChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_callable_task_argument<F> &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto node(F&& f, AdapterChain&& adapter_specs) && {
        return detail::tree_node_binder<
            Derived,
            Level,
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

    template <std::size_t Level, typename F, typename FanoutPolicy, typename AdapterChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::ordinary_callable_task_argument<F> &&
                 detail::fanout_policy<FanoutPolicy> &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto node(F&& f, FanoutPolicy&& fanout_policy, AdapterChain&& adapter_specs) && {
        return detail::tree_node_binder<
            Derived,
            Level,
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
    template <std::size_t Level, typename F>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_callable_task_argument<F>
    [[nodiscard]] constexpr auto node_into(F&& f) && {
        return detail::tree_node_into_binder<
            Derived,
            Level,
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

    template <std::size_t Level, typename F, typename FanoutPolicy>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_callable_task_argument<F> &&
                 detail::fanout_policy<FanoutPolicy>
    [[nodiscard]] constexpr auto node_into(F&& f, FanoutPolicy&& fanout_policy) && {
        return detail::tree_node_into_binder<
            Derived,
            Level,
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

    template <std::size_t Level, typename F, typename AdapterChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_callable_task_argument<F> &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto node_into(F&& f, AdapterChain&& adapter_specs) && {
        return detail::tree_node_into_binder<
            Derived,
            Level,
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

    template <std::size_t Level, typename F, typename FanoutPolicy, typename AdapterChain>
        requires (detail::append_level_valid_v<Level, Nodes...>) &&
                 detail::direct_output_callable_task_argument<F> &&
                 detail::fanout_policy<FanoutPolicy> &&
                 detail::adapter_chain_like<AdapterChain>
    [[nodiscard]] constexpr auto node_into(F&& f, FanoutPolicy&& fanout_policy, AdapterChain&& adapter_specs) && {
        return detail::tree_node_into_binder<
            Derived,
            Level,
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
