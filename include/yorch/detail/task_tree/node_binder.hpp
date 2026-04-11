#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

#include "../../bind.hpp"

namespace yorch::detail {

struct no_fanout_policy_tag {};

template <typename BuilderStorage, std::size_t Level, typename FanoutPolicy>
struct tree_node_binder_base {
    BuilderStorage builder;
    FanoutPolicy fanout_policy;

protected:
    template <typename Self>
    [[nodiscard]] static constexpr decltype(auto) builder_ref(Self&& self) {
        auto&& forwarded_self = std::forward<Self>(self);
        using builder_storage_t = std::remove_cvref_t<decltype(forwarded_self.builder)>;

        if constexpr (std::is_pointer_v<builder_storage_t>) {
            return *forwarded_self.builder;
        } else {
            return forward_member<Self>(forwarded_self.builder);
        }
    }

};

template <typename BuilderStorage, std::size_t Level, typename F, typename FanoutPolicy, typename AdapterChain>
struct tree_node_binder : tree_node_binder_base<BuilderStorage, Level, FanoutPolicy> {
    F func;
    AdapterChain adapter_specs;

    constexpr tree_node_binder(
        BuilderStorage builder,
        F func,
        FanoutPolicy fanout_policy,
        AdapterChain adapter_specs)
        : tree_node_binder_base<BuilderStorage, Level, FanoutPolicy> {
              std::move(builder),
              std::move(fanout_policy)}
        , func(std::move(func))
        , adapter_specs(std::move(adapter_specs)) {}

private:
    template <typename Self, typename Task>
    static constexpr auto append_task(Self&& self, Task&& task) {
        if constexpr (std::is_same_v<FanoutPolicy, no_fanout_policy_tag>) {
            return tree_node_binder_base<BuilderStorage, Level, FanoutPolicy>::builder_ref(std::forward<Self>(self))
                .template node<Level>(std::forward<Task>(task));
        } else {
            return tree_node_binder_base<BuilderStorage, Level, FanoutPolicy>::builder_ref(std::forward<Self>(self))
                .template node<Level>(
                    std::forward<Task>(task),
                    forward_member<Self>(self.fanout_policy));
        }
    }

public:

    template <typename... Specs>
        requires bind_signature_matches<F, Specs...>
    constexpr auto operator()(Specs&&... specs) const& {
        return this->append_task(
            *this,
            yorch::task(func, adapter_specs)(std::forward<Specs>(specs)...));
    }

    template <typename... Specs>
        requires bind_signature_matches<F, Specs...>
    constexpr auto operator()(Specs&&... specs) && {
        return this->append_task(
            std::move(*this),
            yorch::task(std::move(func), std::move(adapter_specs))(std::forward<Specs>(specs)...));
    }
};

template <typename BuilderStorage, std::size_t Level, typename F, typename FanoutPolicy, typename AdapterChain>
struct tree_node_into_binder : tree_node_binder_base<BuilderStorage, Level, FanoutPolicy> {
    F func;
    AdapterChain adapter_specs;

    constexpr tree_node_into_binder(
        BuilderStorage builder,
        F func,
        FanoutPolicy fanout_policy,
        AdapterChain adapter_specs)
        : tree_node_binder_base<BuilderStorage, Level, FanoutPolicy> {
              std::move(builder),
              std::move(fanout_policy)}
        , func(std::move(func))
        , adapter_specs(std::move(adapter_specs)) {}

private:
    template <typename Self, typename Task>
    static constexpr auto append_task(Self&& self, Task&& task) {
        if constexpr (std::is_same_v<FanoutPolicy, no_fanout_policy_tag>) {
            return tree_node_binder_base<BuilderStorage, Level, FanoutPolicy>::builder_ref(std::forward<Self>(self))
                .template node_into<Level>(std::forward<Task>(task));
        } else {
            return tree_node_binder_base<BuilderStorage, Level, FanoutPolicy>::builder_ref(std::forward<Self>(self))
                .template node_into<Level>(
                    std::forward<Task>(task),
                    forward_member<Self>(self.fanout_policy));
        }
    }

public:

    template <typename... Specs>
        requires inferred_bind_into_signature_matches<F, Specs...>
    constexpr auto operator()(Specs&&... specs) const& {
        return this->append_task(
            *this,
            yorch::task_into(func, adapter_specs)(std::forward<Specs>(specs)...));
    }

    template <typename... Specs>
        requires inferred_bind_into_signature_matches<F, Specs...>
    constexpr auto operator()(Specs&&... specs) && {
        return this->append_task(
            std::move(*this),
            yorch::task_into(std::move(func), std::move(adapter_specs))(std::forward<Specs>(specs)...));
    }
};

} // namespace yorch::detail
