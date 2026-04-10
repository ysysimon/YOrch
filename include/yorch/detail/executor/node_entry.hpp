#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

#include "../../context.hpp"
#include "../../executor/concepts.hpp" // IWYU pragma: keep
#include "../../result.hpp"
#include "result.hpp"

namespace yorch::detail {

template <typename T>
struct exec_context_traits;

template <typename Ctx, typename Prev>
struct exec_context_traits<exec_context<Ctx, Prev>> {
    using ctx_type = Ctx;
    using prev_type = Prev;
};

template <typename Ctx>
[[nodiscard]] constexpr auto make_exec_context(Ctx& ctx, no_prev) noexcept
    -> exec_context<Ctx> {
    return {ctx};
}

template <typename Ctx, typename Prev>
[[nodiscard]] constexpr auto make_exec_context(Ctx& ctx, Prev prev) noexcept
    -> exec_context<Ctx, Prev> {
    return {ctx, prev};
}

[[nodiscard]] constexpr auto make_exec_context(no_prev) noexcept
    -> exec_context<void> {
    return {};
}

template <typename Prev>
[[nodiscard]] constexpr auto make_exec_context(Prev prev) noexcept
    -> exec_context<void, Prev> {
    return {prev};
}

template <typename Plan, std::size_t I, typename Slots>
[[nodiscard]] constexpr auto make_node_prev_view(Slots& slots) {
    if constexpr (Plan::template parent_index<I> == Plan::no_parent) {
        return no_prev {};
    } else {
        constexpr auto parent = Plan::template parent_index<I>;

        if constexpr (std::is_void_v<typename Plan::template output_type<parent>>) {
            return no_prev {};
        } else if constexpr (Plan::template child_count<parent> > 1) {
            return std::as_const(slots).template prev_view_for<parent>();
        } else {
            return slots.template prev_view_for<parent>();
        }
    }
}

template <typename Plan, typename Slots, std::size_t I>
using node_prev_view_t =
    decltype(make_node_prev_view<Plan, I>(std::declval<Slots&>()));

struct node_enter_result {
    step_result step = step_result::success();
    bool payload_live = false;
};

template <std::size_t I, typename Plan, typename Slots, typename Ec>
[[nodiscard]] constexpr node_enter_result enter_node_with_exec_context(
    Plan& plan,
    Slots& slots,
    Ec& ec) {
    using task_t = typename Plan::template task_type<I>;
    using exec_traits_t = exec_context_traits<std::remove_cvref_t<Ec>>;
    using raw_result_t = typename Plan::template raw_result_type<I>;

    static_assert(
        executable_task<
            task_t&,
            typename exec_traits_t::ctx_type,
            typename exec_traits_t::prev_type> ||
        executable_direct_output_task<
            task_t&,
            typename exec_traits_t::ctx_type,
            typename exec_traits_t::prev_type>,
        "Plan nodes executed through run_plan(...) must expose a noexcept invoke_raw(exec_context<...>&) or invoke_into(exec_context<...>&, direct_out<...>) surface");

    auto& task = plan.template entry<I>().task;
    node_enter_result result {};

    if constexpr (executable_direct_output_task<
                      task_t&,
                      typename exec_traits_t::ctx_type,
                      typename exec_traits_t::prev_type>) {
        result.step = finalize_direct_output_step<I>(
            slots,
            task.invoke_into(ec, slots.template out<I>()));
        result.payload_live = result.step.ok();
    } else if constexpr (std::is_void_v<raw_result_t>) {
        task.invoke_raw(ec);
    } else {
        auto raw = task.invoke_raw(ec);
        result.step = extract_step_result(raw);

        if (result.step.ok()) {
            result.payload_live = store_node_output<I>(slots, raw);
        }
    }

    return result;
}

template <std::size_t I, typename Plan, typename Slots>
[[nodiscard]] constexpr node_enter_result enter_node(Plan& plan, Slots& slots) {
    auto prev = make_node_prev_view<Plan, I>(slots);
    auto ec = make_exec_context(prev);
    return enter_node_with_exec_context<I>(plan, slots, ec);
}

template <std::size_t I, typename Plan, typename Slots, typename Ctx>
[[nodiscard]] constexpr node_enter_result enter_node(Plan& plan, Slots& slots, Ctx& ctx) {
    auto prev = make_node_prev_view<Plan, I>(slots);
    auto ec = make_exec_context(ctx, prev);
    return enter_node_with_exec_context<I>(plan, slots, ec);
}

} // namespace yorch::detail
