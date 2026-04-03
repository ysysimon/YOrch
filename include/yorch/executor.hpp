#pragma once
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "bind.hpp"
#include "slots.hpp"
#include "task_adapters.hpp"

namespace yorch {

/**
 * @brief Describes the main execution protocol accepted by `run_task(...)`.
 *
 * A task is executable on the main path only when it exposes
 * `invoke_raw(exec_context<...>&)` and that call is itself `noexcept`.
 * Throwing tasks are intentionally excluded from this concept; they must be
 * adapted first, for example with `catch_as_failure(...)`, before they can be
 * handed to the executor.
 *
 * @tparam Task Task object type.
 * @tparam Ctx Context schema.
 * @tparam Prev Direct-parent slot view type.
 */
template <typename Task, typename Ctx, typename Prev = no_prev>
concept executable_task =
    requires(Task&& task, exec_context<Ctx, Prev>& ec) {
        std::forward<Task>(task).invoke_raw(ec);
    } &&
    detail::declared_task_raw_result_matches_invoke_raw_v<Task, Ctx, Prev> &&
    requires(Task&& task, exec_context<Ctx, Prev>& ec) {
        requires noexcept(std::forward<Task>(task).invoke_raw(ec));
    };

template <typename Task, typename Ctx, typename Prev = no_prev>
concept executable_direct_output_task =
    direct_output_task<Task, Ctx, Prev> &&
    requires(
        Task&& task,
        exec_context<Ctx, Prev>& ec,
        detail::typed_slot<detail::declared_task_output_t<Task>>& slot) {
        {
            std::forward<Task>(task).invoke_into(
                ec,
                result_out<detail::declared_task_output_t<Task>> {slot})
        } -> std::same_as<step_result>;
        requires noexcept(std::forward<Task>(task).invoke_into(
            ec,
            result_out<detail::declared_task_output_t<Task>> {slot}));
    };

namespace detail {

template <typename T>
struct exec_context_traits;

template <typename Ctx, typename Prev>
struct exec_context_traits<exec_context<Ctx, Prev>> {
    using ctx_type = Ctx;
    using prev_type = Prev;
};

/**
 * @brief Normalizes a raw task return into scheduler-facing `step_result`.
 *
 * Plain value payloads are treated as successful completion, `step_result`
 * passes through unchanged, and `task_result<T>` contributes only its control
 * status here. Payload extraction itself is handled by later execution stages.
 *
 * @tparam R Raw return object type.
 * @param r Raw return object emitted by `invoke_raw(...)`.
 * @return Normalized `step_result`.
 */
template <typename R>
[[nodiscard]] constexpr step_result normalize_task_result(R&& r) { // NOLINT(readability-identifier-length)
    using raw_t = std::remove_cvref_t<R>;

    if constexpr (std::is_same_v<raw_t, step_result>) {
        return std::forward<R>(r);
    } else if constexpr (is_task_result_v<raw_t>) {
        return std::forward<R>(r).step;
    } else {
        static_cast<void>(r);
        return step_result::success();
    }
}

template <typename Spec>
struct is_from_prev_spec : std::false_type {};

template <typename T>
struct is_from_prev_spec<from_prev_t<T>> : std::true_type {};

template <typename Spec>
inline constexpr bool is_from_prev_spec_v =
    is_from_prev_spec<std::remove_cvref_t<Spec>>::value;

template <typename Arg>
inline constexpr bool exact_const_lvalue_ref_v =
    std::is_lvalue_reference_v<Arg> &&
    std::is_const_v<std::remove_reference_t<Arg>>;

/**
 * @brief Checks whether every `from_prev(...)` binding inside a `bound_task`
 * maps to an exact `const T&` callable parameter.
 *
 * This helper is used by the current fan-out validation path. When a parent
 * node has multiple direct children, those children may only read the parent
 * payload through `from_prev(...)`; they must not take a mutable reference,
 * take the value by copy, or request an rvalue-style consuming parameter.
 *
 * The check is performed position-by-position across the bound task's spec
 * tuple and callable signature:
 *
 * - if the `I`-th spec is not `from_prev(...)`, that parameter position is
 *   irrelevant to this validation and therefore accepted
 * - if the `I`-th spec is `from_prev(...)`, then the callable's `I`-th
 *   parameter type must be exactly `const T&`
 *
 * The final result is the logical AND of all parameter positions.
 *
 * @tparam F Callable type stored inside `bound_task`.
 * @tparam SpecsTuple Tuple type storing one bound spec per callable parameter.
 * @tparam I Compile-time parameter indices used to walk the callable/spec pair.
 * @param Unused compile-time index sequence selecting the parameter positions.
 * @return `true` when all `from_prev(...)` positions bind as exact
 * `const T&`; otherwise `false`.
 */
template <typename F, typename SpecsTuple, std::size_t... I>
[[nodiscard]] consteval bool bound_task_fanout_prev_valid_impl(std::index_sequence<I...>) {
    return (((!is_from_prev_spec_v<std::tuple_element_t<I, SpecsTuple>>) ||
             exact_const_lvalue_ref_v<nth_arg_t<I, F>>) && ...);
}

template <typename Task>
struct fanout_prev_task_valid : std::true_type {};

template <typename F, typename... Specs>
struct fanout_prev_task_valid<bound_task<F, Specs...>>
    : std::bool_constant<
          bound_task_fanout_prev_valid_impl<
              std::remove_cvref_t<F>,
              std::tuple<Specs...>>(
              std::index_sequence_for<Specs...> {})> {};

template <typename F, typename T, typename... Specs>
struct fanout_prev_task_valid<bound_output_task<F, T, Specs...>>
    : std::bool_constant<
          bound_task_fanout_prev_valid_impl<
              std::remove_cvref_t<F>,
              std::tuple<Specs...>>(
              std::index_sequence_for<Specs...> {})> {};

template <typename Task>
struct fanout_prev_task_valid<catch_failure_task<Task>>
    : fanout_prev_task_valid<Task> {};

template <typename Task, typename Policy>
struct fanout_prev_task_valid<catch_failure_with_policy_task<Task, Policy>>
    : fanout_prev_task_valid<Task> {};

template <typename Task, typename Policy>
struct fanout_prev_task_valid<retry_task<Task, Policy>>
    : fanout_prev_task_valid<Task> {};

template <typename Task>
inline constexpr bool fanout_prev_task_valid_v =
    fanout_prev_task_valid<std::remove_cvref_t<Task>>::value;

template <typename SpecsTuple, std::size_t... I>
[[nodiscard]] consteval bool bound_task_uses_from_prev_impl(std::index_sequence<I...>) {
    return (is_from_prev_spec_v<std::tuple_element_t<I, SpecsTuple>> || ...);
}

template <typename Task>
struct task_uses_from_prev : std::false_type {};

template <typename F, typename... Specs>
struct task_uses_from_prev<bound_task<F, Specs...>>
    : std::bool_constant<
          bound_task_uses_from_prev_impl<
              std::tuple<Specs...>>(
              std::index_sequence_for<Specs...> {})> {};

template <typename F, typename T, typename... Specs>
struct task_uses_from_prev<bound_output_task<F, T, Specs...>>
    : std::bool_constant<
          bound_task_uses_from_prev_impl<
              std::tuple<Specs...>>(
              std::index_sequence_for<Specs...> {})> {};

template <typename Task>
struct task_uses_from_prev<catch_failure_task<Task>>
    : task_uses_from_prev<Task> {};

template <typename Task, typename Policy>
struct task_uses_from_prev<catch_failure_with_policy_task<Task, Policy>>
    : task_uses_from_prev<Task> {};

template <typename Task, typename Policy>
struct task_uses_from_prev<retry_task<Task, Policy>>
    : task_uses_from_prev<Task> {};

template <typename Task>
inline constexpr bool task_uses_from_prev_v =
    task_uses_from_prev<std::remove_cvref_t<Task>>::value;

/**
 * @brief Minimal static-plan protocol required by the fan-out `from_prev`
 * validation path for a specific node index.
 *
 * This intentionally checks only the plan surface consumed by
 * `node_fanout_prev_valid()`: a root sentinel, per-node parent lookup,
 * per-node task type exposure, and per-node child count exposure.
 *
 * The concept is purposefully interface-based rather than tied to
 * `compiled_plan<...>` so future plan backends can reuse the same validation
 * logic as long as they expose the same compile-time protocol.
 *
 * @tparam Plan Candidate plan type.
 * @tparam I Node index being validated.
 */
template <typename Plan, std::size_t I>
concept fanout_validatable_plan_node =
    requires {
        { Plan::no_parent } -> std::convertible_to<std::size_t>;
        { Plan::template parent_index<I> } -> std::convertible_to<std::size_t>;
        typename Plan::template task_type<I>;
    };

/**
 * @brief Checks whether node `I` is structurally allowed to use
 * `from_prev(...)` at all.
 *
 * This validation is intentionally narrower than full `from_prev` type
 * checking. It only answers the question "does this node have a readable
 * direct-parent payload source?" and therefore rejects two cases early at the
 * `run_plan(...)` boundary:
 *
 * - the root node uses `from_prev(...)`
 * - the direct parent exists but its compiled output type is `void`
 *
 * It does not attempt to validate the requested payload type `T` inside
 * `from_prev<T>()`, nor whether that `T` matches the direct parent payload
 * exactly. Those finer-grained binding checks remain in the resolution layer,
 * so this helper stays focused on plan-structure legality instead of becoming
 * a full duplicate of `resolve_as(from_prev_t<...>, ...)`.
 *
 * @tparam Plan Compiled plan type being validated.
 * @tparam I Node index within `Plan`.
 * @return `true` when node `I` either does not use `from_prev(...)` or has a
 * non-void direct parent payload source; otherwise `false`.
 */
template <typename Plan, std::size_t I>
    requires fanout_validatable_plan_node<Plan, I>
[[nodiscard]] consteval bool node_prev_source_valid() {
    using task_t = typename Plan::template task_type<I>;

    if constexpr (Plan::template parent_index<I> == Plan::no_parent) {
        return !task_uses_from_prev_v<task_t>;
    } else {
        constexpr auto parent = Plan::template parent_index<I>;

        if constexpr (std::is_void_v<typename Plan::template output_type<parent>>) {
            return !task_uses_from_prev_v<task_t>;
        } else {
            return true;
        }
    }
}

template <typename Plan, std::size_t I>
    requires fanout_validatable_plan_node<Plan, I>
[[nodiscard]] consteval bool node_fanout_prev_valid() {
    if constexpr (Plan::template parent_index<I> == Plan::no_parent) {
        return true;
    } else {
        constexpr auto parent = Plan::template parent_index<I>;

        if constexpr (Plan::template child_count<parent> <= 1) {
            return true;
        } else {
            return fanout_prev_task_valid_v<typename Plan::template task_type<I>>;
        }
    }
}

template <typename Plan, std::size_t... I>
    requires (fanout_validatable_plan_node<Plan, I> && ...)
[[nodiscard]] consteval bool plan_prev_source_valid_impl(std::index_sequence<I...>) {
    return (node_prev_source_valid<Plan, I>() && ...);
}

template <typename Plan, std::size_t... I>
    requires (fanout_validatable_plan_node<Plan, I> && ...)
[[nodiscard]] consteval bool plan_fanout_prev_valid_impl(std::index_sequence<I...>) {
    return (node_fanout_prev_valid<Plan, I>() && ...);
}

template <typename Plan>
inline constexpr bool plan_prev_source_valid_v =
    plan_prev_source_valid_impl<Plan>(std::make_index_sequence<Plan::node_count> {});

template <typename Plan>
inline constexpr bool plan_fanout_prev_valid_v =
    plan_fanout_prev_valid_impl<Plan>(std::make_index_sequence<Plan::node_count> {});

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

template <typename R>
[[nodiscard]] constexpr step_result extract_step_result(R&& r) { // NOLINT(readability-identifier-length)
    return normalize_task_result(std::forward<R>(r));
}

/**
 * @brief Materializes the payload portion of a node's raw task return into the
 * node's compiled slot, when that return category actually carries a value.
 *
 * This helper is the value-storage counterpart to `extract_step_result(...)`.
 * The executor first derives control-flow status from the raw return, then
 * uses this function to decide whether the current node produced a payload that
 * should remain alive for its subtree and therefore be written into slot `I`.
 *
 * The storage rules are:
 *
 * - `void` and `step_result` do not carry a payload, so no slot write happens
 * - plain `T` writes that returned value directly into slot `I`
 * - `task_result<T>` writes only the wrapped `value` and only when the step
 *   status is `success`; non-success `task_result<T>` values leave the slot
 *   empty
 *
 * Payloads are moved into the slot because the slot becomes the owning storage
 * for the remainder of the node's subtree traversal.
 *
 * @tparam I Node index whose slot should receive the payload.
 * @tparam Slots Concrete `plan_exec_slots<...>`-like storage type.
 * @tparam Raw Raw return object type produced by `invoke_raw(...)`.
 * @param slots Plan slot storage used by the current DFS execution.
 * @param raw Raw task return object to inspect and, when applicable, move from.
 */
template <std::size_t I, typename Slots, typename Raw>
[[nodiscard]] constexpr bool store_node_output(Slots& slots, Raw& raw) {
    using raw_t = std::remove_cvref_t<Raw>;

    if constexpr (std::is_void_v<raw_t> ||
                  std::is_same_v<raw_t, step_result>) {
        static_cast<void>(slots);
        static_cast<void>(raw);
        return false;
    } else if constexpr (is_task_result_v<raw_t>) {
        if (raw.step.ok()) {
            YORCH_ASSERT(raw.has_value() &&
                         "Successful task_result<T> must carry a payload before slot storage");
            slots.template emplace<I>(std::move(raw).value());
            return true;
        }

        return false;
    } else {
        slots.template emplace<I>(std::move(raw));
        return true;
    }
}

template <typename Slots, std::size_t I>
struct node_slot_guard {
    Slots& slots;
    bool armed = false;

    constexpr explicit node_slot_guard(Slots& stored_slots) noexcept
        : slots(stored_slots) {}

    node_slot_guard(const node_slot_guard&) = delete;
    node_slot_guard& operator=(const node_slot_guard&) = delete;
    node_slot_guard(node_slot_guard&&) = delete;
    node_slot_guard& operator=(node_slot_guard&&) = delete;

    constexpr void arm() noexcept {
        armed = true;
    }

    ~node_slot_guard() {
        if (armed) {
            slots.template destroy<I>();
        }
    }
};

template <std::size_t I, typename Slots>
[[nodiscard]] constexpr step_result finalize_direct_output_step(Slots& slots, step_result step) {
    if (step.ok()) {
        YORCH_ASSERT(slots.template has_value<I>() &&
                     "Direct-output tasks returning success must construct their payload");
    } else {
        slots.template destroy<I>();
    }

    return step;
}

template <std::size_t I, std::size_t Ord = 0, typename Plan, typename Slots>
[[nodiscard]] constexpr step_result run_children(Plan& plan, Slots& slots);

template <std::size_t I, std::size_t Ord = 0, typename Plan, typename Slots, typename Ctx>
[[nodiscard]] constexpr step_result run_children(Plan& plan, Slots& slots, Ctx& ctx);

/**
 * @brief Executes node `I` itself after `run_node(...)` has already assembled
 * the correct execution context for that node.
 *
 * This function is the "node body" of the DFS executor:
 *
 * - it invokes the current task with the ready-to-use `exec_context`
 * - it extracts control-flow state from the raw return
 * - it materializes any successful payload into slot `I`
 * - it delegates child traversal to the injected `invoke_children` callback
 *
 * The callback is supplied by `run_node(...)` so this helper stays focused on
 * node-local execution rather than knowing how siblings are enumerated. The
 * slot guard keeps the current node's payload alive for the entire subtree and
 * destroys it automatically on every exit path, including early returns.
 *
 * @tparam I Node index being executed.
 * @tparam Plan Compiled plan type.
 * @tparam Slots Concrete slot storage type for the active plan run.
 * @tparam Ec Concrete `exec_context<...>` type prepared for node `I`.
 * @tparam ChildInvoker Callable that continues DFS into node `I`'s children.
 * @param plan Mutable compiled plan containing the node task object.
 * @param slots Per-run slot storage that owns subtree-visible payloads.
 * @param ec Borrowed execution context already tailored to node `I`.
 * @param invoke_children Continuation that traverses node `I`'s children.
 * @return The resulting `step_result` for node `I` and, when reached, its
 * child subtree.
 */
template <std::size_t I, typename Plan, typename Slots, typename Ec, typename ChildInvoker>
[[nodiscard]] constexpr step_result run_node_body(
    Plan& plan,
    Slots& slots,
    Ec& ec,
    // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
    ChildInvoker&& invoke_children) {
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
        "Plan nodes executed through run_plan(...) must expose a noexcept invoke_raw(exec_context<...>&) or invoke_into(exec_context<...>&, result_out<...>) surface");

    auto& task = plan.template entry<I>().task;
    node_slot_guard<Slots, I> guard {slots};

    if constexpr (executable_direct_output_task<
                      task_t&,
                      typename exec_traits_t::ctx_type,
                      typename exec_traits_t::prev_type>) {
        const auto step = finalize_direct_output_step<I>(
            slots,
            task.invoke_into(ec, slots.template out<I>()));

        if (!step.ok()) {
            return step;
        }

        guard.arm();
    } else if constexpr (std::is_void_v<raw_result_t>) {
        task.invoke_raw(ec);
    } else {
        const auto step = [&]() constexpr -> step_result {
            auto raw = task.invoke_raw(ec);
            const auto current = extract_step_result(raw);

            if (current.ok()) {
                if (store_node_output<I>(slots, raw)) {
                    guard.arm();
                }
            }

            return current;
        }();

        if (!step.ok()) {
            return step;
        }
    }

    return invoke_children();
}

/**
 * @brief Enters node `I` in the serial DFS executor without an external
 * context object.
 *
 * `run_node(...)` is the layer that bridges static plan structure into a
 * concrete `exec_context` for one node. It computes the direct-parent view for
 * node `I`, packages that view into an `exec_context`, and then hands control
 * to `run_node_body(...)`.
 *
 * Depth-first descent happens because the node body later invokes
 * `run_children<I, 0>(...)`, which selects a child and recursively calls that
 * child's own `run_node<child>(...)`.
 *
 * @tparam I Node index being entered.
 * @tparam Plan Compiled plan type.
 * @tparam Slots Concrete slot storage type for the active plan run.
 * @param plan Mutable compiled plan containing the node task object.
 * @param slots Per-run slot storage that owns subtree-visible payloads.
 * @return The resulting `step_result` for node `I` and its subtree.
 */
template <std::size_t I, typename Plan, typename Slots>
[[nodiscard]] constexpr step_result run_node(Plan& plan, Slots& slots) {
    auto prev = make_node_prev_view<Plan, I>(slots);
    auto ec = make_exec_context(prev);

    return run_node_body<I>(
        plan,
        slots,
        ec,
        [&]() constexpr -> step_result {
            return run_children<I>(plan, slots);
        });
}

/**
 * @brief Enters node `I` in the serial DFS executor with an external typed
 * context object.
 *
 * This overload performs the same DFS entry work as the context-free version,
 * but also threads the caller-provided execution context through the node's
 * `exec_context<Ctx, Prev>`.
 *
 * @tparam I Node index being entered.
 * @tparam Plan Compiled plan type.
 * @tparam Slots Concrete slot storage type for the active plan run.
 * @tparam Ctx Borrowed context type exposed through `from_ctx(...)`.
 * @param plan Mutable compiled plan containing the node task object.
 * @param slots Per-run slot storage that owns subtree-visible payloads.
 * @param ctx Borrowed typed context object shared by this plan execution.
 * @return The resulting `step_result` for node `I` and its subtree.
 */
template <std::size_t I, typename Plan, typename Slots, typename Ctx>
[[nodiscard]] constexpr step_result run_node(Plan& plan, Slots& slots, Ctx& ctx) {
    auto prev = make_node_prev_view<Plan, I>(slots);
    auto ec = make_exec_context(ctx, prev);

    return run_node_body<I>(
        plan,
        slots,
        ec,
        [&]() constexpr -> step_result {
            return run_children<I>(plan, slots, ctx);
        });
}

/**
 * @brief Executes the direct children of node `I` one by one in DFS order,
 * starting at child ordinal `Ord`, without an external context object.
 *
 * This helper is the sibling-enumeration half of the DFS recursion. It does
 * not itself "go deeper" by incrementing `Ord`; the actual descent into a
 * child subtree happens at `run_node<child>(...)`. Advancing to
 * `run_children<I, Ord + 1>(...)` means "the current child has finished, move
 * on to the next sibling of the same parent."
 *
 * From the parent's point of view, both `success` and `abort_branch` mean the
 * current child subtree has ended and sibling traversal may continue. Other
 * non-success states stop the remaining DFS and propagate upward unchanged.
 *
 * @tparam I Parent node index whose child list is being traversed.
 * @tparam Ord Current child ordinal within node `I`'s direct-child list.
 * @tparam Plan Compiled plan type.
 * @tparam Slots Concrete slot storage type for the active plan run.
 * @param plan Mutable compiled plan containing the node task objects.
 * @param slots Per-run slot storage that owns subtree-visible payloads.
 * @return The aggregate `step_result` for the remaining children of node `I`.
 */
template <std::size_t I, std::size_t Ord, typename Plan, typename Slots>
[[nodiscard]] constexpr step_result run_children(Plan& plan, Slots& slots) {
    static_cast<void>(plan);
    static_cast<void>(slots);

    if constexpr (Ord == Plan::template child_count<I>) {
        return step_result::success();
    } else {
        constexpr auto child = Plan::template child_index<I, Ord>;

        const auto child_step = run_node<child>(plan, slots);

        if (child_step.status == step_status::abort_branch) {
            return run_children<I, Ord + 1>(plan, slots);
        }

        if (!child_step.ok()) {
            return child_step;
        }

        return run_children<I, Ord + 1>(plan, slots);
    }
}

/**
 * @brief Executes the direct children of node `I` one by one in DFS order,
 * starting at child ordinal `Ord`, with an external typed context object.
 *
 * This overload is identical in traversal semantics to the context-free
 * version and exists only to thread `ctx` through recursive child entry.
 *
 * @tparam I Parent node index whose child list is being traversed.
 * @tparam Ord Current child ordinal within node `I`'s direct-child list.
 * @tparam Plan Compiled plan type.
 * @tparam Slots Concrete slot storage type for the active plan run.
 * @tparam Ctx Borrowed context type exposed through `from_ctx(...)`.
 * @param plan Mutable compiled plan containing the node task objects.
 * @param slots Per-run slot storage that owns subtree-visible payloads.
 * @param ctx Borrowed typed context object shared by this plan execution.
 * @return The aggregate `step_result` for the remaining children of node `I`.
 */
template <std::size_t I, std::size_t Ord, typename Plan, typename Slots, typename Ctx>
[[nodiscard]] constexpr step_result run_children(Plan& plan, Slots& slots, Ctx& ctx) {
    static_cast<void>(plan);
    static_cast<void>(slots);
    static_cast<void>(ctx);

    if constexpr (Ord == Plan::template child_count<I>) {
        return step_result::success();
    } else {
        constexpr auto child = Plan::template child_index<I, Ord>;

        const auto child_step = run_node<child>(plan, slots, ctx);

        if (child_step.status == step_status::abort_branch) {
            return run_children<I, Ord + 1>(plan, slots, ctx);
        }

        if (!child_step.ok()) {
            return child_step;
        }

        return run_children<I, Ord + 1>(plan, slots, ctx);
    }
}

}  // namespace detail

/**
 * @brief Executes a ready-to-run task against the provided execution context.
 *
 * In the current design, `run_task(...)` remains thin: it resolves no
 * arguments by itself and only normalizes the raw return emitted by the task's
 * `invoke_raw(...)` protocol. Direct-output tasks that write into a slot-like
 * sink should instead use `run_task_into(...)`.
 *
 * The function intentionally accepts only the no-throw execution surface
 * modeled by `executable_task`; potentially throwing tasks must first be
 * wrapped into a no-throw adapter such as `catch_as_failure(...)`.
 *
 * @tparam Task Executable task type, typically `bound_task`.
 * @tparam Ctx Borrowed execution-context schema.
 * @param task Task object to execute.
 * @param ec Borrowed execution context.
 * @return The normalized `step_result` derived from the raw task return.
 */
template <typename Task, typename Ctx, typename Prev = no_prev>
    requires executable_task<Task, Ctx, Prev>
[[nodiscard]] constexpr step_result run_task(Task&& task, exec_context<Ctx, Prev>& ec)
    noexcept(
        noexcept(std::forward<Task>(task).invoke_raw(ec)) &&
        (std::is_void_v<detail::raw_task_result_t<Task&&, Ctx, Prev>> ||
         noexcept(detail::normalize_task_result(
             std::forward<Task>(task).invoke_raw(ec)))))
{
    using raw_result_t = detail::raw_task_result_t<Task&&, Ctx, Prev>;

    static_assert(detail::declared_task_raw_result_matches_invoke_raw_v<Task, Ctx, Prev>,
                  "Task raw_result_type must match invoke_raw(exec_context<...>&) return type");

    if constexpr (std::is_void_v<raw_result_t>) {
        std::forward<Task>(task).invoke_raw(ec);
        return step_result::success();
    } else {
        return detail::normalize_task_result(
            std::forward<Task>(task).invoke_raw(ec)
        );
    }
}

/**
 * @brief Executes a direct-output task against the provided execution context.
 *
 * Unlike `run_task(...)`, this overload is for tasks whose payload is written
 * into a caller-provided output sink instead of returned through `invoke_raw(...)`.
 * The caller owns the destination slot lifetime; this helper only enforces the
 * direct-output success/non-success contract on top of that sink.
 *
 * @tparam Task Direct-output task type exposing `invoke_into(...)`.
 * @tparam Ctx Borrowed execution-context schema.
 * @tparam Prev Direct-parent slot view type.
 * @param task Task object to execute.
 * @param ec Borrowed execution context.
 * @param out Output sink receiving the task payload on success.
 * @return The resulting `step_result` from the direct-output execution.
 */
template <typename Task, typename Ctx, typename Prev = no_prev>
    requires executable_direct_output_task<Task, Ctx, Prev>
[[nodiscard]] constexpr step_result run_task_into(
    Task&& task,
    exec_context<Ctx, Prev>& ec,
    result_out<detail::declared_task_output_t<Task>> out)
    noexcept(noexcept(std::forward<Task>(task).invoke_into(ec, out)))
{
    const auto step = std::forward<Task>(task).invoke_into(ec, out);

    if (step.ok()) {
        YORCH_ASSERT(out.has_value());
    } else if (out.has_value()) {
        out.destroy();
    }

    return step;
}

/**
 * @brief Executes a compiled plan using serial depth-first traversal.
 *
 * `abort_branch` is consumed locally by the parent and only terminates the
 * current child subtree; all other non-success statuses stop the remaining DFS
 * and bubble to the caller.
 *
 * The call stack shape is:
 *
 * - `run_plan(...)` creates per-run slot storage and enters the root
 * - `run_node<I>(...)` prepares node `I`'s direct-parent view and
 *   `exec_context`
 * - `run_node_body<I>(...)` executes node `I`, stores its payload if any, and
 *   delegates to child traversal
 * - `run_children<I, Ord>(...)` enumerates node `I`'s children and descends
 *   into each child via `run_node<child>(...)`
 *
 * This separation keeps node-local execution, context assembly, and sibling
 * traversal as distinct steps in the DFS executor.
 *
 * @tparam Plan Compiled plan type.
 * @param plan Mutable plan whose stored task objects are executed in DFS order.
 * @return Final `step_result` of the whole execution.
 */
template <typename Plan>
    requires (Plan::node_count > 0) &&
             detail::plan_prev_source_valid_v<Plan> &&
             detail::plan_fanout_prev_valid_v<Plan>
[[nodiscard]] constexpr step_result run_plan(Plan& plan) {
    plan_exec_slots<Plan> slots;
    return detail::run_node<0>(plan, slots);
}

/**
 * @brief Executes a compiled plan using serial depth-first traversal with an
 * external typed execution context.
 *
 * @tparam Plan Compiled plan type.
 * @tparam Ctx Borrowed context type.
 * @param plan Mutable plan whose stored task objects are executed in DFS order.
 * @param ctx Borrowed context exposed to each node through `from_ctx(...)`.
 * @return Final `step_result` of the whole execution.
 */
template <typename Plan, typename Ctx>
    requires (Plan::node_count > 0) &&
             detail::plan_prev_source_valid_v<Plan> &&
             detail::plan_fanout_prev_valid_v<Plan>
[[nodiscard]] constexpr step_result run_plan(Plan& plan, Ctx& ctx) {
    plan_exec_slots<Plan> slots;
    return detail::run_node<0>(plan, slots, ctx);
}

}  // namespace yorch
