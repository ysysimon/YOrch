#pragma once

#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

#include "../../context.hpp"
#include "../../resolve.hpp"
#include "../../slots.hpp"
#include "../../task_adapters.hpp"
#include "../../detail/bind/member_receiver.hpp"
#include "../../detail/bind/traits.hpp"

namespace yorch {

/**
 * @brief Bound executable task composed of a callable and per-parameter specs.
 *
 * At execution time, each spec is resolved against the target parameter type
 * deduced from `F`, then the callable is invoked and its original return value
 * is forwarded back to the caller.
 *
 * @tparam F Stored callable type.
 * @tparam Specs Stored spec types, one for each function parameter.
 */
template <typename F, typename... Specs>
struct bound_task {
    /// Raw return type declared by the stored callable.
    using raw_result_type = detail::result_t<F>;

    /// Stored callable to execute.
    F func;

    /// Stored argument binding specs in parameter order.
    std::tuple<Specs...> specs;

    /**
     * @brief Resolves all specs and executes the bound callable.
     * @tparam Ctx Context schema used for this execution.
     * @tparam Prev Parent slot view type for this execution.
     * @param ec Borrowed execution context.
     * @return User callable's original return value.
     */
    template <typename Ctx, typename Prev>
    constexpr decltype(auto) invoke_raw(exec_context<Ctx, Prev>& ec)
        noexcept(noexcept(call_impl_raw(ec, std::index_sequence_for<Specs...>{}))) {
        return call_impl_raw(ec, std::index_sequence_for<Specs...>{});
    }

private:
    template <typename Ctx, typename Prev, std::size_t... I>
    constexpr decltype(auto) call_impl_raw(exec_context<Ctx, Prev>& ec, std::index_sequence<I...>)
        noexcept(noexcept(std::invoke(
            func,
            resolve_as<detail::nth_arg_t<I, F>>(std::get<I>(specs), ec)...))) {
        return std::invoke(
            func,
            resolve_as<detail::nth_arg_t<I, F>>(std::get<I>(specs), ec)...
        );
    }
};

/**
 * @brief Bound direct-output task that writes its payload into a provided slot.
 *
 * This form keeps parameter resolution identical to `bound_task`, but the
 * stored callable receives a trailing `direct_out<T>` sink instead of
 * returning the payload through its function result.
 *
 * @tparam F Stored callable type.
 * @tparam T Payload type written into the output sink.
 * @tparam Specs Stored input binding specs, one for each non-output parameter.
 */
template <typename F, typename T, typename... Specs>
struct bound_output_task {
    using raw_result_type = step_result;
    using output_type = T;
    using output_protocol = detail::direct_output_protocol_tag;

    F func;
    std::tuple<Specs...> specs;

    template <typename Ctx, typename Prev>
    constexpr step_result invoke_into(exec_context<Ctx, Prev>& ec, direct_out<T> out)
        noexcept(noexcept(call_impl_into(ec, out, std::index_sequence_for<Specs...> {}))) {
        using call_result_t =
            decltype(call_impl_into(ec, out, std::index_sequence_for<Specs...> {}));

        if constexpr (std::is_void_v<call_result_t>) {
            call_impl_into(ec, out, std::index_sequence_for<Specs...> {});
            return step_result::success();
        } else {
            static_assert(std::is_same_v<std::remove_cvref_t<call_result_t>, step_result>,
                          "bind_into(...) callable must return void or yorch::step_result");
            return call_impl_into(ec, out, std::index_sequence_for<Specs...> {});
        }
    }

private:
    template <typename Ctx, typename Prev, std::size_t... I>
    constexpr decltype(auto) call_impl_into(
        exec_context<Ctx, Prev>& ec,
        direct_out<T> out,
        std::index_sequence<I...>)
        noexcept(noexcept(std::invoke(
            func,
            resolve_as<detail::nth_arg_t<I, F>>(std::get<I>(specs), ec)...,
            out))) {
        return std::invoke(
            func,
            resolve_as<detail::nth_arg_t<I, F>>(std::get<I>(specs), ec)...,
            out);
    }
};

template <typename F, typename T, typename... Specs>
struct bound_forward_prev_task {
    using raw_result_type = step_result;
    using output_type = T;
    using output_protocol = detail::forward_prev_output_protocol_tag;

    F func;
    std::tuple<Specs...> specs;

    template <typename Ctx, typename Prev>
    constexpr step_result invoke_raw(exec_context<Ctx, Prev>& ec)
        noexcept(noexcept(call_impl_raw(ec, std::index_sequence_for<Specs...> {}))) {
        using call_result_t =
            decltype(call_impl_raw(ec, std::index_sequence_for<Specs...> {}));

        if constexpr (std::is_void_v<call_result_t>) {
            call_impl_raw(ec, std::index_sequence_for<Specs...> {});
            return step_result::success();
        } else {
            static_assert(std::is_same_v<std::remove_cvref_t<call_result_t>, step_result>,
                          "bind_forward_prev(...) callable must return void or yorch::step_result");
            return call_impl_raw(ec, std::index_sequence_for<Specs...> {});
        }
    }

private:
    template <typename Ctx, typename Prev, std::size_t... I>
    constexpr decltype(auto) call_impl_raw(
        exec_context<Ctx, Prev>& ec,
        std::index_sequence<I...>)
        noexcept(noexcept(std::invoke(
            func,
            resolve_as<detail::nth_arg_t<I, F>>(std::get<I>(specs), ec)...))) {
        return std::invoke(
            func,
            resolve_as<detail::nth_arg_t<I, F>>(std::get<I>(specs), ec)...);
    }
};

/**
 * @brief Bound executable task composed of a member function, receiver spec,
 * and per-parameter specs.
 *
 * This form models the receiver as an explicit binding source distinct from
 * the business parameters. At execution time the receiver is resolved first,
 * then all remaining specs are resolved against the member-function parameter
 * list and invoked through `std::invoke(...)`.
 *
 * @tparam F Stored member-function pointer type.
 * @tparam ReceiverSpec Stored receiver binding spec.
 * @tparam Specs Stored business-parameter specs.
 */
template <typename F, typename ReceiverSpec, typename... Specs>
struct bound_member_task {
    using raw_result_type = detail::member_result_t<F>;

    F func;
    ReceiverSpec receiver_spec;
    std::tuple<Specs...> specs;

    template <typename Ctx, typename Prev>
    constexpr decltype(auto) invoke_raw(exec_context<Ctx, Prev>& ec)
        noexcept(noexcept(call_impl_raw(ec, std::index_sequence_for<Specs...>{}))) {
        return call_impl_raw(ec, std::index_sequence_for<Specs...>{});
    }

private:
    template <typename Ctx, typename Prev, std::size_t... I>
    constexpr decltype(auto) call_impl_raw(exec_context<Ctx, Prev>& ec, std::index_sequence<I...>)
        noexcept(noexcept(detail::invoke_member_with_receiver(
            func,
            detail::resolve_member_receiver<F>(receiver_spec, ec),
            resolve_as<detail::member_nth_arg_t<I, F>>(std::get<I>(specs), ec)...))) {
        auto receiver = detail::resolve_member_receiver<F>(receiver_spec, ec);
        return detail::invoke_member_with_receiver(
            func,
            std::move(receiver),
            resolve_as<detail::member_nth_arg_t<I, F>>(std::get<I>(specs), ec)...);
    }
};

/**
 * @brief Bound direct-output task composed of a member function, receiver
 * spec, and non-output parameter specs.
 *
 * The receiver is resolved first, then all non-output business parameters are
 * resolved in order, and finally the provided `direct_out<T>` sink is passed
 * to the stored member function via `std::invoke(...)`.
 *
 * @tparam F Stored member-function pointer type.
 * @tparam ReceiverSpec Stored receiver binding spec.
 * @tparam T Output payload type.
 * @tparam Specs Stored non-output business-parameter specs.
 */
template <typename F, typename ReceiverSpec, typename T, typename... Specs>
struct bound_member_output_task {
    using raw_result_type = step_result;
    using output_type = T;
    using output_protocol = detail::direct_output_protocol_tag;

    F func;
    ReceiverSpec receiver_spec;
    std::tuple<Specs...> specs;

    template <typename Ctx, typename Prev>
    constexpr step_result invoke_into(exec_context<Ctx, Prev>& ec, direct_out<T> out)
        noexcept(noexcept(call_impl_into(ec, out, std::index_sequence_for<Specs...> {}))) {
        using call_result_t =
            decltype(call_impl_into(ec, out, std::index_sequence_for<Specs...> {}));

        if constexpr (std::is_void_v<call_result_t>) {
            call_impl_into(ec, out, std::index_sequence_for<Specs...> {});
            return step_result::success();
        } else {
            static_assert(std::is_same_v<std::remove_cvref_t<call_result_t>, step_result>,
                          "bind_into_member(...) callable must return void or yorch::step_result");
            return call_impl_into(ec, out, std::index_sequence_for<Specs...> {});
        }
    }

private:
    template <typename Ctx, typename Prev, std::size_t... I>
    constexpr decltype(auto) call_impl_into(
        exec_context<Ctx, Prev>& ec,
        direct_out<T> out,
        std::index_sequence<I...>)
        noexcept(noexcept(detail::invoke_member_with_receiver(
            func,
            detail::resolve_member_receiver<F>(receiver_spec, ec),
            resolve_as<detail::member_nth_arg_t<I, F>>(std::get<I>(specs), ec)...,
            out))) {
        auto receiver = detail::resolve_member_receiver<F>(receiver_spec, ec);
        return detail::invoke_member_with_receiver(
            func,
            std::move(receiver),
            resolve_as<detail::member_nth_arg_t<I, F>>(std::get<I>(specs), ec)...,
            out);
    }
};

} // namespace yorch
