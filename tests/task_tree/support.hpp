#pragma once

#include "yorch/task_tree.hpp"

#include "yorch/executor.hpp" // IWYU pragma: keep
#include "yorch/plan.hpp" // IWYU pragma: keep

#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>

namespace task_tree_test_support {

constexpr auto make_noop_task() {
    return yorch::bind([]() noexcept {});
}

struct ordinary_callable {
    void operator()() const noexcept {}
};

struct member_tree_worker {
    int base = 0;

    int seed(int delta) noexcept {
        base += delta;
        return base;
    }

    yorch::step_result emit_text(int delta, yorch::direct_out<std::string> out) noexcept {
        base += delta;
        return out.success(std::to_string(base));
    }
};

struct move_only_tree_worker {
    explicit move_only_tree_worker(int initial) : base(initial) {}

    move_only_tree_worker(const move_only_tree_worker&) = delete;
    move_only_tree_worker& operator=(const move_only_tree_worker&) = delete;
    move_only_tree_worker(move_only_tree_worker&&) noexcept = default;
    move_only_tree_worker& operator=(move_only_tree_worker&&) noexcept = default;
    ~move_only_tree_worker() = default;

    int base = 0;

    int bump(int delta) noexcept {
        base += delta;
        return base;
    }

    [[nodiscard]] yorch::step_result emit_text(yorch::direct_out<std::string> out) const noexcept {
        return out.success(std::to_string(base));
    }
};

template <typename TaskTree, std::size_t Level>
concept can_append_noop =
    requires(TaskTree&& task_tree) {
        std::forward<TaskTree>(task_tree).template node<Level>(make_noop_task());
    };

template <typename TaskTree, std::size_t Level>
concept can_append_noop_with_fanout =
    requires(TaskTree&& task_tree) {
        std::forward<TaskTree>(task_tree).template node<Level>(
            make_noop_task(),
            yorch::fanout_shared_readonly_policy {});
    };

template <typename TaskTree>
concept can_append_root =
    requires(TaskTree&& task_tree) {
        std::forward<TaskTree>(task_tree).root(make_noop_task());
    };

template <typename TaskTree>
concept can_append_root_with_fanout =
    requires(TaskTree&& task_tree) {
        std::forward<TaskTree>(task_tree).root(
            make_noop_task(),
            yorch::fanout_shared_readonly_policy {});
    };

template <typename TaskTree>
concept can_append_root_callable =
    requires(TaskTree&& task_tree) {
        std::forward<TaskTree>(task_tree).root([]() noexcept {})();
    };

template <typename TaskTree>
concept can_append_root_callable_into =
    requires(TaskTree&& task_tree) {
        std::forward<TaskTree>(task_tree)
            .root_into([](yorch::direct_out<int>) noexcept {})();
    };

template <typename TaskTree>
concept can_append_root_member_callable =
    requires(TaskTree&& task_tree) {
        std::forward<TaskTree>(task_tree).root(&member_tree_worker::seed);
    };

template <typename TaskTree>
concept can_append_root_member_sugar =
    requires(TaskTree&& task_tree) {
        std::forward<TaskTree>(task_tree).root_member(
            &member_tree_worker::seed,
            yorch::value(member_tree_worker {}))(
            yorch::value(1));
    };

template <typename TaskTree>
concept can_append_root_into_member_sugar =
    requires(TaskTree&& task_tree) {
        std::forward<TaskTree>(task_tree).root_into_member(
            &member_tree_worker::emit_text,
            yorch::value(member_tree_worker {}))(
            yorch::value(1));
    };

template <typename TaskTree>
concept can_append_root_member_with_direct_output =
    requires(TaskTree&& task_tree) {
        std::forward<TaskTree>(task_tree).root_member(
            &member_tree_worker::emit_text,
            yorch::value(member_tree_worker {}))(
            yorch::value(1));
    };

template <typename TaskTree>
concept can_append_root_into_member_with_ordinary =
    requires(TaskTree&& task_tree) {
        std::forward<TaskTree>(task_tree).root_into_member(
            &member_tree_worker::seed,
            yorch::value(member_tree_worker {}))(
            yorch::value(1));
    };

template <typename TaskTree>
concept can_append_root_member_without_receiver =
    requires(TaskTree&& task_tree) {
        requires (!std::is_void_v<decltype(
            std::forward<TaskTree>(task_tree).root_member(&member_tree_worker::seed))>);
    };

template <typename TaskTree>
concept can_append_root_into_member_without_receiver =
    requires(TaskTree&& task_tree) {
        requires (!std::is_void_v<decltype(
            std::forward<TaskTree>(task_tree).root_into_member(&member_tree_worker::emit_text))>);
    };

template <typename TaskTree, std::size_t Level>
concept can_append_node_callable_into =
    requires(TaskTree&& task_tree) {
        std::forward<TaskTree>(task_tree)
            .template node_into<Level>([](yorch::direct_out<int>) noexcept {})();
    };

template <typename TaskTree, std::size_t Level>
concept can_append_node_member_sugar =
    requires(TaskTree&& task_tree) {
        std::forward<TaskTree>(task_tree).template node_member<Level>(
            &member_tree_worker::seed,
            yorch::value(member_tree_worker {}))(
            yorch::value(1));
    };

template <typename TaskTree, std::size_t Level>
concept can_append_node_into_member_sugar =
    requires(TaskTree&& task_tree) {
        std::forward<TaskTree>(task_tree).template node_into_member<Level>(
            &member_tree_worker::emit_text,
            yorch::value(member_tree_worker {}))(
            yorch::value(1));
    };

template <typename TaskTree, std::size_t Level>
concept can_append_node_member_without_receiver =
    requires(TaskTree&& task_tree) {
        requires (!std::is_void_v<decltype(
            std::forward<TaskTree>(task_tree).template node_member<Level>(&member_tree_worker::seed))>);
    };

template <typename TaskTree, std::size_t Level>
concept can_append_node_into_member_without_receiver =
    requires(TaskTree&& task_tree) {
        requires (!std::is_void_v<decltype(
            std::forward<TaskTree>(task_tree).template node_into_member<Level>(&member_tree_worker::emit_text))>);
    };

}  // namespace task_tree_test_support
