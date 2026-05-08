#pragma once

#include <yorch/yorch.hpp>

#include <array>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace yorch_bench {

struct large256 {
    std::array<std::uint64_t, 32> words {};
};

static_assert(sizeof(large256) == 256);
static_assert(std::is_trivially_copyable_v<large256>);

[[nodiscard]] inline large256 make_large256(std::uint64_t seed) noexcept {
    large256 value {};
    for (std::size_t i = 0; i < value.words.size(); ++i) {
        value.words[i] = seed + i;
    }
    return value;
}

inline void consume_int(int& sink, int value) noexcept {
    sink += value;
}

inline void consume_large(int& sink, const large256& value) noexcept {
    sink += static_cast<int>(value.words[0]);
}

template <std::size_t Level, std::size_t End, typename Tree>
[[nodiscard]] auto append_chain(Tree&& tree, int& sink) {
    auto next = std::forward<Tree>(tree).template node<Level>(yorch::bind(
        [&sink](const int& value) noexcept -> int {
            const auto next_value = value + 1;
            consume_int(sink, next_value);
            return next_value;
        },
        yorch::borrow_prev<int>()));

    if constexpr (Level == End) {
        return next;
    } else {
        return append_chain<Level + 1, End>(std::move(next), sink);
    }
}

template <std::size_t Nodes>
[[nodiscard]] auto make_chain_tree(int& sink) {
    static_assert(Nodes >= 1);

    auto root = yorch::task_tree.root(yorch::bind([&sink]() noexcept -> int {
        consume_int(sink, 1);
        return 1;
    }));

    if constexpr (Nodes == 1) {
        return root;
    } else {
        return append_chain<1, Nodes - 1>(std::move(root), sink);
    }
}

[[nodiscard]] inline auto make_wide8_tree(int& sink) {
    return yorch::task_tree.root(yorch::bind([&sink]() noexcept -> int {
            consume_int(sink, 1);
            return 1;
        }))
        .node<1>(yorch::bind(
            [&sink](const int& value) noexcept -> int {
                consume_int(sink, value + 1);
                return value + 1;
            },
            yorch::borrow_prev<int>()))
        .node<1>(yorch::bind(
            [&sink](const int& value) noexcept -> int {
                consume_int(sink, value + 2);
                return value + 2;
            },
            yorch::borrow_prev<int>()))
        .node<1>(yorch::bind(
            [&sink](const int& value) noexcept -> int {
                consume_int(sink, value + 3);
                return value + 3;
            },
            yorch::borrow_prev<int>()))
        .node<1>(yorch::bind(
            [&sink](const int& value) noexcept -> int {
                consume_int(sink, value + 4);
                return value + 4;
            },
            yorch::borrow_prev<int>()))
        .node<1>(yorch::bind(
            [&sink](const int& value) noexcept -> int {
                consume_int(sink, value + 5);
                return value + 5;
            },
            yorch::borrow_prev<int>()))
        .node<1>(yorch::bind(
            [&sink](const int& value) noexcept -> int {
                consume_int(sink, value + 6);
                return value + 6;
            },
            yorch::borrow_prev<int>()))
        .node<1>(yorch::bind(
            [&sink](const int& value) noexcept -> int {
                consume_int(sink, value + 7);
                return value + 7;
            },
            yorch::borrow_prev<int>()))
        .node<1>(yorch::bind(
            [&sink](const int& value) noexcept -> int {
                consume_int(sink, value + 8);
                return value + 8;
            },
            yorch::borrow_prev<int>()));
}

[[nodiscard]] inline auto make_balanced15_tree(int& sink) {
    auto child_task = [&sink](const int& value) noexcept -> int {
        const auto next_value = value + 1;
        consume_int(sink, next_value);
        return next_value;
    };

    return yorch::task_tree.root(yorch::bind([&sink]() noexcept -> int {
            consume_int(sink, 1);
            return 1;
        }))
        .node<1>(yorch::bind(child_task, yorch::borrow_prev<int>()))
            .node<2>(yorch::bind(child_task, yorch::borrow_prev<int>()))
                .node<3>(yorch::bind(child_task, yorch::borrow_prev<int>()))
                .node<3>(yorch::bind(child_task, yorch::borrow_prev<int>()))
            .node<2>(yorch::bind(child_task, yorch::borrow_prev<int>()))
                .node<3>(yorch::bind(child_task, yorch::borrow_prev<int>()))
                .node<3>(yorch::bind(child_task, yorch::borrow_prev<int>()))
        .node<1>(yorch::bind(child_task, yorch::borrow_prev<int>()))
            .node<2>(yorch::bind(child_task, yorch::borrow_prev<int>()))
                .node<3>(yorch::bind(child_task, yorch::borrow_prev<int>()))
                .node<3>(yorch::bind(child_task, yorch::borrow_prev<int>()))
            .node<2>(yorch::bind(child_task, yorch::borrow_prev<int>()))
                .node<3>(yorch::bind(child_task, yorch::borrow_prev<int>()))
                .node<3>(yorch::bind(child_task, yorch::borrow_prev<int>()));
}

[[nodiscard]] inline auto make_fanout_consume_copies3_tree(int& sink) {
    return yorch::task_tree.root(
            yorch::bind([&sink]() noexcept -> large256 {
                auto value = make_large256(1);
                consume_large(sink, value);
                return value;
            }),
            yorch::fanout_consume_with_copies_policy {})
        .node<1>(yorch::bind(
            [&sink](large256 value) noexcept -> yorch::step_result {
                consume_large(sink, value);
                return yorch::step_result::success();
            },
            yorch::consume_prev<large256>()))
        .node<1>(yorch::bind(
            [&sink](large256 value) noexcept -> yorch::step_result {
                consume_large(sink, value);
                return yorch::step_result::success();
            },
            yorch::copy_prev<large256>()))
        .node<1>(yorch::bind(
            [&sink](large256 value) noexcept -> yorch::step_result {
                consume_large(sink, value);
                return yorch::step_result::success();
            },
            yorch::copy_prev<large256>()));
}

struct chain8_scenario {
    static constexpr const char* topology = "Chain8";
    static constexpr const char* payload = "Int";
    static constexpr std::int64_t node_count = 8;

    [[nodiscard]] static auto make_tree(int& sink) {
        return make_chain_tree<8>(sink);
    }
};

struct chain32_scenario {
    static constexpr const char* topology = "Chain32";
    static constexpr const char* payload = "Int";
    static constexpr std::int64_t node_count = 32;

    [[nodiscard]] static auto make_tree(int& sink) {
        return make_chain_tree<32>(sink);
    }
};

struct wide8_scenario {
    static constexpr const char* topology = "Wide8";
    static constexpr const char* payload = "Int";
    static constexpr std::int64_t node_count = 9;

    [[nodiscard]] static auto make_tree(int& sink) {
        return make_wide8_tree(sink);
    }
};

struct balanced15_scenario {
    static constexpr const char* topology = "Balanced15";
    static constexpr const char* payload = "Int";
    static constexpr std::int64_t node_count = 15;

    [[nodiscard]] static auto make_tree(int& sink) {
        return make_balanced15_tree(sink);
    }
};

struct fanout_consume_copies3_scenario {
    static constexpr const char* topology = "FanoutConsumeCopies3";
    static constexpr const char* payload = "Large256";
    static constexpr std::int64_t node_count = 4;

    [[nodiscard]] static auto make_tree(int& sink) {
        return make_fanout_consume_copies3_tree(sink);
    }
};

} // namespace yorch_bench
