#include "yorch/yorch.hpp"

#include <iostream>

namespace {

int TaskA(int lhs, int rhs) noexcept {
    const int sum = lhs + rhs;
    std::cout << "TaskA(" << lhs << ", " << rhs << ") -> " << sum << '\n';
    return sum;
}

struct TaskB {
    int operator()(const int& value) const noexcept {
        const int next = value + 10;
        std::cout << "TaskB(" << value << ") -> " << next << '\n';
        return next;
    }
};

struct TaskF {
    int operator()(int& value) const noexcept {
        value *= 2;
        std::cout << "TaskF(" << value << ")\n";
        return value;
    }
};

struct TaskI {
    void operator()(int value) const noexcept {
        std::cout << "TaskI(" << value << ")\n";
    }
};

} // namespace

int main() {
    std::cout << "example_hello_world\n\n";

    // all tasks
    const auto task_a = &TaskA;
    const auto task_b = TaskB {};
    const auto task_c = [](const int& value) noexcept {
        std::cout << "TaskC(" << value << ")\n";
    };
    const auto task_d = [](const int& borrowed, int copied, const char* label) noexcept {
        std::cout << "TaskD(" << borrowed
                  << ", " << copied
                  << ", " << label << ")\n";
    };
    const auto task_e = [offset = 5](const int& value) noexcept -> int {
        const int next = value + offset;
        std::cout << "TaskE(" << value << ", " << offset << ") -> " << next << '\n';
        return next;
    };
    const auto task_f = TaskF {};
    const auto task_g = [](const int& value) noexcept {
        std::cout << "TaskG(" << value << ")\n";
    };
    const auto task_h = [](const int& value) noexcept -> int {
        const int next = value - 1;
        std::cout << "TaskH(" << value << ") -> " << next << '\n';
        return next;
    };
    const auto task_i = TaskI {};

    auto tree = yorch::task_tree
        .root(task_a)(yorch::value(2),yorch::value(3))
        .node<1>(task_b)(yorch::borrow_prev<int>())
            .node<2>(task_c)(yorch::borrow_prev<int>())
        .node<1>(task_d)(
            yorch::borrow_prev<int>(),
            yorch::copy_prev<int>(),
            yorch::value(static_cast<const char*>("copy_prev + value")))
        .node<1>(task_e)(yorch::borrow_prev<int>())
            .node<2>(task_f)(
                yorch::borrow_prev_mut<int>())
                .node<3>(task_g)(yorch::borrow_prev<int>())
        .node<1>(task_h)(yorch::borrow_prev<int>())
            .node<2>(task_i)(yorch::consume_prev<int>());

    auto plan = yorch::compile_plan(tree);
    const auto result = yorch::run_plan(plan);

    std::cout << "\nrun_plan = " << (result.ok() ? "success" : "non-success") << '\n';
    return result.ok() ? 0 : 1;
}
