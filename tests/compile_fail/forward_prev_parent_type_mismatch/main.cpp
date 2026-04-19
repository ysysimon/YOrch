#include "yorch/bind.hpp"
#include "yorch/executor.hpp"
#include "yorch/plan.hpp"
#include "yorch/task_tree.hpp"

int main() {
    auto tree = yorch::task_tree.root(yorch::bind([]() noexcept -> int {
            return 7;
        }))
        .node<1>(yorch::bind_forward_prev<float>(
            [](float& value) noexcept -> yorch::step_result {
                value += 1.0F;
                return yorch::step_result::success();
            },
            yorch::borrow_prev_mut<float>()));

    auto plan = yorch::compile_plan(tree);
    const auto result = yorch::run_plan(plan);
    return static_cast<int>(result.status);
}
