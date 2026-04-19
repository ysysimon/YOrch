#include "yorch/bind.hpp"
#include "yorch/plan.hpp"
#include "yorch/task_tree.hpp"

int main() {
    auto tree = yorch::task_tree.root(yorch::bind([]() noexcept -> yorch::step_result {
            return yorch::step_result::success();
        }))
        .node<1>(yorch::bind_forward_prev<int>(
            [](int&) noexcept -> yorch::step_result {
                return yorch::step_result::success();
            },
            yorch::borrow_prev_mut<int>()));

    auto plan = yorch::compile_plan(tree);
    (void)plan;
    return 0;
}
