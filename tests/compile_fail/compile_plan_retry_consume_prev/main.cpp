#include "yorch/bind.hpp"
#include "yorch/plan.hpp"
#include "yorch/task_adapters.hpp"
#include "yorch/task_tree.hpp"

int main() {
    auto tree = yorch::task_tree.root(yorch::bind([]() noexcept -> int {
            return 1;
        }))
        .node<1>(yorch::with_retry(
            yorch::bind(
                [](int value) noexcept -> yorch::step_result {
                    return value > 0
                        ? yorch::step_result::success()
                        : yorch::step_result::failure();
                },
                yorch::consume_prev<int>()),
            yorch::retry_fixed_policy {1}));

    auto plan = yorch::compile_plan(tree);
    (void)plan;
    return 0;
}
