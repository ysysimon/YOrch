#include "yorch/bind.hpp"

int main() {
    auto task = yorch::bind_forward_prev<int>(
        [](const int&) noexcept -> yorch::step_result {
            return yorch::step_result::success();
        },
        yorch::borrow_prev<int>());

    (void)task;
    return 0;
}
