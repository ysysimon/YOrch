#include "yorch/bind.hpp"

int main() {
    auto task = yorch::bind_forward_prev<void>(
        [](int&) noexcept -> yorch::step_result {
            return yorch::step_result::success();
        },
        yorch::borrow_prev_mut<int>());

    (void)task;
    return 0;
}
