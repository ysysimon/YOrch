#include "yorch/bind.hpp"

int main() {
    auto task = yorch::bind_forward_prev_member<int>(
        [](int&) noexcept -> yorch::step_result {
            return yorch::step_result::success();
        },
        yorch::value(1),
        yorch::borrow_prev_mut<int>());

    (void)task;
    return 0;
}
