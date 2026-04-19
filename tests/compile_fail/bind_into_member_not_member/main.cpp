#include "yorch/bind.hpp"

int main() {
    auto task = yorch::bind_into_member<int>(
        [](int) noexcept -> yorch::step_result {
            return yorch::step_result::success();
        },
        yorch::value(1),
        yorch::value(2));

    (void)task;
    return 0;
}
