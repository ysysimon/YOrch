#include "yorch/bind.hpp"

int main() {
    auto task = yorch::bind(
        [](auto&&) noexcept -> yorch::step_result {
            return yorch::step_result::success();
        },
        yorch::value(1));

    (void)task;
    return 0;
}
