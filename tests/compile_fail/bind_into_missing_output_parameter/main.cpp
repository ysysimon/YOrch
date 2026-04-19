#include "yorch/bind.hpp"

int main() {
    auto task = yorch::bind_into<int>(
        []() noexcept -> yorch::step_result {
            return yorch::step_result::success();
        });

    (void)task;
    return 0;
}
