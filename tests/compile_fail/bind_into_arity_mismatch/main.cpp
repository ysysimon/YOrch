#include "yorch/bind.hpp"

int main() {
    auto task = yorch::bind_into<int>(
        [](int, yorch::direct_out<int>) noexcept -> yorch::step_result {
            return yorch::step_result::success();
        });

    (void)task;
    return 0;
}
