#include "yorch/bind.hpp"

int main() {
    auto task = yorch::bind_forward_prev<int>(
        [](int&, yorch::direct_out<int> out) noexcept -> yorch::step_result {
            return out.success(1);
        },
        yorch::borrow_prev_mut<int>());

    (void)task;
    return 0;
}
