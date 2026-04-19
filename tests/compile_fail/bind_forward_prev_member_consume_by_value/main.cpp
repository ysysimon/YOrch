#include "yorch/bind.hpp"

namespace {

struct worker {
    yorch::step_result consume(int) noexcept {
        return yorch::step_result::success();
    }
};

} // namespace

int main() {
    worker value;
    auto task = yorch::bind_forward_prev_member<int>(
        &worker::consume,
        yorch::value(std::ref(value)),
        yorch::consume_prev<int>());

    (void)task;
    return 0;
}
