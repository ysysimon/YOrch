#include "yorch/bind.hpp"

namespace {

struct worker {
    int value = 0;

    yorch::step_result mutate() noexcept {
        ++value;
        return yorch::step_result::success();
    }
};

} // namespace

int main() {
    auto task = yorch::bind_forward_prev<worker>(
        &worker::mutate,
        yorch::borrow_prev_mut<worker>());

    (void)task;
    return 0;
}
