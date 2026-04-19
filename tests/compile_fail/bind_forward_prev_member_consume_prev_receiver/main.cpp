#include "yorch/bind.hpp"

namespace {

struct worker {
    yorch::step_result mutate(int delta) noexcept {
        (void)delta;
        return yorch::step_result::success();
    }
};

} // namespace

int main() {
    auto task = yorch::bind_forward_prev_member<worker>(
        &worker::mutate,
        yorch::consume_prev<worker>(),
        yorch::value(1));

    (void)task;
    return 0;
}
