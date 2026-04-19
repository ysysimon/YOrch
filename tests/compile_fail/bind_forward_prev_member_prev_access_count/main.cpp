#include "yorch/bind.hpp"

namespace {

struct worker {
    yorch::step_result merge(worker&, int) noexcept {
        return yorch::step_result::success();
    }
};

} // namespace

int main() {
    auto task = yorch::bind_forward_prev_member<worker>(
        &worker::merge,
        yorch::borrow_prev_mut<worker>(),
        yorch::borrow_prev_mut<worker>(),
        yorch::value(1));

    (void)task;
    return 0;
}
