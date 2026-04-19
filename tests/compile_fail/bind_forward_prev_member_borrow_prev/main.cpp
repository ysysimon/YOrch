#include "yorch/bind.hpp"

namespace {

struct worker {
    yorch::step_result read() const noexcept {
        return yorch::step_result::success();
    }
};

} // namespace

int main() {
    auto task = yorch::bind_forward_prev_member<worker>(
        &worker::read,
        yorch::borrow_prev<worker>());

    (void)task;
    return 0;
}
