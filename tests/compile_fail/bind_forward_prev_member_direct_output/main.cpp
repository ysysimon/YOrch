#include "yorch/bind.hpp"

namespace {

struct worker {
    yorch::step_result emit(int, yorch::direct_out<int>) noexcept {
        return yorch::step_result::success();
    }
};

} // namespace

int main() {
    auto task = yorch::bind_forward_prev_member<int>(
        &worker::emit,
        yorch::borrow_prev_mut<worker>(),
        yorch::value(1));

    (void)task;
    return 0;
}
