#include "yorch/bind.hpp"

namespace {

struct worker {
    yorch::step_result emit(int, yorch::direct_out<int>) noexcept {
        return yorch::step_result::success();
    }
};

} // namespace

int main() {
    auto task = yorch::bind_member(
        &worker::emit,
        yorch::value(worker {}),
        yorch::value(1));

    (void)task;
    return 0;
}
