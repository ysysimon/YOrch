#include "yorch/bind.hpp"

namespace {

struct worker {
    yorch::step_result emit(int, yorch::direct_out<long>) noexcept {
        return yorch::step_result::success();
    }
};

} // namespace

int main() {
    auto task = yorch::bind_into_member<int>(
        &worker::emit,
        yorch::value(worker {}),
        yorch::value(1));

    (void)task;
    return 0;
}
