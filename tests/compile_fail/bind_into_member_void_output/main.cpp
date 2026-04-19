#include "yorch/bind.hpp"

namespace {

struct worker {
    yorch::step_result emit(yorch::direct_out<int>) noexcept {
        return yorch::step_result::success();
    }
};

} // namespace

int main() {
    auto task = yorch::bind_into_member<void>(
        &worker::emit,
        yorch::value(worker {}));

    (void)task;
    return 0;
}
