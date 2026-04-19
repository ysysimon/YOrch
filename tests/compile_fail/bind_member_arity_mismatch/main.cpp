#include "yorch/bind.hpp"

namespace {

struct worker {
    yorch::step_result run(int, int) noexcept {
        return yorch::step_result::success();
    }
};

} // namespace

int main() {
    auto task = yorch::bind_member(
        &worker::run,
        yorch::value(worker {}),
        yorch::value(1));

    (void)task;
    return 0;
}
