#include "yorch/bind.hpp"

namespace {

struct worker {
    yorch::step_result mutate(int) noexcept {
        return yorch::step_result::success();
    }
};

} // namespace

int main() {
    auto task = yorch::bind_forward_prev_member<int>(&worker::mutate);

    (void)task;
    return 0;
}
