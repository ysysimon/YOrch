#include "yorch/bind.hpp"

namespace {

struct service {
    yorch::step_result consume(int) noexcept {
        return yorch::step_result::success();
    }
};

} // namespace

int main() {
    service value;
    auto task = yorch::bind_forward_prev_member<int>(
        &service::consume,
        yorch::value(std::ref(value)),
        yorch::consume_prev<int>());

    (void)task;
    return 0;
}
