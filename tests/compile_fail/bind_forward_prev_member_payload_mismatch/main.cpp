#include "yorch/bind.hpp"

namespace {

struct service {
    yorch::step_result mutate(float&) noexcept {
        return yorch::step_result::success();
    }
};

} // namespace

int main() {
    auto task = yorch::bind_forward_prev_member<int>(
        &service::mutate,
        yorch::value(service {}),
        yorch::borrow_prev_mut<float>());

    (void)task;
    return 0;
}
