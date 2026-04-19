#include "yorch/bind.hpp"

namespace {

struct worker {
    yorch::step_result run(yorch::direct_out<int>) noexcept {
        return yorch::step_result::success();
    }
};

} // namespace

int main() {
    auto task = yorch::bind_into<int>(&worker::run);
    (void)task;
    return 0;
}
