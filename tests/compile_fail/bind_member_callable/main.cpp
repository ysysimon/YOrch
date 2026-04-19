#include "yorch/bind.hpp"

namespace {

struct worker {
    yorch::step_result run() noexcept {
        return yorch::step_result::success();
    }
};

} // namespace

int main() {
    auto task = yorch::bind(&worker::run);
    (void)task;
    return 0;
}
