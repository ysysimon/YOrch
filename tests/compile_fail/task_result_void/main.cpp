#include "yorch/result.hpp"

auto build_invalid_result() -> yorch::task_result<void> {
    return yorch::task_result<void>::failure();
}

int main() {
    static_cast<void>(build_invalid_result);
    return 0;
}
