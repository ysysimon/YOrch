#include "yorch/task_adapters.hpp"

#include "yorch/bind.hpp"

#include <gtest/gtest.h>

#include <exception>
#include <stdexcept>

namespace {

struct throwing_spec_task {
    constexpr yorch::step_result invoke_raw(yorch::exec_context<void>&) {
        return yorch::step_result::success();
    }
};

struct throwing_int_source {
    explicit throwing_int_source(int v) : value(v) {}

    operator int() const {
        throw std::runtime_error("convert");
    }

    int value;
};

struct mismatched_declared_task {
    using raw_result_type = int;

    yorch::step_result invoke_raw(yorch::exec_context<void>&) noexcept {
        return yorch::step_result::success();
    }
};

using catchable_step_bound_task_t = decltype(yorch::bind([]() -> yorch::step_result {
    return yorch::step_result::success();
}));

using policy_payload_bound_task_t = decltype(yorch::bind([]() -> yorch::task_result<int> {
    return yorch::task_result<int>::success(1);
}));

using compatible_exception_policy_t = decltype([](std::exception_ptr) noexcept -> yorch::task_result<int> {
    return yorch::task_result<int>::failure();
});

using incompatible_exception_policy_t = decltype([]() noexcept -> yorch::task_result<int> {
    return yorch::task_result<int>::failure();
});

}  // namespace

static_assert(yorch::catch_wrappable_task<throwing_spec_task&, void>);
static_assert(!yorch::catch_wrappable_task<mismatched_declared_task&, void>);
static_assert(yorch::default_catchable_task<catchable_step_bound_task_t, void>);
static_assert(yorch::catch_policy_compatible_task<
              policy_payload_bound_task_t,
              compatible_exception_policy_t,
              void>);
static_assert(!yorch::catch_policy_compatible_task<
              policy_payload_bound_task_t,
              incompatible_exception_policy_t,
              void>);

TEST(TaskAdaptersTest, CatchAsFailurePreservesStepResultOnSuccess) {
    yorch::exec_context<void> exec;

    auto task = yorch::catch_as_failure(
        yorch::bind(
            []() -> yorch::step_result {
                return yorch::step_result::retry();
            }));

    const auto raw = task.invoke_raw(exec);

    EXPECT_EQ(raw.status, yorch::step_status::retry);
}

TEST(TaskAdaptersTest, CatchAsFailureUsesFallbackPolicyForPayloadTask) {
    yorch::exec_context<void> exec;

    auto task = yorch::catch_as_failure(
        yorch::bind(
            []() -> yorch::task_result<int> {
                throw std::runtime_error("boom");
            }),
        [](std::exception_ptr ep) noexcept -> yorch::task_result<int> {
            try {
                std::rethrow_exception(ep);
            } catch (const std::runtime_error&) {
                return yorch::task_result<int>::failure();
            } catch (...) {
                return yorch::task_result<int>::abort_chain();
            }
            return yorch::task_result<int>::failure();
        });

    const auto raw = task.invoke_raw(exec);

    EXPECT_EQ(raw.step.status, yorch::step_status::failure);
    EXPECT_FALSE(raw.has_value());
}

TEST(TaskAdaptersTest, CatchAsFailureCanCatchResolutionExceptions) {
    yorch::exec_context<void> exec;

    auto task = yorch::catch_as_failure(
        yorch::bind(
            [](int value) noexcept -> int {
                return value;
            },
            yorch::value(throwing_int_source {42})),
        [](std::exception_ptr) noexcept -> int {
            return -7;
        });

    const auto raw = task.invoke_raw(exec);

    EXPECT_EQ(raw, -7);
}
