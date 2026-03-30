#include "yorch/executor.hpp"

#include "yorch/bind.hpp"

#include <gtest/gtest.h>

#include <stdexcept>

namespace {

struct noexcept_task {
    constexpr yorch::step_result invoke_raw(yorch::exec_context<void>&) noexcept {
        return yorch::step_result::success();
    }
};

struct throwing_spec_task {
    constexpr yorch::step_result invoke_raw(yorch::exec_context<void>&) {
        return yorch::step_result::success();
    }
};

struct mismatched_declared_task {
    using raw_result_type = int;

    constexpr yorch::step_result invoke_raw(yorch::exec_context<void>&) noexcept {
        return yorch::step_result::success();
    }
};

}  // namespace

static_assert(yorch::executable_task<noexcept_task&, void>);
static_assert(!yorch::executable_task<throwing_spec_task&, void>);
static_assert(!yorch::executable_task<mismatched_declared_task&, void>);

using caught_throwing_task_t = decltype(yorch::catch_as_failure(throwing_spec_task {}));
static_assert(yorch::executable_task<caught_throwing_task_t&, void>);

using throwing_bound_task_t = decltype(yorch::bind([]() -> yorch::step_result {
    throw std::runtime_error("boom");
}));
static_assert(!yorch::executable_task<throwing_bound_task_t&, void>);

TEST(ExecutorTest, RunTaskExecutesBoundTaskAgainstContext) {
    yorch::context<int, long> ctx(3, 7L);
    yorch::exec_context<decltype(ctx)> exec {ctx};

    auto task = yorch::bind(
        [](int& value, long label, int delta) noexcept -> bool {
            value += delta;
            return label == 7L;
        },
        yorch::from_ctx<int>(),
        yorch::from_ctx<long>(),
        yorch::value(4));

    const auto result = yorch::run_task(task, exec);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(ctx.get<int>(), 7);
}

TEST(ExecutorTest, RunTaskTreatsBoolAsOrdinarySuccessfulPayload) {
    yorch::exec_context<void> exec;

    auto true_task = yorch::bind(
        [](int value) noexcept -> bool {
            return value == 42;
        },
        yorch::value(42));

    auto false_task = yorch::bind(
        []() noexcept -> bool {
            return false;
        });

    const auto true_result = yorch::run_task(true_task, exec);
    const auto false_result = yorch::run_task(false_task, exec);

    EXPECT_EQ(true_result.status, yorch::step_status::success);
    EXPECT_EQ(false_result.status, yorch::step_status::success);
}

TEST(ExecutorTest, RunTaskPropagatesNonSuccessStatuses) {
    yorch::context<int> ctx(5);
    yorch::exec_context<decltype(ctx)> exec {ctx};

    auto retry_task = yorch::bind(
        [](int& value) noexcept -> yorch::step_result {
            value *= 2;
            return yorch::step_result::retry();
        },
        yorch::from_ctx<int>());

    const auto result = yorch::run_task(retry_task, exec);

    EXPECT_EQ(result.status, yorch::step_status::retry);
    EXPECT_EQ(ctx.get<int>(), 10);
}

TEST(ExecutorTest, RunTaskTreatsPlainValueReturnAsSuccess) {
    yorch::context<int> ctx(5);
    yorch::exec_context<decltype(ctx)> exec {ctx};

    auto task = yorch::bind(
        [](int& value) noexcept -> int {
            value += 3;
            return value * 2;
        },
        yorch::from_ctx<int>());

    const auto result = yorch::run_task(task, exec);

    EXPECT_EQ(result.status, yorch::step_status::success);
    EXPECT_EQ(ctx.get<int>(), 8);
}

TEST(ExecutorTest, RunTaskNormalizesTaskResultPayloadCarrier) {
    yorch::context<int> ctx(5);
    yorch::exec_context<decltype(ctx)> exec {ctx};

    auto task = yorch::bind(
        [](int& value) noexcept -> yorch::task_result<int> {
            value += 2;
            return yorch::task_result<int>::success(value * 3);
        },
        yorch::from_ctx<int>());

    const auto result = yorch::run_task(task, exec);

    EXPECT_EQ(result.status, yorch::step_status::success);
    EXPECT_EQ(ctx.get<int>(), 7);
}

TEST(ExecutorTest, RunTaskNormalizesVoidTaskResult) {
    yorch::exec_context<void> exec;

    auto task = yorch::bind(
        []() noexcept -> yorch::task_result<void> {
            return yorch::task_result<void>::abort_branch();
        });

    const auto result = yorch::run_task(task, exec);

    EXPECT_EQ(result.status, yorch::step_status::abort_branch);
}

TEST(ExecutorTest, RunTaskExecutesCatchAsFailureAdapterEndToEnd) {
    yorch::exec_context<void> exec;

    auto task = yorch::catch_as_failure(
        yorch::bind(
            []() {
                throw std::runtime_error("boom");
            }));

    const auto result = yorch::run_task(task, exec);

    EXPECT_EQ(result.status, yorch::step_status::failure);
}
