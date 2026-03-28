#include "yorch/executor.hpp"

#include "yorch/bind.hpp"

#include <gtest/gtest.h>

#include <exception>
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

struct throwing_int_source {
    explicit throwing_int_source(int v) : value(v) {}

    operator int() const {
        throw std::runtime_error("convert");
    }

    int value;
};

}  // namespace

static_assert(yorch::executable_task<noexcept_task&, void>);
static_assert(!yorch::executable_task<throwing_spec_task&, void>);

using caught_throwing_task_t = decltype(yorch::catch_as_failure(throwing_spec_task {}));
static_assert(yorch::executable_task<caught_throwing_task_t&, void>);

using throwing_bound_task_t = decltype(yorch::bind([]() -> yorch::step_result {
    throw std::runtime_error("boom");
}));
static_assert(!yorch::executable_task<throwing_bound_task_t&, void>);

using catchable_step_bound_task_t = decltype(yorch::bind([]() -> yorch::step_result {
    return yorch::step_result::success();
}));

using policy_payload_bound_task_t = decltype(yorch::bind([]() -> yorch::task_result<int> {
    return {yorch::step_result::success(), 1};
}));

using compatible_exception_policy_t = decltype([](std::exception_ptr) noexcept -> yorch::task_result<int> {
    return {yorch::step_result::failure(), -1};
});

using incompatible_exception_policy_t = decltype([]() noexcept -> yorch::task_result<int> {
    return {yorch::step_result::failure(), -1};
});

static_assert(yorch::default_catchable_task<catchable_step_bound_task_t, void>);
static_assert(yorch::catch_policy_compatible_task<
              policy_payload_bound_task_t,
              compatible_exception_policy_t,
              void>);
static_assert(!yorch::catch_policy_compatible_task<
              policy_payload_bound_task_t,
              incompatible_exception_policy_t,
              void>);

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
            return {yorch::step_result::skip(), value * 3};
        },
        yorch::from_ctx<int>());

    const auto result = yorch::run_task(task, exec);

    EXPECT_EQ(result.status, yorch::step_status::skip);
    EXPECT_EQ(ctx.get<int>(), 7);
}

TEST(ExecutorTest, RunTaskNormalizesVoidTaskResult) {
    yorch::exec_context<void> exec;

    auto task = yorch::bind(
        []() noexcept -> yorch::task_result<void> {
            return {yorch::step_result::abort_chain()};
        });

    const auto result = yorch::run_task(task, exec);

    EXPECT_EQ(result.status, yorch::step_status::abort_chain);
}

TEST(ExecutorTest, CatchAsFailureMapsThrowingVoidTaskToFailure) {
    yorch::exec_context<void> exec;

    auto task = yorch::catch_as_failure(
        yorch::bind(
            []() {
                throw std::runtime_error("boom");
            }));

    const auto result = yorch::run_task(task, exec);

    EXPECT_EQ(result.status, yorch::step_status::failure);
}

TEST(ExecutorTest, CatchAsFailurePreservesStepResultOnSuccess) {
    yorch::exec_context<void> exec;

    auto task = yorch::catch_as_failure(
        yorch::bind(
            []() -> yorch::step_result {
                return yorch::step_result::retry();
            }));

    const auto result = yorch::run_task(task, exec);

    EXPECT_EQ(result.status, yorch::step_status::retry);
}

TEST(ExecutorTest, CatchAsFailureUsesFallbackPolicyForPayloadTask) {
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
                return {yorch::step_result::failure(), -1};
            } catch (...) {
                return {yorch::step_result::abort_chain(), -2};
            }
            return {};
        });

    const auto raw = task.invoke_raw(exec);
    const auto result = yorch::run_task(task, exec);

    EXPECT_EQ(raw.step.status, yorch::step_status::failure);
    EXPECT_EQ(raw.value, -1);
    EXPECT_EQ(result.status, yorch::step_status::failure);
}

TEST(ExecutorTest, CatchAsFailureCanCatchResolutionExceptions) {
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
    const auto result = yorch::run_task(task, exec);

    EXPECT_EQ(raw, -7);
    EXPECT_EQ(result.status, yorch::step_status::success);
}
