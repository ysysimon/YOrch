#include "yorch/task_adapters.hpp"

#include "yorch/bind.hpp"

#include <gtest/gtest.h>

#include <exception>
#include <stdexcept>

namespace {

struct throwing_spec_task {
    static constexpr yorch::step_result invoke_raw(yorch::exec_context<void>&) {
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

    static yorch::step_result invoke_raw(yorch::exec_context<void>&) noexcept {
        return yorch::step_result::success();
    }
};

using catchable_step_bound_task_t = decltype(yorch::bind([]() -> yorch::step_result {
    return yorch::step_result::success();
}));

using policy_payload_bound_task_t = decltype(yorch::bind([]() -> yorch::task_result<int> {
    return yorch::task_result<int>::success(1);
}));

// NOLINTNEXTLINE(performance-unnecessary-value-param)
using compatible_exception_policy_t = decltype([](std::exception_ptr) noexcept -> yorch::task_result<int> {
    return yorch::task_result<int>::failure();
});

using compatible_exception_policy_ref_t =
    decltype([](const std::exception_ptr&) noexcept -> yorch::task_result<int> {
        return yorch::task_result<int>::failure();
    });

using incompatible_exception_policy_rvalue_ref_t =
    decltype([](std::exception_ptr&&) noexcept -> yorch::task_result<int> {
        return yorch::task_result<int>::failure();
    });

using incompatible_exception_policy_t = decltype([]() noexcept -> yorch::task_result<int> {
    return yorch::task_result<int>::failure();
});

using direct_output_task_t = decltype(yorch::bind_into<int>(
    [](yorch::result_out<int> out) noexcept -> yorch::step_result {
        return out.success(1);
    }));

// NOLINTNEXTLINE(performance-unnecessary-value-param)
using compatible_direct_output_policy_t = decltype([](std::exception_ptr) noexcept -> yorch::step_result {
    return yorch::step_result::failure();
});

using compatible_direct_output_policy_ref_t =
    decltype([](const std::exception_ptr&) noexcept -> yorch::step_result {
        return yorch::step_result::failure();
    });

}  // namespace

static_assert(yorch::adapter_wrappable_task<throwing_spec_task&, void>);
static_assert(!yorch::adapter_wrappable_task<mismatched_declared_task&, void>);
static_assert(yorch::default_catchable_task<catchable_step_bound_task_t, void>);
static_assert(yorch::direct_output_task<direct_output_task_t, void>);
static_assert(yorch::retry_policy<yorch::retry_fixed_policy>);
static_assert(yorch::retry_policy<yorch::retry_fixed_passthrough_policy>);
static_assert(yorch::retry_policy<yorch::retry_forever_policy>);
static_assert(yorch::catch_policy_compatible_task<
              policy_payload_bound_task_t,
              compatible_exception_policy_t,
              void>);
static_assert(yorch::catch_policy_compatible_task<
              policy_payload_bound_task_t,
              compatible_exception_policy_ref_t,
              void>);
static_assert(!yorch::catch_policy_compatible_task<
              policy_payload_bound_task_t,
              incompatible_exception_policy_rvalue_ref_t,
              void>);
static_assert(yorch::catch_policy_compatible_direct_output_task<
              direct_output_task_t,
              compatible_direct_output_policy_t,
              void>);
static_assert(yorch::catch_policy_compatible_direct_output_task<
              direct_output_task_t,
              compatible_direct_output_policy_ref_t,
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
        [](const std::exception_ptr& ep) noexcept -> yorch::task_result<int> {
            try {
                std::rethrow_exception(ep);
            } catch (const std::runtime_error&) {
                return yorch::task_result<int>::failure();
            } catch (...) {
                return yorch::task_result<int>::abort_branch();
            }
            return yorch::task_result<int>::failure();
        });

    const auto raw = task.invoke_raw(exec);

    EXPECT_EQ(raw.step.status, yorch::step_status::failure);
    EXPECT_FALSE(raw.has_value());
}

TEST(TaskAdaptersTest, CatchAsFailureAlsoAcceptsByValueFallbackPolicy) {
    yorch::exec_context<void> exec;

    auto task = yorch::catch_as_failure(
        yorch::bind(
            []() -> yorch::task_result<int> {
                throw std::runtime_error("boom");
            }),
        // NOLINTNEXTLINE(performance-unnecessary-value-param)
        [](std::exception_ptr ep) noexcept -> yorch::task_result<int> {
            try {
                std::rethrow_exception(ep);
            } catch (const std::runtime_error&) {
                return yorch::task_result<int>::abort_branch();
            } catch (...) {
                return yorch::task_result<int>::failure();
            }
            return yorch::task_result<int>::failure();
        });

    const auto raw = task.invoke_raw(exec);

    EXPECT_EQ(raw.step.status, yorch::step_status::abort_branch);
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
        [](const std::exception_ptr&) noexcept -> int {
            return -7;
        });

    const auto raw = task.invoke_raw(exec);

    EXPECT_EQ(raw, -7);
}

TEST(TaskAdaptersTest, WithRetryFixedPolicyRetriesStepResultUntilSuccess) {
    yorch::exec_context<void> exec;
    int attempts = 0;

    auto task = yorch::with_retry(
        yorch::bind(
            [&]() noexcept -> yorch::step_result {
                ++attempts;
                return attempts < 3
                    ? yorch::step_result::retry()
                    : yorch::step_result::success();
            }),
        yorch::retry_fixed_policy {4});

    const auto raw = task.invoke_raw(exec);

    EXPECT_EQ(raw.status, yorch::step_status::success);
    EXPECT_EQ(attempts, 3);
}

TEST(TaskAdaptersTest, WithRetryFixedPolicyReturnsFailureWhenBudgetIsExhausted) {
    yorch::exec_context<void> exec;
    int attempts = 0;

    auto task = yorch::with_retry(
        yorch::bind(
            [&]() noexcept -> yorch::step_result {
                ++attempts;
                return yorch::step_result::retry();
            }),
        yorch::retry_fixed_policy {2});

    const auto raw = task.invoke_raw(exec);

    EXPECT_EQ(raw.status, yorch::step_status::failure);
    EXPECT_EQ(attempts, 3);
}

TEST(TaskAdaptersTest, WithRetryFixedPolicyPreservesPayloadResults) {
    yorch::exec_context<void> exec;
    int attempts = 0;

    auto task = yorch::with_retry(
        yorch::bind(
            [&]() noexcept -> yorch::task_result<int> {
                ++attempts;

                if (attempts < 2) {
                    return yorch::task_result<int>::retry();
                }

                return yorch::task_result<int>::success(17);
            }),
        yorch::retry_fixed_policy {3});

    const auto raw = task.invoke_raw(exec);

    EXPECT_EQ(raw.step.status, yorch::step_status::success);
    EXPECT_TRUE(raw.has_value());
    EXPECT_EQ(raw.value(), 17);
    EXPECT_EQ(attempts, 2);
}

TEST(TaskAdaptersTest, WithRetryFixedPolicyConvertsPayloadRetryExhaustionToFailure) {
    yorch::exec_context<void> exec;
    int attempts = 0;

    auto task = yorch::with_retry(
        yorch::bind(
            [&]() noexcept -> yorch::task_result<int> {
                ++attempts;
                return yorch::task_result<int>::retry();
            }),
        yorch::retry_fixed_policy {1});

    const auto raw = task.invoke_raw(exec);

    EXPECT_EQ(raw.step.status, yorch::step_status::failure);
    EXPECT_FALSE(raw.has_value());
    EXPECT_EQ(attempts, 2);
}

TEST(TaskAdaptersTest, WithRetryFixedPassthroughPolicyPreservesRetryWhenBudgetIsExhausted) {
    yorch::exec_context<void> exec;
    int attempts = 0;

    auto task = yorch::with_retry(
        yorch::bind(
            [&]() noexcept -> yorch::step_result {
                ++attempts;
                return yorch::step_result::retry();
            }),
        yorch::retry_fixed_passthrough_policy {2});

    const auto raw = task.invoke_raw(exec);

    EXPECT_EQ(raw.status, yorch::step_status::retry);
    EXPECT_EQ(attempts, 3);
}

TEST(TaskAdaptersTest, WithRetryForeverPolicyKeepsRetryingUntilSuccess) {
    yorch::exec_context<void> exec;
    int attempts = 0;

    auto task = yorch::with_retry(
        yorch::bind(
            [&]() noexcept -> yorch::step_result {
                ++attempts;
                return attempts < 4
                    ? yorch::step_result::retry()
                    : yorch::step_result::success();
            }),
        yorch::retry_forever_policy {});

    const auto raw = task.invoke_raw(exec);

    EXPECT_EQ(raw.status, yorch::step_status::success);
    EXPECT_EQ(attempts, 4);
}

TEST(TaskAdaptersTest, CatchAsFailureSupportsDirectOutputTasks) {
    yorch::exec_context<void> exec;

    auto task = yorch::catch_as_failure(
        yorch::bind_into<int>(
            [](yorch::result_out<int>) -> yorch::step_result {
                throw std::runtime_error("boom");
            }));

    yorch::detail::typed_slot<int> slot;
    const auto step = task.invoke_into(exec, yorch::result_out<int> {slot});

    EXPECT_EQ(step.status, yorch::step_status::failure);
    EXPECT_FALSE(slot.has_value());
}

TEST(TaskAdaptersTest, WithRetrySupportsDirectOutputTasks) {
    yorch::exec_context<void> exec;
    int attempts = 0;

    auto task = yorch::with_retry(
        yorch::bind_into<int>(
            [&](yorch::result_out<int> out) noexcept -> yorch::step_result {
                ++attempts;

                if (attempts < 3) {
                    out.emplace(attempts);
                    return yorch::step_result::retry();
                }

                return out.success(17);
            }),
        yorch::retry_fixed_policy {4});

    yorch::detail::typed_slot<int> slot;
    const auto step = task.invoke_into(exec, yorch::result_out<int> {slot});

    EXPECT_TRUE(step.ok());
    EXPECT_TRUE(slot.has_value());
    EXPECT_EQ(slot.get(), 17);
    EXPECT_EQ(attempts, 3);
}
