#include "support.hpp" // IWYU pragma: keep

namespace {

struct noexcept_task {
    static constexpr yorch::step_result invoke_raw(yorch::exec_context<void>&) noexcept {
        return yorch::step_result::success();
    }
};

struct throwing_spec_task {
    static constexpr yorch::step_result invoke_raw(yorch::exec_context<void>&) {
        return yorch::step_result::success();
    }
};

struct mismatched_declared_task {
    using raw_result_type = int;

    static constexpr yorch::step_result invoke_raw(yorch::exec_context<void>&) noexcept {
        return yorch::step_result::success();
    }
};

struct member_executor_worker {
    int seen = 0;
    int base = 0;

    yorch::step_result mutate(int& value, int delta) noexcept {
        seen = value;
        base += delta;
        value += delta;
        return yorch::step_result::success();
    }

    yorch::step_result emit(int& value, yorch::direct_out<int> out) noexcept {
        seen = value;
        value += base;
        return out.success(value);
    }
};

}  // namespace

static_assert(yorch::executable_task<noexcept_task&, void>);
static_assert(!yorch::executable_task<throwing_spec_task&, void>);
static_assert(!yorch::executable_task<mismatched_declared_task&, void>);

using caught_throwing_task_t = decltype(yorch::catch_as_failure(throwing_spec_task {}));
static_assert(yorch::executable_task<caught_throwing_task_t&, void>);

using retrying_task_t = decltype(yorch::with_retry(
    yorch::bind([]() noexcept -> yorch::step_result {
        return yorch::step_result::retry();
    }),
    yorch::retry_fixed_policy {1}));
static_assert(yorch::executable_task<retrying_task_t&, void>);

using direct_output_task_t = decltype(yorch::bind_into<int>(
    [](yorch::direct_out<int> out) noexcept -> yorch::step_result {
        return out.success(1);
    }));
static_assert(!yorch::executable_task<direct_output_task_t&, void>);
static_assert(yorch::executable_direct_output_task<direct_output_task_t&, void>);

using throwing_bound_task_t = decltype(yorch::bind([]() -> yorch::step_result {
    throw std::runtime_error("boom");
}));
static_assert(!yorch::executable_task<throwing_bound_task_t&, void>);

TEST(ExecutorTest, RunTaskExecutesBoundTaskAgainstContext) {
    yorch::context<int, long> ctx(3, 7L);
    yorch::exec_context<decltype(ctx)> exec {ctx};

    auto task = yorch::bind(
           // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
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

TEST(ExecutorTest, RunTaskNormalizesStatusOnlyStepResult) {
    yorch::exec_context<void> exec;

    auto task = yorch::bind(
        []() noexcept -> yorch::step_result {
            return yorch::step_result::abort_branch();
        });

    const auto result = yorch::run_task(task, exec);

    EXPECT_EQ(result.status, yorch::step_status::abort_branch);
}

TEST(ExecutorTest, RunTaskIntoSupportsDirectOutputTasks) {
    yorch::context<int> ctx(4);
    yorch::exec_context<decltype(ctx)> exec {ctx};

    auto task = yorch::bind_into<int>(
        [](int& value, yorch::direct_out<int> out) noexcept -> yorch::step_result {
            value += 3;
            return out.success(value * 2);
        },
        yorch::from_ctx<int>());

    yorch::detail::typed_slot<int> slot;
    const auto result = yorch::run_task_into(task, exec, yorch::direct_out<int> {slot});

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(ctx.get<int>(), 7);
    EXPECT_TRUE(slot.has_value());
    EXPECT_EQ(slot.get(), 14);
}

TEST(ExecutorTest, RunTaskSupportsBoundMemberTask) {
    member_executor_worker worker;
    yorch::context<int> ctx(4);
    yorch::exec_context<decltype(ctx)> exec {ctx};

    auto task = yorch::bind_member(
        &member_executor_worker::mutate,
        yorch::value(std::ref(worker)),
        yorch::from_ctx<int>(),
        yorch::value(3));

    const auto result = yorch::run_task(task, exec);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(worker.seen, 4);
    EXPECT_EQ(worker.base, 3);
    EXPECT_EQ(ctx.get<int>(), 7);
}

TEST(ExecutorTest, RunTaskIntoSupportsBoundMemberDirectOutputTask) {
    member_executor_worker worker;
    worker.base = 2;

    yorch::context<int> ctx(5);
    yorch::exec_context<decltype(ctx)> exec {ctx};

    auto task = yorch::bind_into_member<int>(
        &member_executor_worker::emit,
        yorch::value(std::ref(worker)),
        yorch::from_ctx<int>());

    yorch::detail::typed_slot<int> slot;
    const auto result = yorch::run_task_into(task, exec, yorch::direct_out<int> {slot});

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(worker.seen, 5);
    EXPECT_EQ(ctx.get<int>(), 7);
    EXPECT_TRUE(slot.has_value());
    EXPECT_EQ(slot.get(), 7);
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
