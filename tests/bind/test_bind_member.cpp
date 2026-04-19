#include "support.hpp"

using namespace bind_test_support;

namespace {

static_assert(yorch::detail::validate_bind_member<
                  decltype(&member_worker::accumulate),
                  yorch::from_ctx_t<member_worker>,
                  yorch::from_ctx_t<int>,
                  yorch::from_ctx_t<std::string>,
                  yorch::value_t<int>>() == yorch::detail::bind_member_error::ok);
static_assert(yorch::detail::validate_bind_member<
                  decltype([](int) noexcept -> yorch::step_result {
                      return yorch::step_result::success();
                  }),
                  yorch::from_ctx_t<member_worker>,
                  yorch::value_t<int>>() == yorch::detail::bind_member_error::callable_not_member);
static_assert(yorch::detail::validate_bind_member<
                  decltype(&member_worker::advance_and_emit),
                  yorch::from_ctx_t<member_worker>,
                  yorch::value_t<int>>() == yorch::detail::bind_member_error::direct_output_member_not_supported);
static_assert(yorch::detail::validate_bind_member<
                  decltype(&member_worker::accumulate),
                  yorch::from_ctx_t<member_worker>,
                  yorch::value_t<int>>() == yorch::detail::bind_member_error::arity_mismatch);

static_assert(yorch::detail::validate_bind_into_member<
                  int,
                  decltype(&member_worker::advance_and_emit),
                  yorch::from_ctx_t<member_worker>,
                  yorch::value_t<int>>() == yorch::detail::bind_into_member_error::ok);
static_assert(yorch::detail::validate_bind_into_member<
                  void,
                  decltype(&member_worker::advance_and_emit),
                  yorch::from_ctx_t<member_worker>,
                  yorch::value_t<int>>() == yorch::detail::bind_into_member_error::invalid_output_type);
static_assert(yorch::detail::validate_bind_into_member<
                  int&,
                  decltype(&member_worker::advance_and_emit),
                  yorch::from_ctx_t<member_worker>,
                  yorch::value_t<int>>() == yorch::detail::bind_into_member_error::invalid_output_type);
static_assert(yorch::detail::validate_bind_into_member<
                  int,
                  decltype([](int) noexcept -> yorch::step_result {
                      return yorch::step_result::success();
                  }),
                  yorch::from_ctx_t<member_worker>,
                  yorch::value_t<int>>() == yorch::detail::bind_into_member_error::callable_not_member);
static_assert(yorch::detail::validate_bind_into_member<
                  int,
                  decltype(&member_worker::note),
                  yorch::from_ctx_t<member_worker>>() == yorch::detail::bind_into_member_error::last_parameter_not_direct_out);
static_assert(yorch::detail::validate_bind_into_member<
                  int,
                  decltype(&move_only_member_worker::emit),
                  yorch::from_ctx_t<move_only_member_worker>,
                  yorch::value_t<int>>() == yorch::detail::bind_into_member_error::arity_mismatch);
static_assert(yorch::detail::validate_bind_forward_prev_member<
                  member_worker,
                  decltype(&member_worker::mutate_state),
                  yorch::borrow_prev_mut_t<member_worker>,
                  yorch::value_t<int>>() == yorch::detail::bind_forward_prev_member_error::ok);
static_assert(yorch::detail::validate_bind_forward_prev_member<
                  void,
                  decltype(&member_worker::mutate_state),
                  yorch::borrow_prev_mut_t<member_worker>,
                  yorch::value_t<int>>() == yorch::detail::bind_forward_prev_member_error::invalid_output_type);
static_assert(yorch::detail::validate_bind_forward_prev_member<
                  member_worker,
                  decltype([](int) noexcept -> yorch::step_result {
                      return yorch::step_result::success();
                  }),
                  yorch::value_t<int>,
                  yorch::value_t<int>>() == yorch::detail::bind_forward_prev_member_error::callable_not_member);
static_assert(yorch::detail::validate_bind_forward_prev_member<
                  int,
                  decltype(&member_worker::advance_and_emit),
                  yorch::from_ctx_t<member_worker>,
                  yorch::value_t<int>>() == yorch::detail::bind_forward_prev_member_error::direct_output_member_not_supported);
static_assert(yorch::detail::validate_bind_forward_prev_member<
                  int,
                  decltype(&member_worker::accumulate),
                  yorch::from_ctx_t<member_worker>,
                  yorch::borrow_prev_mut_t<int>>() == yorch::detail::bind_forward_prev_member_error::arity_mismatch);
static_assert(yorch::detail::validate_bind_forward_prev_member<
                  int,
                  decltype(&member_worker::mutate_state),
                  yorch::value_t<std::reference_wrapper<member_worker>>,
                  yorch::consume_prev_t<int>>() == yorch::detail::bind_forward_prev_member_error::binding_mode_not_supported);
static_assert(yorch::detail::validate_bind_forward_prev_member<
                  member_worker,
                  decltype(&member_worker::mutate_state),
                  yorch::borrow_prev_t<member_worker>,
                  yorch::value_t<int>>() == yorch::detail::bind_forward_prev_member_error::binding_mode_not_supported);

} // namespace

TEST(BindTest, BoundMemberTaskResolvesReceiverFromContext) {
    yorch::context<member_worker, int, std::string> ctx(member_worker {}, 3, "job");
    yorch::exec_context<decltype(ctx)> exec {ctx};

    auto task = yorch::bind_member(
        &member_worker::accumulate,
        yorch::from_ctx<member_worker>(),
        yorch::from_ctx<int>(),
        yorch::from_ctx<std::string>(),
        yorch::value(4));

    static_assert(std::is_same_v<decltype(task.receiver_spec), yorch::from_ctx_t<member_worker>>);
    static_assert(std::is_same_v<
                  decltype(task.specs),
                  std::tuple<
                      yorch::from_ctx_t<int>,
                      yorch::from_ctx_t<std::string>,
                      yorch::value_t<int>>>);

    const auto result = task.invoke_raw(exec);
    auto& worker = ctx.get<member_worker>();

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(worker.seen_value, 3);
    ASSERT_NE(worker.seen_label, nullptr);
    EXPECT_EQ(worker.seen_label, &ctx.get<std::string>());
    EXPECT_EQ(worker.state, 4);
    EXPECT_EQ(ctx.get<int>(), 7);
}

TEST(BindTest, BoundMemberTaskSupportsReferenceWrapperReceiverStorage) {
    member_worker worker;
    yorch::exec_context<void> exec;

    auto task = yorch::bind_member(
        &member_worker::note,
        yorch::value(std::ref(worker)),
        yorch::value(7));

    const auto result = task.invoke_raw(exec);

    EXPECT_TRUE(result);
    EXPECT_EQ(worker.seen_value, 7);
    EXPECT_EQ(worker.state, 7);
}

TEST(BindTest, BoundMemberTaskCanBorrowConstReceiverFromDirectParentSlot) {
    member_worker worker;
    worker.state = 11;

    yorch::exec_context<void, decltype(yorch::prev_slot(worker))> exec {
        yorch::prev_slot(worker)};

    auto task = yorch::bind_member(
        &member_worker::read_state,
        yorch::borrow_prev<member_worker>(),
        yorch::value(4));

    const auto result = task.invoke_raw(exec);

    EXPECT_EQ(result, 15);
}

TEST(BindTest, BoundMemberTaskCanBorrowMutableReceiverFromDirectParentSlot) {
    member_worker worker;
    worker.state = 3;

    yorch::exec_context<void, decltype(yorch::prev_slot(worker))> exec {
        yorch::prev_slot(worker)};

    auto task = yorch::bind_member(
        &member_worker::mutate_state,
        yorch::borrow_prev_mut<member_worker>(),
        yorch::value(5));

    const auto result = task.invoke_raw(exec);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(worker.state, 8);
}

TEST(BindTest, BoundMemberTaskCanCopyDirectParentReceiverWithoutMutatingSource) {
    member_worker worker;
    worker.state = 9;

    yorch::exec_context<void, decltype(yorch::prev_slot(worker))> exec {
        yorch::prev_slot(worker)};

    auto task = yorch::bind_member(
        &member_worker::mutate_state,
        yorch::copy_prev<member_worker>(),
        yorch::value(4));

    const auto result = task.invoke_raw(exec);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(worker.state, 9);
}

TEST(BindTest, BoundMemberTaskCanConsumeDirectParentReceiver) {
    move_only_member_worker worker {7};

    yorch::exec_context<void, decltype(yorch::prev_slot(worker))> exec {
        yorch::prev_slot(worker)};

    auto task = yorch::bind_member(
        &move_only_member_worker::bump,
        yorch::consume_prev<move_only_member_worker>(),
        yorch::value(5));

    const auto result = task.invoke_raw(exec);

    EXPECT_EQ(result, 12);
}

TEST(BindTest, BoundConstMemberTaskSupportsCopyAndConsumeReceivers) {
    member_worker copy_worker;
    copy_worker.state = 8;

    yorch::exec_context<void, decltype(yorch::prev_slot(copy_worker))> copy_exec {
        yorch::prev_slot(copy_worker)};

    auto copy_task = yorch::bind_member(
        &member_worker::read_state,
        yorch::copy_prev<member_worker>(),
        yorch::value(3));

    EXPECT_EQ(copy_task.invoke_raw(copy_exec), 11);

    move_only_member_worker consume_worker {10};
    yorch::exec_context<void, decltype(yorch::prev_slot(consume_worker))> consume_exec {
        yorch::prev_slot(consume_worker)};

    auto consume_task = yorch::bind_into_member<int>(
        &move_only_member_worker::emit,
        yorch::consume_prev<move_only_member_worker>());

    yorch::detail::typed_slot<int> slot;
    const auto step = consume_task.invoke_into(consume_exec, yorch::direct_out<int> {slot});

    EXPECT_TRUE(step.ok());
    EXPECT_TRUE(slot.has_value());
    EXPECT_EQ(slot.get(), 10);
}

TEST(BindTest, BoundMemberOutputTaskResolvesReceiverAndWritesToOutputSink) {
    member_worker worker;
    worker.state = 2;

    yorch::context<member_worker, int> ctx(worker, 6);
    yorch::exec_context<decltype(ctx)> exec {ctx};

    auto task = yorch::bind_into_member<std::string>(
        &member_worker::emit,
        yorch::from_ctx<member_worker>(),
        yorch::from_ctx<int>());

    yorch::detail::typed_slot<std::string> slot;
    const auto result = task.invoke_into(exec, yorch::direct_out<std::string> {slot});

    EXPECT_TRUE(result.ok());
    EXPECT_TRUE(slot.has_value());
    EXPECT_EQ(slot.get(), "8");
    EXPECT_EQ(ctx.get<int>(), 8);
    EXPECT_EQ(ctx.get<member_worker>().seen_value, 6);
}

TEST(BindTest, BoundMemberOutputTaskSupportsCopyAndConsumeReceivers) {
    member_worker copy_worker;
    copy_worker.state = 5;

    yorch::exec_context<void, decltype(yorch::prev_slot(copy_worker))> copy_exec {
        yorch::prev_slot(copy_worker)};

    auto copy_task = yorch::bind_into_member<int>(
        &member_worker::advance_and_emit,
        yorch::copy_prev<member_worker>(),
        yorch::value(4));

    yorch::detail::typed_slot<int> copy_slot;
    const auto copy_result = copy_task.invoke_into(copy_exec, yorch::direct_out<int> {copy_slot});

    EXPECT_TRUE(copy_result.ok());
    EXPECT_TRUE(copy_slot.has_value());
    EXPECT_EQ(copy_slot.get(), 9);
    EXPECT_EQ(copy_worker.state, 5);

    move_only_member_worker consume_worker {13};
    yorch::exec_context<void, decltype(yorch::prev_slot(consume_worker))> consume_exec {
        yorch::prev_slot(consume_worker)};

    auto consume_task = yorch::bind_into_member<int>(
        &move_only_member_worker::emit,
        yorch::consume_prev<move_only_member_worker>());

    yorch::detail::typed_slot<int> consume_slot;
    const auto consume_result =
        consume_task.invoke_into(consume_exec, yorch::direct_out<int> {consume_slot});

    EXPECT_TRUE(consume_result.ok());
    EXPECT_TRUE(consume_slot.has_value());
    EXPECT_EQ(consume_slot.get(), 13);
}

TEST(BindTest, BoundMemberForwardPrevTaskCanForwardReceiverFromBorrowPrevMut) {
    member_worker worker;
    worker.state = 9;

    yorch::exec_context<void, decltype(yorch::prev_slot(worker))> exec {
        yorch::prev_slot(worker)};

    auto task = yorch::bind_forward_prev_member<member_worker>(
        &member_worker::mutate_state,
        yorch::borrow_prev_mut<member_worker>(),
        yorch::value(4));

    static_assert(std::is_same_v<
                  decltype(task.receiver_spec),
                  yorch::borrow_prev_mut_t<member_worker>>);
    static_assert(std::is_same_v<
                  decltype(task.specs),
                  std::tuple<yorch::value_t<int>>>);
    static_assert(yorch::detail::task_uses_forward_prev_output_protocol_v<decltype(task)>);
    static_assert(yorch::detail::task_prev_access_valid_v<decltype(task)>);
    static_assert(yorch::detail::task_uses_borrow_prev_mut_v<decltype(task)>);

    const auto result = task.invoke_raw(exec);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(worker.state, 13);
}

TEST(BindTest, BoundMemberForwardPrevTaskCanForwardMutableParameterFromContextReceiver) {
    struct service {
        int* seen = nullptr;

        yorch::step_result update(forward_prev_probe& probe, int delta) noexcept {
            seen = &probe.value;
            probe.value += delta;
            return yorch::step_result::success();
        }
    };

    service svc;
    forward_prev_probe probe {6};

    yorch::context<service> ctx(svc);
    yorch::exec_context<decltype(ctx), decltype(yorch::prev_slot(probe))> exec {
        ctx,
        yorch::prev_slot(probe)};

    auto task = yorch::bind_forward_prev_member<forward_prev_probe>(
        &service::update,
        yorch::from_ctx<service>(),
        yorch::borrow_prev_mut<forward_prev_probe>(),
        yorch::value(3));

    static_assert(yorch::detail::task_uses_forward_prev_output_protocol_v<decltype(task)>);
    static_assert(yorch::detail::task_prev_access_valid_v<decltype(task)>);
    static_assert(yorch::detail::task_uses_borrow_prev_mut_v<decltype(task)>);

    const auto result = task.invoke_raw(exec);

    EXPECT_TRUE(result.ok());
    EXPECT_EQ(probe.value, 9);
    ASSERT_NE(ctx.get<service>().seen, nullptr);
    EXPECT_EQ(ctx.get<service>().seen, &probe.value);
}
