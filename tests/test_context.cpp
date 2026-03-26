#include "yorch/context.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <type_traits>
#include <utility>

namespace {

struct move_only {
    int value = 0;

    constexpr move_only() = default;
    constexpr explicit move_only(int in) : value(in) {}

    move_only(const move_only&) = delete;
    move_only& operator=(const move_only&) = delete;
    constexpr move_only(move_only&&) noexcept = default;
    constexpr move_only& operator=(move_only&&) noexcept = default;

    ~move_only() = default;
};

struct no_default {
    explicit no_default(int in) : value(in) {}
    int value;
};

}  // namespace

TEST(ContextTest, ContainsReportsSchemaMembership) {
    using test_context = yorch::context<int, std::string, double>;

    static_assert(test_context::contains<int>());
    static_assert(test_context::contains<std::string>());
    static_assert(!test_context::contains<float>());

    EXPECT_TRUE((test_context::contains<double>()));
    EXPECT_FALSE((test_context::contains<char>()));
}

TEST(ContextTest, DefaultConstructionIsAvailableOnlyForDefaultInitializableSchemas) {
    static_assert(std::default_initializable<yorch::context<int, std::string>>);
    static_assert(!std::default_initializable<yorch::context<no_default>>);

    yorch::context<int, std::string> ctx;

    EXPECT_EQ(ctx.get<int>(), 0);
    EXPECT_TRUE(ctx.get<std::string>().empty());
}

TEST(ContextTest, ExplicitConstructionStoresAndReturnsValuesByType) {
    yorch::context<int, std::string, std::unique_ptr<int>> ctx(
        7,
        "hello",
        std::make_unique<int>(42));

    EXPECT_EQ(ctx.get<int>(), 7);
    EXPECT_EQ(ctx.get<std::string>(), "hello");
    ASSERT_NE(ctx.get<std::unique_ptr<int>>(), nullptr);
    EXPECT_EQ(*ctx.get<std::unique_ptr<int>>(), 42);

    ctx.get<int>() = 9;
    ctx.get<std::string>().append(" world");
    *ctx.get<std::unique_ptr<int>>() = 100;

    EXPECT_EQ(ctx.get<int>(), 9);
    EXPECT_EQ(ctx.get<std::string>(), "hello world");
    EXPECT_EQ(*ctx.get<std::unique_ptr<int>>(), 100);
}

TEST(ContextTest, GetPreservesReferenceAndConstQualifications) {
    using test_context = yorch::context<int, move_only>;

    static_assert(std::is_same_v<decltype(std::declval<test_context&>().get<int>()), int&>);
    static_assert(std::is_same_v<decltype(std::declval<const test_context&>().get<int>()), const int&>);
    static_assert(std::is_same_v<decltype(std::declval<test_context&&>().get<int>()), int&&>);
    static_assert(std::is_same_v<decltype(std::declval<const test_context&&>().get<int>()), const int&&>);

    test_context ctx(11, move_only {22});
    const test_context& const_ctx = ctx;

    EXPECT_EQ(ctx.get<int>(), 11);
    EXPECT_EQ(const_ctx.get<move_only>().value, 22);

    move_only moved = std::move(ctx).get<move_only>();
    EXPECT_EQ(moved.value, 22);
}

TEST(ContextTest, ExecContextBorrowsAnExistingContext) {
    yorch::context<int, std::string> ctx(3, "task");
    yorch::exec_context<decltype(ctx)> exec {ctx};

    exec.ctx.get<int>() = 5;
    exec.ctx.get<std::string>() = "updated";

    EXPECT_EQ(ctx.get<int>(), 5);
    EXPECT_EQ(ctx.get<std::string>(), "updated");

    static_assert(std::is_empty_v<yorch::exec_context<void>>);
}

TEST(ContextTest, PrevSlotViewBorrowsDirectParentByReference) {
    int parent_value = 7;
    auto view = yorch::prev_slot(parent_value);

    static_assert(std::is_same_v<decltype(view.get<int>()), int&>);

    view.get<int>() = 11;

    EXPECT_EQ(parent_value, 11);
}

TEST(ContextTest, PrevSlotViewReportsContainedTypeByNormalizedPayloadType) {
    int parent_value = 7;
    using view_t = decltype(yorch::prev_slot(parent_value));

    static_assert(view_t::contains<int>());
    static_assert(view_t::contains<const int>());
    static_assert(!view_t::contains<long>());

    EXPECT_TRUE((view_t::contains<volatile int>()));
    EXPECT_FALSE((view_t::contains<std::string>()));
}

TEST(ContextTest, PrevSlotViewGetPreservesUnderlyingPayloadConstness) {
    int mutable_parent = 7;
    const int const_parent = 9;

    auto mutable_view = yorch::prev_slot(mutable_parent);
    auto const_view = yorch::prev_slot(const_parent);

    static_assert(std::is_same_v<decltype(mutable_view.get<int>()), int&>);
    static_assert(std::is_same_v<decltype(std::as_const(mutable_view).get<int>()), int&>);
    static_assert(std::is_same_v<decltype(const_view.get<int>()), const int&>);
    static_assert(std::is_same_v<decltype(std::as_const(const_view).get<int>()), const int&>);

    mutable_view.get<int>() = 13;

    EXPECT_EQ(mutable_parent, 13);
    EXPECT_EQ(const_view.get<int>(), 9);
}

TEST(ContextTest, ExecContextPrevViewReturnsStoredPrevSlotView) {
    yorch::context<int> ctx(3);
    int parent_value = 5;
    yorch::exec_context<decltype(ctx), decltype(yorch::prev_slot(parent_value))> exec {
        ctx,
        yorch::prev_slot(parent_value)};

    auto&& view = exec.prev_view();

    static_assert(std::is_same_v<decltype(view), decltype(exec.prev)&>);
    static_assert(std::is_same_v<decltype(std::as_const(exec).prev_view()), const decltype(exec.prev)&>);

    view.get<int>() = 8;

    EXPECT_EQ(parent_value, 8);
}

TEST(ContextTest, ExecContextVoidPrevViewReturnsStoredPrevSlotView) {
    int parent_value = 5;
    yorch::exec_context<void, decltype(yorch::prev_slot(parent_value))> exec {
        yorch::prev_slot(parent_value)};

    auto&& view = exec.prev_view();

    static_assert(std::is_same_v<decltype(view), decltype(exec.prev)&>);
    static_assert(std::is_same_v<decltype(std::as_const(exec).prev_view()), const decltype(exec.prev)&>);

    view.get<int>() = 10;

    EXPECT_EQ(parent_value, 10);
}

TEST(ContextTest, ExecContextWithoutPrevUsesNoPrevView) {
    yorch::context<int> ctx(3);
    yorch::exec_context<decltype(ctx)> exec {ctx};
    yorch::exec_context<void> void_exec;

    static_assert(std::is_same_v<decltype(exec.prev_view()), yorch::no_prev>);
    static_assert(std::is_same_v<decltype(std::as_const(exec).prev_view()), yorch::no_prev>);
    static_assert(std::is_same_v<decltype(void_exec.prev_view()), yorch::no_prev>);
    static_assert(std::is_same_v<decltype(std::as_const(void_exec).prev_view()), yorch::no_prev>);

    EXPECT_FALSE((decltype(exec.prev_view())::contains<int>()));
    EXPECT_FALSE((decltype(void_exec.prev_view())::contains<int>()));
}
