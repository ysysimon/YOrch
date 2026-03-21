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
