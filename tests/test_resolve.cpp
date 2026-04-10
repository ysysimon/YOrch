#include "yorch/resolve.hpp"
#include "yorch/detail/executor/prev_validation.hpp"

#include <gtest/gtest.h>

#include <string>
#include <type_traits>

TEST(ResolveTest, BindFromLvaluePreservesReferenceBinding) {
    int source = 7;

    auto&& bound = yorch::bind_from_lvalue<int&>(source);
    auto&& const_bound = yorch::bind_from_lvalue<const int&>(source);

    static_assert(std::is_same_v<decltype(bound), int&>);
    static_assert(std::is_same_v<decltype(const_bound), const int&>);

    bound = 11;

    EXPECT_EQ(source, 11);
    EXPECT_EQ(const_bound, 11);
    EXPECT_EQ(&bound, &source);
    EXPECT_EQ(&const_bound, &source);
}

TEST(ResolveTest, BindFromLvalueCopiesValueArguments) {
    int source = 5;

    auto copied = yorch::bind_from_lvalue<long>(source);

    static_assert(std::is_same_v<decltype(copied), long>);

    source = 9;

    EXPECT_EQ(copied, 5);
}

TEST(ResolveTest, BindFromLvalueRejectsRvalueReferenceTargets) {
    static_assert(!yorch::detail::supports_bind_from_lvalue_v<int&&, int>);
    static_assert(!yorch::detail::supports_bind_from_lvalue_v<std::string&&, std::string>);
    static_assert(!yorch::bindable_from_lvalue<int&&, int>);
    static_assert(yorch::detail::supports_bind_from_lvalue_v<int&, int>);
    static_assert(yorch::detail::supports_bind_from_lvalue_v<const std::string&, std::string>);
    static_assert(yorch::detail::supports_bind_from_lvalue_v<std::string, std::string>);
    static_assert(yorch::bindable_from_lvalue<std::string, std::string>);

    SUCCEED();
}

TEST(ResolveTest, ValueConceptsRejectRvalueReferenceTargets) {
    static_assert(!yorch::resolvable_mutable_value<std::string&&, std::string>);
    static_assert(!yorch::resolvable_const_value<std::string&&, std::string>);
    static_assert(yorch::resolvable_mutable_value<const std::string&, std::string>);
    static_assert(yorch::resolvable_mutable_value<std::string, std::string>);
    static_assert(yorch::resolvable_const_value<const std::string&, std::string>);
    static_assert(yorch::resolvable_const_value<std::string, std::string>);

    SUCCEED();
}

TEST(ResolveTest, ResolveFromContextBorrowsOrCopiesAccordingToArg) {
    yorch::context<int, std::string> ctx(3, "job");
    yorch::exec_context<decltype(ctx)> exec {ctx};

    auto&& bound = yorch::resolve_as<int&>(yorch::from_ctx<int>(), exec);
    auto copied = yorch::resolve_as<std::string>(yorch::from_ctx<std::string>(), exec);

    static_assert(std::is_same_v<decltype(bound), int&>);
    static_assert(std::is_same_v<decltype(copied), std::string>);

    bound = 8;
    ctx.get<std::string>() = "updated";

    EXPECT_EQ(ctx.get<int>(), 8);
    EXPECT_EQ(copied, "job");
}

TEST(ResolveTest, ResolveMutableValueSpecSupportsConstRefAndValueBinding) {
    auto spec = yorch::value(std::string("hello"));
    yorch::exec_context<void> exec;

    auto&& bound = yorch::resolve_as<const std::string&>(spec, exec);
    auto copied = yorch::resolve_as<std::string>(spec, exec);

    static_assert(std::is_same_v<decltype(bound), const std::string&>);
    static_assert(std::is_same_v<decltype(copied), std::string>);

    spec.v = "updated";

    EXPECT_EQ(bound, "updated");
    EXPECT_EQ(copied, "hello");
    EXPECT_EQ(bound.data(), spec.v.data());
}

TEST(ResolveTest, ResolveConstValueSpecPreservesConstSourceRules) {
    const auto spec = yorch::value(std::string("fixed"));
    yorch::exec_context<void> exec;

    auto&& bound = yorch::resolve_as<const std::string&>(spec, exec);
    auto copied = yorch::resolve_as<std::string>(spec, exec);

    static_assert(std::is_same_v<decltype(bound), const std::string&>);
    static_assert(std::is_same_v<decltype(copied), std::string>);

    EXPECT_EQ(bound, "fixed");
    EXPECT_EQ(copied, "fixed");
    EXPECT_EQ(bound.data(), spec.v.data());
}

TEST(ResolveTest, ResolveBorrowPrevMutBorrowsMutableDirectParentSlot) {
    int parent_value = 4;
    yorch::exec_context<void, decltype(yorch::prev_slot(parent_value))> exec {
        yorch::prev_slot(parent_value)};

    auto&& bound = yorch::resolve_as<int&>(yorch::borrow_prev_mut<int>(), exec);

    static_assert(std::is_same_v<decltype(bound), int&>);

    bound = 9;

    EXPECT_EQ(parent_value, 9);
}

TEST(ResolveTest, ResolveBorrowPrevPreservesConstParentSourceRules) {
    const std::string parent_value = "root";
    yorch::exec_context<void, decltype(yorch::prev_slot(parent_value))> exec {
        yorch::prev_slot(parent_value)};

    auto&& bound =
        yorch::resolve_as<const std::string&>(yorch::borrow_prev<std::string>(), exec);

    static_assert(std::is_same_v<decltype(bound), const std::string&>);

    EXPECT_EQ(bound, "root");
    EXPECT_EQ(bound.data(), parent_value.data());
}

TEST(ResolveTest, ResolveConsumePrevMovesFromMutableDirectParentSlot) {
    std::string parent_value = "root";
    yorch::exec_context<void, decltype(yorch::prev_slot(parent_value))> exec {
        yorch::prev_slot(parent_value)};

    auto consumed = yorch::resolve_as<std::string>(yorch::consume_prev<std::string>(), exec);

    static_assert(std::is_same_v<decltype(consumed), std::string>);

    EXPECT_EQ(consumed, "root");
    EXPECT_TRUE(parent_value.empty());
}

TEST(ResolveTest, ResolveConsumePrevSupportsRvalueReferenceTargets) {
    std::string parent_value = "root";
    yorch::exec_context<void, decltype(yorch::prev_slot(parent_value))> exec {
        yorch::prev_slot(parent_value)};

    auto&& consumed =
        yorch::resolve_as<std::string&&>(yorch::consume_prev<std::string>(), exec);

    static_assert(std::is_same_v<decltype(consumed), std::string&&>);

    consumed = "moved";

    EXPECT_EQ(parent_value, "moved");
}

TEST(ResolveTest, PrevAccessBindingValidationMatchesDeclaredAccessModes) {
    static_assert(yorch::detail::prev_access_binding_valid_v<
                  yorch::borrow_prev_t<int>,
                  const int&>);
    static_assert(!yorch::detail::prev_access_binding_valid_v<
                  yorch::borrow_prev_t<int>,
                  int&>);
    static_assert(!yorch::detail::prev_access_binding_valid_v<
                  yorch::borrow_prev_t<int>,
                  int>);

    static_assert(yorch::detail::prev_access_binding_valid_v<
                  yorch::borrow_prev_mut_t<int>,
                  int&>);
    static_assert(!yorch::detail::prev_access_binding_valid_v<
                  yorch::borrow_prev_mut_t<int>,
                  const int&>);
    static_assert(!yorch::detail::prev_access_binding_valid_v<
                  yorch::borrow_prev_mut_t<int>,
                  int&&>);

    static_assert(yorch::detail::prev_access_binding_valid_v<
                  yorch::consume_prev_t<int>,
                  int>);
    static_assert(yorch::detail::prev_access_binding_valid_v<
                  yorch::consume_prev_t<int>,
                  int&&>);
    static_assert(!yorch::detail::prev_access_binding_valid_v<
                  yorch::consume_prev_t<int>,
                  const int&>);
    static_assert(!yorch::detail::prev_access_binding_valid_v<
                  yorch::consume_prev_t<int>,
                  int&>);

    SUCCEED();
}
