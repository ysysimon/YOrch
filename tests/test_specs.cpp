#include "yorch/specs.hpp"

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

}  // namespace

TEST(SpecsTest, FromCtxNormalizesCvrefTypes) {
    constexpr auto int_spec = yorch::from_ctx<const int&>();
    constexpr auto string_spec = yorch::from_ctx<volatile std::string&&>();

    static_assert(std::is_same_v<decltype(int_spec), const yorch::from_ctx_t<int>>);
    static_assert(std::is_same_v<typename decltype(int_spec)::type, int>);
    static_assert(std::is_same_v<typename decltype(string_spec)::type, std::string>);

    EXPECT_TRUE((std::is_empty_v<decltype(int_spec)>));
    EXPECT_TRUE((std::is_empty_v<decltype(string_spec)>));
}

TEST(SpecsTest, BorrowPrevNormalizesCvrefTypes) {
    constexpr auto int_spec = yorch::borrow_prev<const int&>();
    constexpr auto string_spec = yorch::borrow_prev<volatile std::string&&>();

    static_assert(std::is_same_v<decltype(int_spec), const yorch::borrow_prev_t<int>>);
    static_assert(std::is_same_v<typename decltype(int_spec)::type, int>);
    static_assert(std::is_same_v<typename decltype(string_spec)::type, std::string>);

    EXPECT_TRUE((std::is_empty_v<decltype(int_spec)>));
    EXPECT_TRUE((std::is_empty_v<decltype(string_spec)>));
}

TEST(SpecsTest, BorrowPrevMutNormalizesCvrefTypes) {
    constexpr auto int_spec = yorch::borrow_prev_mut<const int&>();
    constexpr auto string_spec = yorch::borrow_prev_mut<volatile std::string&&>();

    static_assert(std::is_same_v<decltype(int_spec), const yorch::borrow_prev_mut_t<int>>);
    static_assert(std::is_same_v<typename decltype(int_spec)::type, int>);
    static_assert(std::is_same_v<typename decltype(string_spec)::type, std::string>);

    EXPECT_TRUE((std::is_empty_v<decltype(int_spec)>));
    EXPECT_TRUE((std::is_empty_v<decltype(string_spec)>));
}

TEST(SpecsTest, ConsumePrevNormalizesCvrefTypes) {
    constexpr auto int_spec = yorch::consume_prev<const int&>();
    constexpr auto string_spec = yorch::consume_prev<volatile std::string&&>();

    static_assert(std::is_same_v<decltype(int_spec), const yorch::consume_prev_t<int>>);
    static_assert(std::is_same_v<typename decltype(int_spec)::type, int>);
    static_assert(std::is_same_v<typename decltype(string_spec)::type, std::string>);

    EXPECT_TRUE((std::is_empty_v<decltype(int_spec)>));
    EXPECT_TRUE((std::is_empty_v<decltype(string_spec)>));
}

TEST(SpecsTest, ValueDecaysLvalueReferencesIntoOwnedStorage) {
    std::string source = "hello";

    auto spec = yorch::value(source);

    static_assert(std::is_same_v<decltype(spec), yorch::value_t<std::string>>);
    static_assert(std::is_same_v<typename decltype(spec)::stored_type, std::string>);

    source = "updated";

    EXPECT_EQ(spec.v, "hello");
    EXPECT_NE(spec.v.data(), source.data());
}

TEST(SpecsTest, ValueSupportsMoveOnlyPayloads) {
    auto spec = yorch::value(move_only {42});

    static_assert(std::is_same_v<decltype(spec), yorch::value_t<move_only>>);
    static_assert(std::is_same_v<decltype((std::move(spec).v)), move_only&&>);

    EXPECT_EQ(spec.v.value, 42);
}

TEST(SpecsTest, ValuePreservesPointerLikeOwnershipObjects) {
    auto spec = yorch::value(std::make_unique<int>(7));

    static_assert(std::is_same_v<decltype(spec), yorch::value_t<std::unique_ptr<int>>>);
    ASSERT_NE(spec.v, nullptr);
    EXPECT_EQ(*spec.v, 7);
}
