#include "support.hpp"

using namespace bind_test_support;

TEST(BindTest, FunctionTraitsSupportFreeFunctionsAndLambdas) {
    auto lambda =
        [](int&, const std::string&) -> yorch::step_result {
        return yorch::step_result::success();
    };

    using function_traits = yorch::detail::function_traits<decltype(&add_and_check_positive)>;
    using lambda_traits = yorch::detail::function_traits<decltype(lambda)>;

    static_assert(function_traits::arity == 2);
    static_assert(std::is_same_v<function_traits::result_type, bool>);
    static_assert(std::is_same_v<function_traits::arg<0>, int&>);
    static_assert(std::is_same_v<function_traits::arg<1>, int>);

    static_assert(lambda_traits::arity == 2);
    static_assert(std::is_same_v<lambda_traits::result_type, yorch::step_result>);
    static_assert(std::is_same_v<yorch::detail::nth_arg_t<0, decltype(lambda)>, int&>);
    static_assert(
        std::is_same_v<yorch::detail::nth_arg_t<1, decltype(lambda)>, const std::string&>);

    EXPECT_TRUE((std::is_same_v<yorch::detail::result_t<decltype(lambda)>, yorch::step_result>));
}

TEST(BindTest, MemberFunctionTraitsCaptureReceiverAndBusinessParameters) {
    using member_traits = yorch::detail::member_function_traits<decltype(&member_worker::accumulate)>;
    using const_member_traits = yorch::detail::member_function_traits<decltype(&member_worker::read_state)>;
    using direct_output_member_t = decltype(&member_worker::emit);

    static_assert(member_traits::arity == 3);
    static_assert(std::is_same_v<member_traits::class_type, member_worker>);
    static_assert(std::is_same_v<member_traits::receiver_arg_type, member_worker&>);
    static_assert(std::is_same_v<member_traits::result_type, yorch::step_result>);
    static_assert(std::is_same_v<member_traits::arg<0>, int&>);
    static_assert(std::is_same_v<member_traits::arg<1>, const std::string&>);
    static_assert(std::is_same_v<member_traits::arg<2>, long>);

    static_assert(std::is_same_v<const_member_traits::receiver_arg_type, const member_worker&>);
    static_assert(std::is_same_v<const_member_traits::arg<0>, int>);

    static_assert(yorch::detail::ordinary_member_bind_callable<decltype(&member_worker::accumulate)>);
    static_assert(!yorch::detail::ordinary_bind_callable<decltype(&member_worker::accumulate)>);
    static_assert(yorch::detail::inferable_direct_output_member_callable<direct_output_member_t>);
    static_assert(!yorch::detail::inferable_direct_output_callable<direct_output_member_t>);

    EXPECT_TRUE((std::is_same_v<
                 yorch::detail::member_result_t<decltype(&member_worker::read_state)>,
                 int>));
}
