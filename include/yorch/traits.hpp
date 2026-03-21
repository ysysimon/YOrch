#pragma once

#include <cstddef>

namespace yorch::detail {

template <typename F>
struct function_traits;

template <std::size_t I, typename F>
using nth_arg_t = typename function_traits<F>::template arg<I>;

}  // namespace yorch::detail
