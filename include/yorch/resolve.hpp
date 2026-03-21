#pragma once

#include "yorch/context.hpp"

namespace yorch::detail {

template <typename Arg, typename Spec>
decltype(auto) resolve_as(Spec&& spec, exec_context& ec);

}  // namespace yorch::detail
