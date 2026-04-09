#pragma once

#include <type_traits>

namespace yorch {

struct exec_serial_dfs_recursive_policy {};
struct exec_serial_dfs_explicit_heap_stack_policy {};

} // namespace yorch

namespace yorch::detail {

template <typename ExecPolicy>
concept exec_policy =
    std::is_same_v<std::remove_cvref_t<ExecPolicy>, yorch::exec_serial_dfs_recursive_policy> ||
    std::is_same_v<std::remove_cvref_t<ExecPolicy>, yorch::exec_serial_dfs_explicit_heap_stack_policy>;

} // namespace yorch::detail
