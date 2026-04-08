#pragma once

#include <type_traits>

namespace yorch {

struct slot_layout_one_to_one_policy {};
struct slot_layout_serial_dfs_compact_policy {};

} // namespace yorch

namespace yorch::detail {

template <typename LayoutPolicy>
concept slot_layout_policy =
    std::is_same_v<std::remove_cvref_t<LayoutPolicy>, yorch::slot_layout_one_to_one_policy> ||
    std::is_same_v<std::remove_cvref_t<LayoutPolicy>, yorch::slot_layout_serial_dfs_compact_policy>;

} // namespace yorch::detail
