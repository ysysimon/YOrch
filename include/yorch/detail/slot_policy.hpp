#pragma once

namespace yorch::detail {

enum class slot_policy : unsigned char {
    none,
    maybe_payload,
    must_payload,
};

} // namespace yorch::detail
