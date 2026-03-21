#pragma once

#include "yorch/context.hpp"
#include "yorch/result.hpp"

namespace yorch {

class executor {
public:
    template <typename Step>
    [[nodiscard]] constexpr step_result run(Step&&, exec_context&) const noexcept {
        return {};
    }
};

}  // namespace yorch
