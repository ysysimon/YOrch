#pragma once

#include <utility>

namespace yorch {

class schedule_builder;

class chain_builder {
public:
    constexpr chain_builder() = default;

    template <typename Step>
    [[nodiscard]] constexpr auto then(Step&&) const -> chain_builder {
        return {};
    }

    template <typename Step>
    [[nodiscard]] constexpr auto step(Step&&) const -> schedule_builder;
};

class schedule_builder {
public:
    constexpr schedule_builder() = default;

    template <typename Step>
    [[nodiscard]] constexpr auto chain(Step&&) const -> chain_builder {
        return {};
    }

    template <typename Step>
    [[nodiscard]] constexpr auto step(Step&&) const -> schedule_builder {
        return {};
    }
};

template <typename Step>
inline constexpr auto chain_builder::step(Step&&) const -> schedule_builder {
    return {};
}

class schedule_root {
public:
    template <typename Step>
    [[nodiscard]] constexpr auto chain(Step&&) const -> chain_builder {
        return {};
    }

    template <typename Step>
    [[nodiscard]] constexpr auto step(Step&&) const -> schedule_builder {
        return {};
    }
};

inline constexpr schedule_root schedule {};

}  // namespace yorch
