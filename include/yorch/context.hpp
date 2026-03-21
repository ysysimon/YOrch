#pragma once

namespace yorch {

struct context {};

struct prev_store {
    
    template <typename T>
    [[nodiscard]] constexpr bool contains() const noexcept {
        return false;
    }

    template <typename T>
    T& get();

    template <typename T>
    const T& get() const;

    template <typename T>
    T take();
};

struct exec_context {
    context* ctx = nullptr;
    prev_store* prev = nullptr;
};

}  // namespace yorch
