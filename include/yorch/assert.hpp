#pragma once

#include <cassert>

// YOrch uses a library-scoped assertion macro so consumers can configure
// contract checks consistently for both compiled and header-only targets.
#if defined(YORCH_ENABLE_ASSERTS) && YORCH_ENABLE_ASSERTS
#define YORCH_ASSERT(condition) assert(condition)
#else
#define YORCH_ASSERT(condition) static_cast<void>(0)
#endif
