# YOrch Documentation

`YOrch` is a `C++20` library skeleton designed for task orchestration scenarios.

## Overview

The current repository focuses on the following capabilities:

- A unified namespace and a public header entry point
- Core abstractions such as result types, execution contexts, and scheduler builders
- `CMake`-based entry points for installation, testing, and documentation generation
- An overview document for the current execution model

## Public Headers

The main public headers are located under `include/yorch/`:

- `yorch.hpp`
- `result.hpp`
- `context.hpp`
- `specs.hpp`
- `resolve.hpp`
- `bind.hpp`
- `executor.hpp`
- `schedule.hpp`
- `traits.hpp`

## Additional Docs

- `execution_model.md`

## Build Docs

```bash
cmake --preset docs
cmake --build --preset docs --target docs
```

After the build completes, open `build/docs/docs/html/index.html` to view the generated documentation.
