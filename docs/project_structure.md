# Project Structure

This document is a high-level project structure sketch. It is not a complete or authoritative list of source files; individual implementation files may be added, removed, or reorganized without requiring this page to mirror every file-level change.

```text
YOrch/
  CMakeLists.txt           # Top-level build configuration
  CMakePresets.json        # Presets for build, test, and docs workflows
  cmake/                   # CMake package/configuration helpers
  include/
    yorch/                 # Public C++ headers and header-only internals
  src/                     # Library implementation sources
  tests/                   # Unit and integration tests
  examples/                # Small usage examples
  docs/                    # Generated-documentation inputs
  scripts/                 # Project maintenance scripts
  wiki/                    # Human-facing explanatory notes
  README.md
  LICENSE
  AGENTS.md
```

Document build entrance:

```bash
cmake --preset docs
cmake --build --preset docs --target docs
```
