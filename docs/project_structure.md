```text
YOrch/
  CMakeLists.txt
  CMakePresets.json
  cmake/
    YOrchConfig.cmake.in
  include/
    yorch/
      yorch.hpp
      result.hpp
      context.hpp
      specs.hpp
      resolve.hpp
      bind.hpp
      executor.hpp
      task_tree.hpp
      traits.hpp
  src/
    yorch.cpp
  tests/
    CMakeLists.txt
    test_result.cpp
    test_bind.cpp
    test_task_tree.cpp
  docs/
    Doxyfile.in             # Doxygen config tempalte
    index.md                # Doc index
    execution_model.md
    project_structure.md
  README.md
  LICENSE
  AGENTS.md
```

Document build entrance：

```bash
cmake --preset docs
cmake --build --preset docs --target docs
```
