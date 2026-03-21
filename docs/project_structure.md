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
      schedule.hpp
      traits.hpp
  src/
    yorch.cpp
  tests/
    CMakeLists.txt
    test_result.cpp
    test_bind.cpp
    test_schedule.cpp
  docs/
    Doxyfile.in             # Doxygen 配置模板
    index.md                # 文档首页
    project_structure.md
  README.md
  LICENSE
  AGENTS.md
```

文档构建入口：

```bash
cmake --preset docs
cmake --build --preset docs --target docs
```
