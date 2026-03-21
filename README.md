# YOrch

`YOrch` 是一个面向任务编排场景的 `C++` 库骨架。当前仓库先固定项目结构、命名约定和构建入口，不提前承诺具体执行语义。

## Naming

- 项目名：`YOrch`
- 命名空间：`yorch`
- 内部实现命名空间：`yorch::detail`

## CMake Targets

- `yorch::yorch`
  轻量编译库入口，适合后续承接非模板实现或显式实例化。
- `yorch::yorch_header_only`
  纯头文件入口，适合当前以模板和 `inline` 为主的使用方式。
- `docs`
  `Doxygen` 文档生成目标，仅在配置时启用 `YORCH_BUILD_DOCS=ON` 后可用。

示例：

```cmake
target_link_libraries(app PRIVATE yorch::yorch)
```

或：

```cmake
target_link_libraries(app PRIVATE yorch::yorch_header_only)
```

## Current Modules

- `result.hpp`
- `context.hpp`
- `specs.hpp`
- `resolve.hpp`
- `bind.hpp`
- `executor.hpp`
- `schedule.hpp`
- `traits.hpp`
- `yorch.hpp`

## Documentation

生成 API 文档：

```bash
cmake --preset docs
cmake --build --preset docs --target docs
```

生成完成后，首页位于 `build/docs/docs/html/index.html`。

## Roadmap

1. `Phase 1`：补齐 `step_result`、`exec_context`、`value/from_ctx`、`bind`
2. `Phase 2`：补齐 `task_result<T>`、`PrevStore`、`from_prev<T>`、链式结果传递
3. `Phase 3`：补齐 `schedule/chain/then/step` 的最小执行闭环
4. `Phase 4`：补齐 `traits` 与自动参数/返回值推导
