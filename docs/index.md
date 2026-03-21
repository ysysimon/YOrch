# YOrch Documentation

`YOrch` 是一个面向任务编排场景的 `C++20` 库骨架。

## Overview

当前仓库重点提供以下能力：

- 统一的命名空间与公开头文件入口
- 结果类型、执行上下文与调度构建器等基础抽象
- 基于 `CMake` 的安装、测试与文档生成入口

## Public Headers

主要公开头文件位于 `include/yorch/`：

- `yorch.hpp`
- `result.hpp`
- `context.hpp`
- `specs.hpp`
- `resolve.hpp`
- `bind.hpp`
- `executor.hpp`
- `schedule.hpp`
- `traits.hpp`

## Build Docs

```bash
cmake --preset docs
cmake --build --preset docs --target docs
```

生成完成后可打开 `build/docs/docs/html/index.html` 查看文档。
