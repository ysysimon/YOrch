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

`run_plan(...)` 当前支持两个独立的 policy 维度：

- `LayoutPolicy`：控制 payload slot 的布局方式
- `ExecPolicy`：控制 plan 的遍历执行方式

默认执行策略是 recursive serial DFS；如需在深树场景下避免依赖函数调用栈深度，可显式选择 `exec_serial_dfs_explicit_heap_stack_policy`。

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

## Plan Types

当用户需要在自己的类里长期保存 compiled plan 时，推荐做法是：

- 用独立的 `make_tree(...)` 函数或类内 `static make_tree(...)` 构造 `task_tree_builder`
- 用 `yorch::compiled_plan_t<Tree>` 命名成员变量类型
- 在构造函数里调用 `yorch::compile_plan(...)` 初始化该成员

示例：

```cpp
inline auto make_demo_tree(int value) {
    return yorch::task_tree
        .root(yorch::bind([value]() noexcept -> int {
            return value;
        }))
        .node<1>(yorch::bind([](int) noexcept {},
                             yorch::from_prev<int>()));
}

template <typename Tree>
class runner {
public:
    using plan_type = yorch::compiled_plan_t<Tree>;

    explicit runner(Tree&& tree)
        : plan_(yorch::compile_plan(std::forward<Tree>(tree))) {}

private:
    plan_type plan_;
};

auto tree = make_demo_tree(42);
runner r(std::move(tree));
```

如果 `tree` 只是 compile 的一次性输入，通常没有必要把 `task_tree_builder` 也作为成员长期保存。

## Roadmap

1. `Phase 1`：补齐 `step_result`、`exec_context`、`value/from_ctx`、`bind`
2. `Phase 2`：补齐 `task_result<T>`、`PrevStore`、`from_prev<T>`、链式结果传递
3. `Phase 3`：补齐 `schedule/chain/then/step` 的最小执行闭环
4. `Phase 4`：补齐 `traits` 与自动参数/返回值推导
