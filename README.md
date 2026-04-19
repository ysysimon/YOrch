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
        .root([value]() noexcept -> int {
            return value;
        })()
        .node<1>([](const int&) noexcept {})(yorch::borrow_prev<int>());
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

## Member Functions

成员函数请显式使用 member 专用入口：

- 预绑定 task：`bind_member(...)` / `bind_into_member(...)`
- `forward_prev` 预绑定 task：`bind_forward_prev_member(...)`
- 两段式 sugar：`task_member(...)` / `task_into_member(...)` / `task_forward_prev_member(...)`
- `task_tree` sugar：`root_member(...)` / `root_into_member(...)` / `root_forward_prev_member(...)` / `node_member<Level>(...)` / `node_into_member<Level>(...)` / `node_forward_prev_member<Level>(...)`

不要把 `&Class::method` 直接传给 `bind(...)`、`task(...)`、`task_into(...)`、`task_tree.root(...)` 或 `task_tree.node<...>(...)` 这类 ordinary callable 入口。

普通 callable 的 `forward_prev` sugar 入口也已经单独提供：

- 两段式 sugar：`task_forward_prev(...)`
- `task_tree` sugar：`root_forward_prev(...)` / `node_forward_prev<Level>(...)`

示例：

```cpp
struct worker {
    int run(int value) noexcept { return value + 1; }

    yorch::step_result emit(
        int value,
        yorch::direct_out<std::string> out) noexcept {
        return out.success(std::to_string(value));
    }
};

worker w;

auto task = yorch::bind_member(
    &worker::run,
    yorch::value(std::ref(w)),
    yorch::value(41));

auto direct_output_task = yorch::bind_into_member<std::string>(
    &worker::emit,
    yorch::value(std::ref(w)),
    yorch::value(42));

auto sugar_task = yorch::task_member(
    &worker::run,
    yorch::value(std::ref(w)))(
    yorch::value(41));

auto tree = yorch::task_tree
    .root_member(
        &worker::run,
        yorch::value(std::ref(w)))(
        yorch::value(41))
    .node_into_member<1>(
        &worker::emit,
        yorch::value(std::ref(w)))(
        yorch::value(42));
```

receiver 也可以来自 `context` 或 direct parent：

- `from_ctx<T>()`：从 `context` 借用对象
- `borrow_prev<T>()`：从 direct parent 只读借用对象，只能绑定到 `const member function`
- `borrow_prev_mut<T>()`：从 direct parent 可变借用对象
- `copy_prev<T>()`：从 direct parent 复制一个副本，在副本上调用成员函数，不会回写 parent
- `consume_prev<T>()`：从 direct parent move 出对象后调用成员函数，会按 exclusive access 处理，因此不能和其他 `prev_access` 绑定混用，也不能进入 `retry(...)`

## Roadmap

1. `Phase 1`：补齐 `step_result`、`exec_context`、`value/from_ctx`、`bind`
2. `Phase 2`：补齐 `task_result<T>`、`PrevStore`、`borrow_prev<T>`、链式结果传递
3. `Phase 3`：补齐 `schedule/chain/then/step` 的最小执行闭环
4. `Phase 4`：补齐 `traits` 与自动参数/返回值推导
