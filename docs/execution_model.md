# Execution Model

本文说明当前 `YOrch` 中 `context`、`spec`、`resolve`、`bind`、`executor` 之间的关系，以及一次任务执行的大致流程。

## Core Roles

### `context<Ts...>`

`context<Ts...>` 是按类型访问的静态容器，用来存放执行时需要共享的数据。当前实现使用 `std::tuple` 作为底层存储，并要求同一个类型在 schema 中只能出现一次。

它的职责很单纯：

- 持有实际数据
- 提供 `contains<T>()`
- 提供 `get<T>()`

`context` 本身不知道任务、参数绑定或调度流程。

### `exec_context<Ctx, Prev>`

`exec_context` 是执行期借用视图，不拥有数据。

它把两类信息交给执行逻辑：

- 当前 `context`
- 可选的 direct parent 输出视图

如果当前执行路径没有 direct parent 输出，则使用 `no_prev` 作为哨兵类型。

### `spec`

`spec` 用来描述“函数参数应该从哪里来”，当前主要有三类：

- `from_ctx<T>()`：从当前 `context` 里按类型取值
- `from_prev<T>()`：从 direct parent 输出里按类型取值
- `value(x)`：把一个值直接存进 `spec`

`spec` 本身只负责描述来源，不负责真正取值。

### `resolve_as<Arg>(spec, ec)`

`resolve_as(...)` 负责把 `spec` 解析成 callable 真正需要的参数。

它会根据 `spec` 的种类选择不同的解析路径：

- `from_ctx<T>`：从 `ec.ctx` 里取出 `T`
- `from_prev<T>`：从 `ec.prev_view()` 里取出 `T`
- `value_t<T>`：从 `spec` 自己持有的值里取出对象

取到源对象之后，还会继续根据目标参数类型 `Arg` 做一次适配：

- 如果参数是 `T&`，则传引用
- 如果参数是 `const T&`，则传常量引用
- 如果参数是值类型 `T`，则拷贝或转换出一个新值

这一步由 `bind_from_lvalue(...)` 完成。

### `bind(f, specs...)`

`bind(...)` 把 callable 和一组 `spec` 打包成一个 `bound_task`。

在这个阶段会做几件事：

- 通过 `function_traits` 读取 callable 的参数列表
- 检查 `spec` 数量是否与参数数量一致
- 把 callable 和 `spec` 都按值存起来

`bind(...)` 只负责组装，不执行任务，也不提前取参数。

### `run_task(task, ec)`

`run_task(...)` 是当前最薄的一层执行入口。

它要求 `task.invoke_raw(ec)` 这一协议存在，并且当前主路径要求该调用满足 `noexcept`。参数解析仍然由 `bind` 完成，但返回值规整现在放在 `executor` 这一层做。

## Execution Flow

下面用一个简化示例说明执行过程：

```cpp
auto task = yorch::bind(
    f,
    yorch::from_ctx<int>(),
    yorch::value(4),
    yorch::from_prev<std::string>());
```

一次执行大致经过以下步骤：

1. 调用 `bind(...)`，生成一个 `bound_task<F, Specs...>`
2. 创建 `exec_context`，把当前 `context` 和可选的 direct parent 输出借给执行过程
3. 调用 `run_task(task, ec)`
4. `bound_task::invoke_raw(ec)` 按参数顺序展开每个 `spec`
5. 对第 `I` 个参数，先从 callable 签名中取出真实参数类型，再调用 `resolve_as<Arg>(spec, ec)`
6. `resolve_as(...)` 从 `context`、direct parent，或者 `value_t` 中取出源对象
7. `bind_from_lvalue(...)` 把源对象适配成 callable 需要的参数形式
8. 调用 callable
9. `run_task(...)` 把原始返回值规整成 `step_result`

当前支持的返回值规整规则如下：

- `step_result`：原样返回
- `task_result<T>` / `task_result<void>`：返回其中的 `step`
- `void`：视为 `success`
- 普通返回值 `T`，包括 `bool`：当前统一视为 `success`

## Compile-time vs Runtime

当前这套设计更偏向“编译期决定结构，运行期做薄执行”。

### 编译期主要完成

- 提取 callable 参数类型
- 校验 `spec` 数量
- 校验 `context` 中是否存在目标类型
- 选择正确的 `resolve_as(...)` overload
- 决定参数应按引用还是按值传递

### 运行期主要完成

- 从 `context` 或 direct parent 中取对象
- 从 `value_t` 中读取已保存的值
- 调用 callable
- 在 `executor` 中把原始返回值规整成 `step_result`

## What Remains After Build

当前实现的核心逻辑主要位于头文件中，因此很多能力是模板实例化后的结果，而不是集中放在一个固定的库实现里。

实际构建后可以把产物分成两类理解：

- 运行时对象：例如 `context<Ts...>`、`exec_context<...>`、`bound_task<...>`、`value_t<T>`、`step_result`
- 编译期机制：例如 `function_traits`、`index_sequence`、`static_assert`、overload 选择

前者会作为真实对象或实例化代码出现在使用方目标文件里，后者主要在编译期间起作用。

当前仓库中的 `yorch` 编译库本身只提供一个很薄的 anchor 源文件；`bind`、`resolve`、`executor` 的核心行为仍主要来自头文件实例化。
