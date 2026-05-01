# 执行器

执行器负责运行已经编译好的 plan。通常流程是：

```cpp
auto plan = yorch::compile_plan(tree);
const auto result = yorch::run_plan(plan);
```

可以把执行阶段理解成四件事：

- 把任务结构编译成 plan
- 准备一次执行需要的 context
- 选择可选的 policy
- 调用 `run_plan(...)` 并检查执行结果

## plan

任务树本身更像是“构建期结构”。它记录了有哪些任务，以及任务之间的层级关系。
执行之前，需要先调用 `compile_plan(...)`：

```cpp
auto plan = yorch::compile_plan(tree);
```

plan 是执行器真正读取的对象。它保存任务节点、父子关系、参数来源、slot 信息，
也会承载执行前的约束检查。比如空任务树、不合法的 prev-access、forward-prev
或 fanout policy 组合，会在进入执行之前被拒绝。

## context

`context` 用来保存一次执行共享的数据。任务注册时可以通过 `from_ctx<T>()`
声明某个参数来自 context；真正执行时，再把 context 传给 `run_plan(...)`。

```cpp
struct Config {
    int base = 0;
};

struct Logger {
    void info(const char* message) const noexcept;
};

using AppContext = yorch::context<Config, Logger>;

auto tree = yorch::task_tree
    .root([](const Config& config) noexcept {
        return config.base + 1;
    })(yorch::from_ctx<Config>());

AppContext ctx(
    Config {.base = 41},
    Logger {}
);

auto plan = yorch::compile_plan(tree);
const auto result = yorch::run_plan(plan, ctx);
```

如果多个任务都需要共享配置、日志器、缓存句柄或运行期状态，推荐先用 `using`
给 context 类型起一个业务名字。这样后续创建、传参和函数签名都会更清楚：

```cpp
struct Cache {
    int current() const noexcept;
};

using AppContext = yorch::context<Config, Logger, Cache>;

auto tree = yorch::task_tree
    .root([](const Config& config, const Logger& logger) noexcept {
        logger.info("start");
        return config.base;
    })(yorch::from_ctx<Config>(), yorch::from_ctx<Logger>())
    .node<1>([](int value, const Cache& cache) noexcept {
        return value + cache.current();
    })(yorch::borrow_prev<int>(), yorch::from_ctx<Cache>());

AppContext ctx(
    Config {.base = 10},
    Logger {},
    Cache {}
);

auto plan = yorch::compile_plan(tree);
const auto result = yorch::run_plan(plan, ctx);
```

`context<Ts...>` 按类型保存对象。同一个 context 中每种类型只能出现一次，
这样任务参数可以通过类型明确地找到来源。执行器只在本次 `run_plan(...)`
期间借用这个 context，不负责创建或延长它的生命周期。

如果任务不需要 context，可以直接调用：

```cpp
const auto result = yorch::run_plan(plan);
```

## policy

`run_plan(...)` 可以使用默认设置，也可以通过模板参数选择可选策略。
目前主要有两类 policy：

- slot layout policy：决定执行期 slot 如何布局
- execution policy：决定执行器如何遍历和调度 plan

policy 写在 `run_plan(...)` 的模板参数位置，顺序固定为：

```cpp
yorch::run_plan<SlotLayoutPolicy, ExecutionPolicy>(plan);
```

如果执行时还需要 context，policy 仍然写在 `run_plan` 后面的模板参数中，
context 仍然作为普通函数参数传入：

```cpp
yorch::run_plan<SlotLayoutPolicy, ExecutionPolicy>(plan, ctx);
```

默认写法通常已经够用。它等价于使用 `slot_layout_one_to_one_policy`
和 `exec_serial_dfs_recursive_policy`：

```cpp
const auto result = yorch::run_plan(plan);
```

### slot_layout_one_to_one_policy

`slot_layout_one_to_one_policy` 是默认的 slot layout policy。
它为每个有输出 payload 的任务节点准备独立的物理 slot。

这种策略最直接，适合刚开始使用、调试执行流程，或者希望 slot 和任务节点之间的关系更容易理解时使用。
因为它是默认值，所以通常不需要显式写出来：

```cpp
const auto result = yorch::run_plan(plan);
```

如果想显式写出默认 slot layout，可以放在 `run_plan` 的第一个模板参数位置：

```cpp
const auto result = yorch::run_plan<
    yorch::slot_layout_one_to_one_policy
>(plan);
```

### slot_layout_serial_dfs_compact_policy

`slot_layout_serial_dfs_compact_policy` 会利用当前同步串行 DFS 的执行顺序，
尽量复用不会同时存活的 slot 空间。

这种策略适合任务树较深、payload 较大，或者希望减少执行期临时存储占用的场景。
它不改变任务的执行顺序和业务语义，只改变执行器内部如何安排 slot。

使用时把它写在 `run_plan` 的第一个模板参数位置：

```cpp
const auto result = yorch::run_plan<
    yorch::slot_layout_serial_dfs_compact_policy
>(plan);
```

### exec_serial_dfs_recursive_policy

`exec_serial_dfs_recursive_policy` 是默认的 execution policy。
它使用 C++ 调用栈递归进入子节点，语义上就是同步串行 DFS：
根任务先执行，父任务成功后再进入子任务，子树完成后回到同层后续任务。

因为它是默认值，所以通常不需要显式写出来：

```cpp
const auto result = yorch::run_plan(plan);
```

如果需要显式写出 execution policy，它必须放在第二个模板参数位置。
由于第一个位置属于 slot layout policy，所以需要同时写出 slot layout policy：

```cpp
const auto result = yorch::run_plan<
    yorch::slot_layout_one_to_one_policy,
    yorch::exec_serial_dfs_recursive_policy
>(plan);
```

### exec_serial_dfs_explicit_heap_stack_policy

`exec_serial_dfs_explicit_heap_stack_policy` 保持相同的同步串行 DFS 语义，
但不通过 C++ 调用栈递归进入子节点，而是使用显式 heap stack 保存遍历过程。

这种策略适合任务树可能很深，或者希望避免深递归调用栈时使用。
它仍然是同步、串行、DFS 执行，不会让任务并发运行。

使用时把 slot layout policy 写在第一个模板参数位置，
把 execution policy 写在第二个模板参数位置：

```cpp
const auto result = yorch::run_plan<
    yorch::slot_layout_one_to_one_policy,
    yorch::exec_serial_dfs_explicit_heap_stack_policy
>(plan);
```

也可以同时选择 compact slot layout：

```cpp
const auto result = yorch::run_plan<
    yorch::slot_layout_serial_dfs_compact_policy,
    yorch::exec_serial_dfs_explicit_heap_stack_policy
>(plan);
```

为了让调用处更短，也可以用 `using` 给常用组合起名字：

```cpp
using CompactSlots = yorch::slot_layout_serial_dfs_compact_policy;
using HeapStackDfs = yorch::exec_serial_dfs_explicit_heap_stack_policy;

const auto result = yorch::run_plan<CompactSlots, HeapStackDfs>(plan);
```

这些 policy 都不改变任务本身的业务语义，只影响执行器如何保存中间结果、
如何遍历任务结构，以及使用递归调用栈还是显式 heap stack。当前它们都属于
task tree 的同步串行 DFS 执行模型。

## 同步串行 DFS

当前已经支持的是 task tree 的同步串行 DFS 执行。

同步表示 `run_plan(...)` 会在当前线程中完成执行并返回结果。串行表示同一时间只执行一个任务。
DFS 表示执行器采用深度优先顺序：从根任务开始，父任务成功后进入第一个子任务及其后代，
再回到同层的后续子任务。

在这个模式下，plan 的结构、节点关系和大部分执行路径都已经在编译期确定。
执行器会尽量利用这些静态信息做优化和展开，把调度逻辑压到编译期处理，
避免为了表达任务编排而额外引入动态调度、运行时查表或类型擦除开销。

例如下面的结构：

```text
A
  B
    D
  C
```

当前执行顺序是：

```text
A
B
D
C
```

如果某个任务返回失败状态，执行器会根据返回结果停止当前分支，或者终止整个 plan。
这让失败传播和资源生命周期都保持清晰，也符合任务树“一条父链成功后子树才有执行前提”的模型。

## 后续执行方式

后续会继续补充更多执行方式，包括：

- 异步串行：仍然保持一次只推进一个任务，但允许执行过程以异步形式挂起和恢复
- 异步并行：在依赖关系允许时，让多个互不阻塞的任务并发推进

这些模式的目标是复用同一套任务注册和任务结构，只替换执行阶段的调度方式。

## 执行结果

`run_plan(...)` 返回 `yorch::step_result`。常见用法是通过 `ok()` 判断整个 plan
是否成功完成：

```cpp
const auto result = yorch::run_plan(plan);

if (!result.ok()) {
    // handle failure
}
```
