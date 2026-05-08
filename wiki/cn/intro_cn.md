# YOrch

[English](../../README.md) | [中文](intro_cn.md)

## 简介
YOrch 是一个面向 **任务** 的编排与执行库，充分利用了编译时展开优化，目前是纯头文件的。

## Hello World

下面是一个极简例子：

```cpp
#include "yorch/yorch.hpp"

#include <iostream>

int main() {
    auto tree = yorch::task_tree
        .root([]() noexcept {
            std::cout << "TaskA\n";
        })()
        .node<1>([]() noexcept {
            std::cout << "TaskB\n";
        })()
        .node<1>([]() noexcept {
            std::cout << "TaskC\n";
        })()
        .node<2>([]() noexcept {
            std::cout << "TaskD\n";
        })();

    auto plan = yorch::compile_plan(tree);
    const auto result = yorch::run_plan(plan);

    return result.ok() ? 0 : 1;
}
```

这里 `.root(...)()` 注册根任务 `TaskA`，`.node<1>(...)()` 注册第一层子任务，
`.node<2>(...)()` 注册更深一层的任务。当前 `task_tree` 表达的是树形依赖结构。

输出为：

```text
TaskA
TaskB
TaskC
TaskD
```

## 什么是任务
YOrch 接受 C++ `callable` 作为 **任务**

### 当前支持的 callable

- 普通函数
  - `void run(int)`
- 普通函数指针
  - `void (*fp)(int) = run`
- 普通函数引用
  - `void (&fr)(int) = run`
- 非泛型 lambda
  - `[](int x) { return x + 1; }`
- 只有一个明确 `operator()` 的函数对象
  - `struct F { void operator()(int) const; }`
- `const` 调用的函数对象
  - `[](int x) { return x; }`
- 拥有单一具体调用签名的函数包装对象
  - `std::function<int(int)>`
- 非静态成员函数指针
  - `&Worker::run`
- `const` 非静态成员函数指针
  - `&Worker::read`
- `noexcept` 函数与 `noexcept` 成员函数
  - `void run() noexcept`
  - `void Worker::run() noexcept`
- 静态成员函数
  - `&Worker::make`

### 将会支持的特殊 callable

- CUDA kernel
  - `my_kernel<<<grid, block>>>(args...)`

### 当前不支持的 callable

- 重载函数名
  - 同时存在 `void run(int)` 和 `void run(double)` 时直接传 `run`
- 泛型 lambda
  - `[](auto x) { return x; }`
- 带模板 `operator()` 的函数对象
  - `struct F { template <class T> void operator()(T); }`
- 带多个重载 `operator()` 的函数对象
  - `struct F { void operator()(int); void operator()(double); }`
- 需要运行期决定签名的动态 callable
  - 只在运行期才知道目标签名的类型擦除封装
- C 风格变参函数
  - `void log(const char* fmt, ...)`
- ref-qualified 成员函数指针
  - `void Worker::run() &`
- `volatile` 或 `const volatile` 成员函数指针
  - `void Worker::run() volatile`


将 `callable` 注册为 **任务** 的具体 `API` 可以在 [这里](register_task.md) 找到 

## 任务结构
YOrch 支持不同的任务结构，根据任务的直接依赖数量可以分类为
- **树 Tree**
- **有向无环图 DAG，Directed Acyclic Graph** (WIP)

### 任务树 Task Tree
任务树是一种层级化的依赖模型。它不是任意连线的图，而是从一个根任务开始，
逐层向下展开的执行结构。

它有几个核心特点：

- 除了根任务之外，每个任务只有一个直接父任务。
- 父任务先于它的子任务执行。
- 一个父任务可以展开出多个子任务，用来表达后续的多个处理分支。
- 如果某个任务失败，它下面的整棵子树都不再继续执行。

可以把它理解成下面这样的结构：

```text
load_config
  init_runtime
    load_assets
      build_scene
      build_index
    start_workers
      worker_a
      worker_b
```

这段伪代码表示：

- `init_runtime` 的直接父任务是 `load_config`
- `load_assets` 和 `start_workers` 的直接父任务都是 `init_runtime`
- `build_scene` 和 `build_index` 只依赖 `load_assets`
- `worker_a` 和 `worker_b` 只依赖 `start_workers`

如果 `load_assets` 失败，那么 `build_scene` 和 `build_index` 所在的子树会停止；
但 `start_workers` 这一侧是否继续，取决于它自己的父链是否成功。

如果 `init_runtime` 失败，那么它下面的所有后续任务都失去执行前提，
包括 `load_assets`、`start_workers` 以及它们各自的子任务。

这种模型适合表达阶段清晰、上下游关系明确的编排流程。它牺牲了任意 DAG 的自由连线能力，
换来的是更简单的依赖来源、更明确的失败传播路径，以及更容易理解的执行计划。

构建 **任务树** 的具体 `API` 可以在 [这里](build_tree.md) 找到 

### 任务图 Task Graph
仍然在开发中。

## 执行器
任务结构描述的是“要做什么”和“任务之间如何依赖”，执行器负责把这个结构真正跑起来。
在 YOrch 中，任务树需要先通过 `compile_plan(...)` 编译成 plan，再交给
`run_plan(...)` 执行。plan 可以理解成执行阶段使用的静态说明书：里面记录了任务节点、
父子关系、参数来源、slot 布局和执行前需要确认的约束。

执行阶段还会涉及几个概念：`context` 用来保存一次执行共享的数据，并在调用
`run_plan(...)` 时传入；policy 用来选择可选的执行细节，例如 slot layout policy
和 execution policy；执行结果通过 `step_result` 返回，用来判断整个 plan 是否成功完成。

当前已经支持的是 **task tree 的同步串行 DFS 执行**。执行器会从根任务开始，父任务成功后
再按深度优先顺序进入子任务；如果任务返回失败状态，执行器会按结果语义停止当前分支或终止整个执行。
在这个模式下，YOrch 会尽量利用编译时已知的 plan 信息做优化和展开，避免为任务调度引入额外的运行时开销。
后续会继续补充 **异步串行** 和 **异步并行** 的执行方式，让同一套任务结构可以在更多运行模型下复用。

执行器相关概念和 `API` 可以在 [这里](executor.md) 找到

## 性能测试
YOrch 使用 `Google Benchmark` 做运行期性能测试。由于当前库主要支持的是静态任务树构建，
任务结构通常在执行前就已经通过 `compile_plan(...)` 固化成 plan，因此当前 benchmark
重点测试的是已经编译好的 plan 在 `run_plan(...)` 阶段的执行成本，而不是任务树构建或
`compile_plan(...)` 本身的成本。

可以通过 benchmark preset 构建：

```bash
cmake --preset benchmark
cmake --build --preset benchmark
```

运行 `run_plan` benchmark：

```bash
./build/benchmark/benchmarks/bench_run_plan --benchmark_min_time=0.05s
```

当前 `bench_run_plan` 会输出两类结果：

- `RunPlan/Runtime/...`：现实执行基线。benchmark 会用 `DoNotOptimize(plan)`、
  `DoNotOptimize(sink)` 和 `ClobberMemory()` 减少过度优化，尽量测到实际
  `run_plan(...)` 调度、slot、prev-access 和 policy 路径的成本。
- `RunPlan/Optimized/...`：优化上限。benchmark 不对 plan 加额外 barrier，
  允许编译器尽量 inline、fold 和消除抽象，用来观察 YOrch 的静态执行路径在理想条件下
  能被优化到什么程度。

benchmark 名称的结构是：

```text
RunPlan/<Mode>/<Topology>/<Payload>/<NodeCount>/<SlotLayout>/<ExecPolicy>
```

例如：

```text
RunPlan/Runtime/Chain32/Int/32/Compact/HeapStack
```

表示用 `Runtime` 模式测试 32 个节点的链式 plan，payload 是 `int`，
slot layout 使用 `Compact`，execution policy 使用显式 heap stack。

当前测试覆盖的 topology 包括：

- `Chain8` 和 `Chain32`：链式任务树，用来观察深度增长后的执行成本。
- `Wide8`：一个 root 下挂多个 sibling，用来观察 fanout/sibling traversal 成本。
- `Balanced15`：15 个节点的平衡树，用来观察更接近常规任务树的混合结构。
- `FanoutConsumeCopies3`：使用 `fanout_consume_with_copies_policy`，
  1 个 `consume_prev` child 和 2 个 `copy_prev` child，用来观察 staging/copy 成本。

当前测试覆盖的 policy 组合包括：

- `OneToOne/Recursive`
- `Compact/Recursive`
- `OneToOne/HeapStack`
- `Compact/HeapStack`

其中 `Recursive` 指 `exec_serial_dfs_recursive_policy`，使用编译期递归展开的同步 DFS；
`HeapStack` 指 `exec_serial_dfs_explicit_heap_stack_policy`，使用显式 heap stack 保存遍历状态。
`OneToOne` 是一节点一物理 slot 的默认布局；`Compact` 会按同步串行 DFS 的生命周期复用 slot。

文中的数字来自一次本地测试，只能作为量级参考。不同 `CPU`、编译器、编译选项、
系统负载、温度状态和操作系统调度都会影响纳秒级 benchmark 的绝对数值。
严肃比较时应该记录硬件、操作系统、编译器版本、`CMake` preset、构建类型和运行参数，
并优先比较同一环境下不同 policy 或 topology 之间的相对差异。

这次示例结果的环境是：

- **CPU**: Apple M4
- **OS**: macOS
- **Compiler**: `AppleClang`
- **Build**: `benchmark` preset, `Release`
- **Command**: `./build/benchmark/benchmarks/bench_run_plan --benchmark_min_time=0.05s`

### 当前结果解读

一次典型结果中，`Runtime` 模式下的 `Recursive` 路径大致是：

```text
RunPlan/Runtime/Chain8/Int/8/OneToOne/Recursive       11.2 ns
RunPlan/Runtime/Chain32/Int/32/OneToOne/Recursive     50.6 ns
RunPlan/Runtime/Balanced15/Int/15/OneToOne/Recursive  20.8 ns
```

折算下来，`Recursive` 路径大约在 `1.4 ns/node` 到 `1.6 ns/node` 这个量级。
这说明当前同步串行 DFS 的静态执行路径非常轻，和直接手写必要的静态 `if` 判断、
内联函数调用链属于接近的开销量级。它不是动态调度器式的成本，更像是 template 展开后的
结构化函数调用。

同一批测试中，`HeapStack` 路径大致是：

```text
RunPlan/Runtime/Chain8/Int/8/OneToOne/HeapStack       40.7 ns
RunPlan/Runtime/Chain32/Int/32/OneToOne/HeapStack      186 ns
RunPlan/Runtime/Balanced15/Int/15/OneToOne/HeapStack  76.8 ns
```

折算下来，`HeapStack` 路径大约在 `5 ns/node` 到 `6 ns/node` 这个量级。
它仍然很轻，但相比 `Recursive` 明显更贵，因为它需要维护 runtime frame、heap stack、
node index 和 dispatch table。它的主要价值是避免深递归调用栈风险，而不是追求最低延迟。

`Optimized` 模式下，一些 `Recursive` case 可以接近：

```text
RunPlan/Optimized/Chain8/Int/8/OneToOne/Recursive      1.59 ns
RunPlan/Optimized/Wide8/Int/9/OneToOne/Recursive       1.59 ns
RunPlan/Optimized/Balanced15/Int/15/OneToOne/Recursive 1.59 ns
```

这不表示真实业务执行一定只需要 `1.59 ns`，而是说明在非常理想、可静态分析的条件下，
编译器可以看穿大量 YOrch 抽象，把 `Recursive` 的静态执行路径 inline 或 fold 掉。
这正是 `Optimized` 结果的意义：它展示的是 zero-cost abstraction 的优化上限。

值得注意的是，`Optimized/Chain32/Recursive` 在当前结果中仍然接近 `Runtime`：

```text
RunPlan/Optimized/Chain32/Int/32/OneToOne/Recursive  50.6 ns
```

这说明编译器并不会对所有形状都做同样程度的消除。更长的依赖链可能触发 inline 或优化阈值，
于是保留了更多实际执行路径。这个差异可以帮助我们判断哪些任务形状更容易被优化器完全吃掉。

### 对实时热路径的结论

从当前结果看，YOrch 框架本身的底噪很低。对于普通实时应用、游戏帧循环、音视频外围逻辑、
高频业务调度等场景，如果任务本身会做内存访问、计算、日志、锁、IO 或其他业务逻辑，
`run_plan(...)` 的调度成本通常不会是主要瓶颈。

如果热路径中的 task 极轻，例如每个任务只做几个整数操作，那么框架成本就会变得可见。
这种情况下推荐优先使用：

```cpp
yorch::run_plan<
    yorch::slot_layout_one_to_one_policy,
    yorch::exec_serial_dfs_recursive_policy
>(plan);
```

也就是默认 slot layout 加 `Recursive` execution policy。只有当 plan 可能非常深，
并且调用栈深度风险比最低延迟更重要时，再考虑 `HeapStack`。

对于硬实时、audio callback、交易 tick 内核、每秒千万级 tight loop 这类极端热路径，
仍然建议用真实业务 payload、真实任务树形状和目标机器单独测量。当前 microbenchmark
可以说明 YOrch 的框架开销处在很低的量级，但不能替代真实 workload benchmark。

运行 benchmark 时，macOS 上可能看到类似输出：

```text
Unable to determine clock rate from sysctl: hw.cpufrequency
This does not affect benchmark measurements, only the metadata output.
***WARNING*** Failed to set thread affinity. Estimated CPU frequency may be incorrect.
```

这表示 `Google Benchmark` 没能读取准确 CPU 频率或固定线程亲和性。它不影响 `Time` 和
`CPU` 的计时本身，但 `Run on (10 X 24 MHz CPU s)` 这类 metadata 不应作为真实 CPU 频率解读。
严肃对比时应该多跑几轮，关注趋势、中位数和不同 policy 之间的稳定差异。
