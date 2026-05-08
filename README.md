# YOrch

<p align="center">
  <img src="./assets/banner.png" alt="Banner" width="100%">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-76B900.svg" alt="License"></a>
  <a href="README.md"><img src="https://img.shields.io/badge/Language-English-1f6feb.svg" alt="English"></a>
  <a href="wiki/cn/intro_cn.md"><img src="https://img.shields.io/badge/Language-%E4%B8%AD%E6%96%87-1f6feb.svg" alt="中文"></a>
</p>

## Introduction

YOrch is a **task-oriented** orchestration and execution library. It takes
advantage of compile-time expansion and optimization opportunities, and is
currently header-only.

## Hello World

Here is a minimal example:

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

Here `.root(...)()` registers the root task `TaskA`, `.node<1>(...)()`
registers first-level child tasks, and `.node<2>(...)()` registers a deeper
task. The current `task_tree` represents a tree-shaped dependency structure.

Output:

```text
TaskA
TaskB
TaskC
TaskD
```

## What is a task?

YOrch accepts C++ `callable` objects as **tasks**.

### Currently supported callables

- Free functions
  - `void run(int)`
- Free function pointers
  - `void (*fp)(int) = run`
- Free function references
  - `void (&fr)(int) = run`
- Non-generic lambdas
  - `[](int x) { return x + 1; }`
- Function objects with one unambiguous `operator()`
  - `struct F { void operator()(int) const; }`
- Function objects callable through `const`
  - `[](int x) { return x; }`
- Function wrapper objects with one concrete call signature
  - `std::function<int(int)>`
- Non-static member function pointers
  - `&Worker::run`
- `const` non-static member function pointers
  - `&Worker::read`
- `noexcept` functions and `noexcept` member functions
  - `void run() noexcept`
  - `void Worker::run() noexcept`
- Static member functions
  - `&Worker::make`

### Special callables planned for support

- CUDA kernels
  - `my_kernel<<<grid, block>>>(args...)`

### Currently unsupported callables

- Overloaded function names
  - passing `run` directly when both `void run(int)` and `void run(double)` exist
- Generic lambdas
  - `[](auto x) { return x; }`
- Function objects with templated `operator()`
  - `struct F { template <class T> void operator()(T); }`
- Function objects with multiple overloaded `operator()`
  - `struct F { void operator()(int); void operator()(double); }`
- Dynamic callables whose signature is only known at runtime
  - type-erased wrappers whose target signature is runtime-dependent
- C-style variadic functions
  - `void log(const char* fmt, ...)`
- Ref-qualified member function pointers
  - `void Worker::run() &`
- `volatile` or `const volatile` member function pointers
  - `void Worker::run() volatile`

The task registration APIs are described in
<a href="wiki/en/register_task.md">Register Tasks</a>.

## Task Structure

YOrch supports different task structures. According to the number of direct
dependencies a task may have, they can be grouped as:

- **Tree**
- **DAG, Directed Acyclic Graph** (WIP)

### Task Tree

A task tree is a hierarchical dependency model. It is not an arbitrary graph;
instead, it starts from one root task and expands downward level by level.

Its core properties are:

- Every task except the root has exactly one direct parent.
- A parent task runs before its child tasks.
- One parent task may expand into multiple child tasks to express later branches.
- If a task fails, the whole subtree below that task stops executing.

You can think of it as this structure:

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

This pseudocode means:

- `init_runtime` has `load_config` as its direct parent.
- `load_assets` and `start_workers` both have `init_runtime` as their direct
  parent.
- `build_scene` and `build_index` only depend on `load_assets`.
- `worker_a` and `worker_b` only depend on `start_workers`.

If `load_assets` fails, the subtree containing `build_scene` and `build_index`
stops. Whether the `start_workers` side continues depends on whether its own
parent chain succeeds.

If `init_runtime` fails, all following tasks below it lose their execution
precondition, including `load_assets`, `start_workers`, and their respective
children.

This model fits orchestration flows with clear stages and explicit upstream /
downstream relationships. It gives up arbitrary DAG wiring in exchange for
simpler dependency sources, clearer failure propagation paths, and easier-to-read
execution plans.

The task tree construction APIs are described in
<a href="wiki/en/build_tree.md">Build a Task Tree</a>.

### Task Graph

Still in development.

## Executor

The task structure describes "what to do" and "how tasks depend on each other".
The executor is responsible for actually running that structure. In YOrch, a
task tree is first compiled into a plan with `compile_plan(...)`, and then passed
to `run_plan(...)`. A plan can be understood as the static instruction sheet used
during execution: it records task nodes, parent-child relationships, argument
sources, slot layout, and constraints that must be checked before execution.

Execution also involves several concepts: `context` stores shared data for one
execution and can be passed to `run_plan(...)`; policies select optional
execution details such as slot layout policy and execution policy; and execution
results are returned through `step_result`, which tells whether the whole plan
completed successfully.

The currently supported mode is **synchronous serial DFS execution for task
trees**. The executor starts from the root task, enters child tasks only after
their parent succeeds, and follows depth-first order. If a task returns a failure
state, the executor stops the current branch or terminates the whole execution
according to the result semantics. In this mode, YOrch tries to use compile-time
plan information for optimization and expansion, avoiding extra runtime overhead
for task scheduling.

Future execution modes will include **async serial** and **async parallel**
execution, so the same task structure can be reused across more runtime models.

Executor concepts and APIs are described in
<a href="wiki/en/executor.md">Executor</a>.

## Build and Test Commands

Use CMake presets:

```bash
cmake --list-presets
cmake --preset dev
cmake --build --preset dev
ctest --preset dev
```

## Benchmarks

YOrch uses `Google Benchmark` for runtime performance tests. Because the current
library mainly supports static task tree construction, task structures are
usually fixed into a plan through `compile_plan(...)` before execution. The
current benchmarks therefore focus on the execution cost of an already compiled
plan in `run_plan(...)`, rather than the cost of building a task tree or calling
`compile_plan(...)`.

Build the benchmark preset:

```bash
cmake --preset benchmark
cmake --build --preset benchmark
```

Run the `run_plan` benchmark:

```bash
./build/benchmark/benchmarks/bench_run_plan --benchmark_min_time=0.05s
```

The current `bench_run_plan` outputs two categories:

- `RunPlan/Runtime/...`: realistic execution baseline. The benchmark uses
  `DoNotOptimize(plan)`, `DoNotOptimize(sink)`, and `ClobberMemory()` to reduce
  over-optimization and better measure the actual cost of `run_plan(...)`
  dispatch, slots, prev-access, and policy paths.
- `RunPlan/Optimized/...`: optimization ceiling. The benchmark does not add
  extra barriers around the plan, allowing the compiler to inline, fold, and
  eliminate abstractions as much as possible. This shows how far YOrch's static
  execution path can be optimized under ideal conditions.

Benchmark names use this structure:

```text
RunPlan/<Mode>/<Topology>/<Payload>/<NodeCount>/<SlotLayout>/<ExecPolicy>
```

For example:

```text
RunPlan/Runtime/Chain32/Int/32/Compact/HeapStack
```

This means the benchmark uses `Runtime` mode to test a 32-node chain plan, the
payload is `int`, the slot layout is `Compact`, and the execution policy uses an
explicit heap stack.

Current topology coverage includes:

- `Chain8` and `Chain32`: chain-shaped task trees, used to observe execution cost
  as depth grows.
- `Wide8`: multiple siblings under one root, used to observe fanout and sibling
  traversal cost.
- `Balanced15`: a 15-node balanced tree, used to observe a mixed structure closer
  to typical task trees.
- `FanoutConsumeCopies3`: uses `fanout_consume_with_copies_policy`, with one
  `consume_prev` child and two `copy_prev` children, used to observe staging and
  copy cost.

Current policy combinations include:

- `OneToOne/Recursive`
- `Compact/Recursive`
- `OneToOne/HeapStack`
- `Compact/HeapStack`

`Recursive` means `exec_serial_dfs_recursive_policy`, which uses compile-time
recursive expansion for synchronous DFS. `HeapStack` means
`exec_serial_dfs_explicit_heap_stack_policy`, which stores traversal state in an
explicit heap stack. `OneToOne` is the default layout with one physical slot per
node. `Compact` reuses slots according to lifetimes under synchronous serial DFS.

Benchmark numbers should only be treated as order-of-magnitude references.
Different CPUs, compilers, options, system load, thermal state, and OS scheduling
can all affect absolute nanosecond-level results. For serious comparisons,
record the hardware, operating system, compiler version, CMake preset, build
type, and runtime parameters, and prefer comparing relative differences between
policies or topologies in the same environment.

The environment for this example result is:

- **CPU**: Apple M4
- **OS**: macOS
- **Compiler**: `AppleClang`
- **Build**: `benchmark` preset, `Release`
- **Command**: `./build/benchmark/benchmarks/bench_run_plan --benchmark_min_time=0.05s`

### Current Result Interpretation

In one typical result, the `Recursive` path in `Runtime` mode is roughly:

```text
RunPlan/Runtime/Chain8/Int/8/OneToOne/Recursive       11.2 ns
RunPlan/Runtime/Chain32/Int/32/OneToOne/Recursive     50.6 ns
RunPlan/Runtime/Balanced15/Int/15/OneToOne/Recursive  20.8 ns
```

Converted per node, the `Recursive` path is roughly in the `1.4 ns/node` to
`1.6 ns/node` range. This means the current static execution path for
synchronous serial DFS is very light. Its overhead is close to directly writing
the necessary static `if` checks and inline function-call chain by hand. It is
not dynamic scheduler-style overhead; it is closer to structured function calls
after template expansion.

In the same test batch, the `HeapStack` path is roughly:

```text
RunPlan/Runtime/Chain8/Int/8/OneToOne/HeapStack       40.7 ns
RunPlan/Runtime/Chain32/Int/32/OneToOne/HeapStack      186 ns
RunPlan/Runtime/Balanced15/Int/15/OneToOne/HeapStack  76.8 ns
```

Converted per node, the `HeapStack` path is roughly in the `5 ns/node` to
`6 ns/node` range. It is still light, but clearly more expensive than
`Recursive`, because it needs to maintain runtime frames, a heap stack, node
indices, and a dispatch table. Its main value is avoiding deep recursive call
stack risk, not chasing the lowest possible latency.

In `Optimized` mode, some `Recursive` cases can approach:

```text
RunPlan/Optimized/Chain8/Int/8/OneToOne/Recursive      1.59 ns
RunPlan/Optimized/Wide8/Int/9/OneToOne/Recursive       1.59 ns
RunPlan/Optimized/Balanced15/Int/15/OneToOne/Recursive 1.59 ns
```

This does not mean real business execution necessarily takes only `1.59 ns`.
Rather, it means that under ideal, statically analyzable conditions, the compiler
can see through much of YOrch's abstraction and inline or fold away the
`Recursive` static execution path. That is the meaning of the `Optimized`
results: they show the optimization ceiling of the zero-cost abstraction.

It is worth noting that `Optimized/Chain32/Recursive` is still close to
`Runtime` in the current result:

```text
RunPlan/Optimized/Chain32/Int/32/OneToOne/Recursive  50.6 ns
```

This shows that the compiler does not eliminate every shape to the same degree.
A longer dependency chain may hit inline or optimization thresholds, preserving
more of the actual execution path. This difference helps identify which task
shapes are easier for the optimizer to fully consume.

### Conclusion for Real-Time Hot Paths

Based on the current results, the framework-level noise floor of YOrch is low.
For normal real-time applications, game frame loops, audio/video surrounding
logic, high-frequency business scheduling, and similar scenarios, if the tasks
themselves perform memory access, computation, logging, locking, IO, or other
business logic, the scheduling cost of `run_plan(...)` usually should not be the
main bottleneck.

If tasks on the hot path are extremely small, for example each task only performs
a few integer operations, the framework cost becomes visible. In that case,
prefer:

```cpp
yorch::run_plan<
    yorch::slot_layout_one_to_one_policy,
    yorch::exec_serial_dfs_recursive_policy
>(plan);
```

That is the default slot layout plus the `Recursive` execution policy. Consider
`HeapStack` only when the plan may be very deep and call stack depth risk matters
more than the lowest latency.

For extreme hot paths such as hard real-time code, audio callbacks, trading tick
kernels, or tight loops at tens of millions of iterations per second, you should
still measure with the real business payload, real task tree shape, and target
machine. The current microbenchmark shows that YOrch framework overhead is in a
very low range, but it cannot replace workload-specific benchmarks.

On macOS, benchmark runs may print output like:

```text
Unable to determine clock rate from sysctl: hw.cpufrequency
This does not affect benchmark measurements, only the metadata output.
***WARNING*** Failed to set thread affinity. Estimated CPU frequency may be incorrect.
```

This means `Google Benchmark` could not read the exact CPU frequency or pin
thread affinity. It does not affect the `Time` and `CPU` timing measurements
themselves, but metadata such as `Run on (10 X 24 MHz CPU s)` should not be
interpreted as the real CPU frequency. For serious comparisons, run multiple
rounds and focus on trends, medians, and stable differences between policies.
