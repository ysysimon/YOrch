# Executor

<a href="executor.md">English</a> | <a href="../cn/executor.md">中文</a>

The executor runs a compiled plan. The usual flow is:

```cpp
auto plan = yorch::compile_plan(tree);
const auto result = yorch::run_plan(plan);
```

You can think of execution as four steps:

- Compile the task structure into a plan.
- Prepare the context needed for one execution.
- Choose optional policies.
- Call `run_plan(...)` and check the execution result.

## Plan

A task tree is more like a build-time structure. It records which tasks exist and
how they are related by level. Before execution, call `compile_plan(...)`:

```cpp
auto plan = yorch::compile_plan(tree);
```

The plan is the object read by the executor. It stores task nodes, parent-child
relationships, argument sources, and slot information, and it also carries
pre-execution constraint checks. For example, an empty task tree, invalid
prev-access, or invalid forward-prev / fanout policy combinations are rejected
before execution starts.

## Context

`context` stores shared data for one execution. A task can declare an argument
from context with `from_ctx<T>()`; at execution time, pass the context to
`run_plan(...)`.

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

If multiple tasks need shared configuration, loggers, cache handles, or runtime
state, it is recommended to give the context type a business name with `using`.
That keeps construction, passing, and function signatures clearer:

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

`context<Ts...>` stores objects by type. Each type can appear only once in the
same context, so task parameters can unambiguously find their source by type. The
executor only borrows the context during the current `run_plan(...)` call; it
does not create or extend the lifetime of the context.

If tasks do not need context, call:

```cpp
const auto result = yorch::run_plan(plan);
```

## Policy

`run_plan(...)` can use default settings or select optional strategies through
template parameters. There are currently two main policy categories:

- slot layout policy: decides how runtime slots are arranged;
- execution policy: decides how the executor traverses and schedules the plan.

Policies are written as template parameters of `run_plan(...)` in this fixed
order:

```cpp
yorch::run_plan<SlotLayoutPolicy, ExecutionPolicy>(plan);
```

If execution also needs context, policies are still written as template
parameters, and the context remains a normal function argument:

```cpp
yorch::run_plan<SlotLayoutPolicy, ExecutionPolicy>(plan, ctx);
```

The default call is usually enough. It is equivalent to using
`slot_layout_one_to_one_policy` and `exec_serial_dfs_recursive_policy`:

```cpp
const auto result = yorch::run_plan(plan);
```

### slot_layout_one_to_one_policy

`slot_layout_one_to_one_policy` is the default slot layout policy. It prepares an
independent physical slot for every task node that has an output payload.

This policy is the most direct. It is useful when first using the library,
debugging execution flow, or wanting the relationship between slots and task
nodes to be easy to understand. Since it is the default, it usually does not need
to be written explicitly:

```cpp
const auto result = yorch::run_plan(plan);
```

To write the default slot layout explicitly, place it in the first template
parameter position of `run_plan`:

```cpp
const auto result = yorch::run_plan<
    yorch::slot_layout_one_to_one_policy
>(plan);
```

### slot_layout_serial_dfs_compact_policy

`slot_layout_serial_dfs_compact_policy` uses the current synchronous serial DFS
execution order to reuse slot storage whose lifetimes do not overlap.

This policy fits deeper task trees, larger payloads, or cases that want to reduce
temporary runtime storage. It does not change task execution order or business
semantics; it only changes how the executor internally arranges slots.

Use it as the first template parameter of `run_plan`:

```cpp
const auto result = yorch::run_plan<
    yorch::slot_layout_serial_dfs_compact_policy
>(plan);
```

### exec_serial_dfs_recursive_policy

`exec_serial_dfs_recursive_policy` is the default execution policy. It enters
child nodes recursively through the C++ call stack. Semantically, it is
synchronous serial DFS: execute the root, enter children after the parent
succeeds, complete a subtree, and then return to later siblings.

Since it is the default, it usually does not need to be written explicitly:

```cpp
const auto result = yorch::run_plan(plan);
```

If you need to write the execution policy explicitly, it must be in the second
template parameter position. Because the first position belongs to the slot
layout policy, both policies must be written:

```cpp
const auto result = yorch::run_plan<
    yorch::slot_layout_one_to_one_policy,
    yorch::exec_serial_dfs_recursive_policy
>(plan);
```

### exec_serial_dfs_explicit_heap_stack_policy

`exec_serial_dfs_explicit_heap_stack_policy` keeps the same synchronous serial DFS
semantics, but it does not enter child nodes through the C++ call stack. It uses
an explicit heap stack to store traversal state.

This policy is useful when a task tree may be very deep, or when you want to
avoid deep recursive call stacks. It is still synchronous, serial, and DFS; it
does not run tasks concurrently.

Write the slot layout policy in the first template parameter position and the
execution policy in the second:

```cpp
const auto result = yorch::run_plan<
    yorch::slot_layout_one_to_one_policy,
    yorch::exec_serial_dfs_explicit_heap_stack_policy
>(plan);
```

It can also be combined with compact slot layout:

```cpp
const auto result = yorch::run_plan<
    yorch::slot_layout_serial_dfs_compact_policy,
    yorch::exec_serial_dfs_explicit_heap_stack_policy
>(plan);
```

To keep call sites shorter, use `using` aliases for common combinations:

```cpp
using CompactSlots = yorch::slot_layout_serial_dfs_compact_policy;
using HeapStackDfs = yorch::exec_serial_dfs_explicit_heap_stack_policy;

const auto result = yorch::run_plan<CompactSlots, HeapStackDfs>(plan);
```

These policies do not change the business semantics of the tasks. They only
affect how the executor stores intermediate results, how it traverses the task
structure, and whether it uses the recursive call stack or an explicit heap
stack. They all currently belong to the synchronous serial DFS execution model
for task trees.

## Synchronous Serial DFS

The currently supported execution mode for task trees is synchronous serial DFS.

Synchronous means `run_plan(...)` completes on the current thread and returns a
result. Serial means only one task runs at a time. DFS means the executor uses
depth-first order: start from the root, enter the first child and its descendants
after the parent succeeds, and then return to later siblings.

In this mode, the plan structure, node relationships, and most execution paths
are already known at compile time. The executor tries to use that static
information for optimization and expansion, pushing scheduling logic toward
compile time and avoiding extra dynamic scheduling, runtime lookups, or type
erasure just to express task orchestration.

For this structure:

```text
A
  B
    D
  C
```

The current execution order is:

```text
A
B
D
C
```

If a task returns a failure state, the executor uses the result semantics to stop
the current branch or terminate the whole plan. This keeps failure propagation
and resource lifetimes clear, and matches the task-tree model: a subtree only has
the precondition to run after its parent chain has succeeded.

## Future Execution Modes

More execution modes will be added later:

- async serial: still advances one task at a time, but allows execution to
  suspend and resume asynchronously;
- async parallel: advances multiple independent tasks concurrently when
  dependencies allow it.

The goal is to reuse the same task registration and task structure while
replacing only the execution-stage scheduling model.

## Execution Result

`run_plan(...)` returns `yorch::step_result`. The common pattern is to check
whether the whole plan completed successfully with `ok()`:

```cpp
const auto result = yorch::run_plan(plan);

if (!result.ok()) {
    // handle failure
}
```
