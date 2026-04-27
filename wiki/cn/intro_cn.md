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