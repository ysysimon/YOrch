# Design Positioning

本文总结当前 `YOrch` 的设计定位，重点说明它鼓励什么风格、没有试图解决什么问题，以及库与用户分别负责哪些状态与生命周期。

## One-line Summary

当前 `YOrch` 更适合被理解为：

**a tree-structured dataflow orchestration library with functional-style task interfaces**

也就是说，它的核心是 `dataflow orchestration`，而不是纯 `functional programming`；但在任务接口层，它明显鼓励 `functional-style` 的写法。

## What The Library Optimizes For

当前方向主要在优化下面几件事：

- 用显式依赖表达数据流，而不是依赖隐式共享状态
- 用静态结构表达编排关系，而不是依赖宽松的 runtime blackboard
- 用返回值表达下游可消费的 payload，而不是把中间结果散落在外部对象里
- 让库统一接管中间结果的存储与析构时机

在当前方向下，这些目标会进一步体现为：

- `task_tree` 记录静态树结构
- `from_prev<T>()` 只表示 direct parent payload
- `compiled_plan` 负责恢复 parent / child 关系和 per-node output metadata
- 后续 `slot` 与 `serial executor` 负责中间值生命周期闭环

## Functional-style, But Not Purely Functional

这套设计在接口层鼓励用户把 task 写成“小而清晰的输入输出单元”：

- 输入通过 `from_ctx(...)`、`from_prev(...)`、`value(...)` 显式声明
- 输出通过返回值显式表达
- 中间值传播由编排层接手

这很接近 `functional-style` 的任务边界：

```text
input -> transform -> output
```

但它并不是纯 `functional programming`。当前设计并没有要求：

- `immutable data`
- `pure function`
- `referential transparency`
- `no side effect`

相反，用户依然可以：

- 从 `context` 中借用长期共享对象
- 通过 `T&` 修改状态
- 在 task 内执行 `side effect`
- 使用异常适配、`retry`、`abort_chain` 等控制流能力

因此更准确的说法是：

**YOrch uses an imperative execution model with functional-style task boundaries.**

## State Ownership Boundary

当前设计并不是“把所有状态都交给库”，而是明确区分两类状态。

### State Managed By The Library

这部分更偏向 orchestration state：

- 节点之间的数据可见性
- direct parent payload 的传播方式
- 中间结果写入哪个 `slot`
- 中间结果在何时构造、何时析构
- 树形遍历顺序与子树生命周期边界

这些事情由 `plan`、`slot`、`executor` 统一定义，会让编排层更容易做静态检查与生命周期管理。

### State Managed By The User

这部分更偏向业务语义：

- `context` 中放什么共享对象
- 哪些对象允许被修改
- 外部资源的 ownership
- `side effect` 的幂等性与一致性
- 将来并行执行时的 thread-safety

所以更准确的描述是：

**the library manages orchestration state and intermediate lifetimes, while the user still owns business state semantics.**

## What Style This Design Encourages

这套 API 会自然推动用户采用下面几种习惯：

- 把依赖写在参数列表里，而不是藏在 task 内部
- 把中间结果写在返回值里，而不是偷偷写入外部共享容器
- 让 task 尽量保持 shallow-state，减少内部复杂可变状态
- 把长期共享资源放进 `context`
- 把短生命周期的数据流交给 `prev` / `slot`

这并不禁止有状态逻辑，但会显著降低“隐藏状态驱动编排”的吸引力。

## Practical Guidance For Users

比较推荐的使用姿势是：

- 把 task 当成显式输入输出的步骤单元
- 把长期共享对象放进 `context`
- 把短期派生值交给返回值和 `from_prev`
- 尽量避免让 task 自己持有复杂、长生命周期、强可变的内部状态
- 把“数据如何流动”交给编排层，而不是靠隐式约定

