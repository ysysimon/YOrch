# Prev Access 与 Fanout Policy

## 概览

`YOrch` 当前把 direct parent payload 的使用规则拆成两个独立维度：

- `prev access`
  子节点想如何访问 direct parent payload
- `fanout policy`
  父节点允许自己的 output 如何分发给 direct children

两者都会在 `run_plan(...)` 的 compile-time 约束中参与校验：

- `detail::plan_prev_source_valid_v<Plan>`
- `detail::plan_prev_access_valid_v<Plan>`
- `detail::plan_fanout_policy_valid_v<Plan>`

这意味着：

- access mode 是否写对，先按 task-local 规则检查
- 当前树结构和 `fanout policy` 是否允许这种 access，再按 parent-child 聚合规则检查

## Public API

### Prev Access

当前 public API 有四种 direct-parent access spec：

#### `borrow_prev<T>()`

- 只读借用 direct parent payload
- shared access
- 不消费 source
- 只能绑定到 `const T&`

#### `borrow_prev_mut<T>()`

- 独占可变借用 direct parent payload
- 不消费 source
- 需要独占访问权
- 只能绑定到 `T&`

#### `copy_prev<T>()`

- 从 direct parent payload 复制出一个新的 `T`
- shared non-exclusive access
- 不消费 source
- 当前 task 拿到的是 owned value
- 只能绑定到 `T`

#### `consume_prev<T>()`

- 消费 direct parent payload
- one-shot consume access
- 不会 eager destroy slot，slot 仍按现有执行生命周期统一析构
- 只能绑定到 `T` 或 `T&&`

两种接收方式的区别：

- `T`
  从 parent source 中 move-construct 一个新的 owned value
- `T&&`
  直接把 parent source 作为 rvalue reference 绑定给当前 task

四种 spec 都会先做 `cvref` 归一化，只保留基础类型：

```cpp
borrow_prev<const int&>()               -> borrow_prev_t<int>
borrow_prev_mut<const int&>()           -> borrow_prev_mut_t<int>
copy_prev<const int&>()                 -> copy_prev_t<int>
consume_prev<volatile std::string&&>()  -> consume_prev_t<std::string>
```

### Fanout Policy

当前 public API 有三个 `fanout policy`：

#### `fanout_auto_policy`

默认策略，保留原有隐式行为：

- 如果一个 parent 至多有一个 direct child 使用 `prev access`，该 child 可使用任意合法 `prev access`
- 如果一个 parent 有多个 direct children 使用 `prev access`，则只允许 shared access：
  - `borrow_prev`
  - `copy_prev`

适合不想显式声明 fanout 契约、只想保留默认行为的场景。

#### `fanout_shared_readonly_policy`

显式只读广播策略：

- direct children 可使用：
  - `borrow_prev`
  - `copy_prev`
- direct children 不可使用：
  - `borrow_prev_mut`
  - `consume_prev`

适合父节点 output 需要广播给多个消费者，但不允许任何 child 可变借用或消费原始 source 的场景。

#### `fanout_consume_with_copies_policy`

一个 consumer + 多个 copies 的混合策略：

- 至多一个 direct child 可使用 `consume_prev`
- 其他使用 `prev access` 的 direct children 必须全部使用 `copy_prev`
- `borrow_prev` 和 `borrow_prev_mut` 都不允许

适合“一个 child 消费原值，其他 child 读取副本”的场景。

## Builder 与 Plan Metadata

`task_tree_builder` 支持在建树时显式携带 `fanout policy`：

```cpp
auto tree = yorch::task_tree
    .root(task, yorch::fanout_shared_readonly_policy {})
    .node<1>(child_task)
    .node<1>(child_task_2, yorch::fanout_consume_with_copies_policy {});
```

具体入口为：

- `root(task, fanout_policy)`
- `node<Level>(task, fanout_policy)`

`fanout policy` 是 parent node 的 compile-time metadata，不是 `run_plan(...)` 的 runtime 参数。

`compiled_plan` 会保留每个 node 的 `fanout policy` 类型信息：

```cpp
template <std::size_t I>
using fanout_policy_type = typename node_type<I>::fanout_policy_type;
```

这让 validation 和 executor 都能按 parent node 读取策略，而不需要额外 runtime 配置。

## Prev Access 的 Task-Local 规则

第一层校验只看一个 task 自己写得是否合法，不看树结构。

### 1. 参数类型必须和 access mode 精确匹配

- `borrow_prev<T>()` -> `const T&`
- `borrow_prev_mut<T>()` -> `T&`
- `copy_prev<T>()` -> `T`
- `consume_prev<T>()` -> `T` 或 `T&&`

### 2. Shared access 可以重复出现或混用

`borrow_prev(...)` 和 `copy_prev(...)` 都属于 shared access，因此在一个 task 内：

- 多个 `borrow_prev(...)` 合法
- 多个 `copy_prev(...)` 合法
- `borrow_prev(...) + copy_prev(...)` 合法

### 3. Exclusive access 必须唯一

`borrow_prev_mut(...)` 和 `consume_prev(...)` 都属于 exclusive access。

一旦一个 task 使用了 exclusive access，它必须是这个 task 内唯一的 `prev access`。因此下面这些组合都会被拒绝：

- 多个 `borrow_prev_mut(...)`
- 多个 `consume_prev(...)`
- `borrow_prev_mut(...) + borrow_prev(...)`
- `borrow_prev_mut(...) + copy_prev(...)`
- `borrow_prev_mut(...) + consume_prev(...)`
- `consume_prev(...) + borrow_prev(...)`
- `consume_prev(...) + copy_prev(...)`

这样做是为了避免对同一个 direct parent source 同时声明 shared access 和 exclusive access，也避免依赖函数参数求值顺序。

## Fanout Compatibility 规则

第二层校验按 parent 聚合 direct children 的 `prev access` 使用情况，再结合该 parent 的 `fanout policy` 判定是否合法。

### `fanout_auto_policy`

- direct children 中没有 `prev access`：合法
- 只有一个 child 使用 `prev access`：允许任意合法 `prev access`
- 有多个 children 使用 `prev access`：只允许 shared access
  - `borrow_prev`
  - `copy_prev`

### `fanout_shared_readonly_policy`

无论 direct child 数量多少，只允许：

- `borrow_prev`
- `copy_prev`

任何 `borrow_prev_mut` / `consume_prev` 都会被拒绝。

### `fanout_consume_with_copies_policy`

必须满足：

- `consume_prev` child 数量为 `0..1`
- 如果存在 `consume_prev`，则其他使用 `prev access` 的 children 必须全是 `copy_prev`
- 不允许出现 `borrow_prev`
- 不允许出现 `borrow_prev_mut`

可以合法表达：

- 一个 `consume_prev` + 一个 `copy_prev`
- 一个 `consume_prev` + 多个 `copy_prev`
- 只有多个 `copy_prev`

不可以表达：

- 多个 `consume_prev`
- `consume_prev + borrow_prev`
- `consume_prev + borrow_prev_mut`

## Retry 规则

`consume_prev(...)` 当前禁止和 `with_retry(...)` 组合。

原因是：

- `consume_prev(...)` 会消费 direct parent source
- `with_retry(...)` 会重复执行同一个 task

如果允许两者组合，那么第一次 consume 之后，后续 retry 将不再面对一个稳定 source，语义不清晰。

因此当前规则是：

- `borrow_prev(...)` 可以用于 retry task
- `borrow_prev_mut(...)` 可以用于 retry task
- `copy_prev(...)` 可以用于 retry task
- `consume_prev(...)` 不可以用于 retry task

## Runtime 语义

### 一般情况

对不需要 staging 的路径：

- `borrow_prev(...)` 从 direct parent slot 以 readonly source 读取
- `borrow_prev_mut(...)` 从 direct parent slot 以 mutable source 读取
- `copy_prev(...)` 从 direct parent slot 复制出 owned value
- `consume_prev(...)` 从 direct parent slot 以 consume / move 语义读取

### `consume_prev` 的 resolve 语义

`consume_prev<T>()` 的 `resolve` 会把 direct parent payload 当作 rvalue source 使用：

- 目标参数是 `T&&` 时，返回 `T&&`
- 目标参数是 `T` 时，用 `std::move(src)` 来构造新的 `T`

这里的关键点是：

- `consume_prev` 的 consume 作用在上游 source
- `T` 或 `T&&` 只是在描述当前 task 想如何接这个 consumed source

### `fanout_consume_with_copies_policy` 的 staging

当 parent 使用 `fanout_consume_with_copies_policy`，并且 direct children 中同时存在：

- 一个 `consume_prev`
- 至少一个 `copy_prev`

executor 会在 children 开始前先准备 `fanout staging`。

当前实现已经优化成 **per-parent shared stage**：

- 每个需要 staging 的 parent 只保留一份 shared staged object
- 所有 `copy_prev` children 都把这份 staged object 视为 readonly upstream source
- 原始 parent payload 仍保留给唯一的 `consume_prev` child

这保证了：

- sibling 执行顺序不会改变语义
- `consume_prev` 不会抢在 `copy_prev` 的复制源准备之前发生
- `copy_prev` 的公开语义不变，仍然是 non-destructive owned acquisition

### 为什么 `copy_prev` child 仍然从 stage 再复制一次

当前没有把 staged object 直接 move 给某个 `copy_prev` child，原因是：

- `copy_prev` 的公开语义是“从 upstream source 复制一个 owned value”
- shared stage 只是把 source 从 parent slot 换成了 staged slot
- 如果让某个 child 直接领取 staged object，会让 `copy_prev` 在 staged path 下变成不同的 ownership 语义

因此当前模型是：

1. parent output 复制到 shared stage 一次
2. 每个 `copy_prev` child 再从 shared stage 复制出自己的 owned value

这是刻意保留语义稳定性的结果。

## Executor 行为

### Recursive serial DFS

recursive DFS 版本中的 `fanout` prepare / destroy 路径大多是 compile-time 分支：

- 不需要 staging 的 parent，prepare / destroy 会被 `if constexpr` 直接裁掉
- 额外 runtime 开销接近零

### Explicit heap stack DFS

explicit heap stack 版本需要根据 runtime `node_index` 驱动统一循环，因此天然会有 runtime 分支。

当前实现做了一个额外优化：

- frame 上预计算 `requires_fanout_staging`
- 只有真的需要 staging 的 node 才会触发 prepare / destroy dispatch

所以在 explicit stack 版本中：

- 不需要 staging 的 node 只保留一个便宜的 runtime 布尔判断
- 不再经过无意义的 prepare / destroy dispatch + no-op 路径

## 约束与当前边界

当前实现仍有以下边界：

- root 不能使用任何 `prev access`
- direct parent output 为 `void` 时不能使用任何 `prev access`
- `copy_prev<T>()` 要求 `T` 可由 `const T&` 构造
- `fanout_consume_with_copies_policy` 当前只做 shared stage，不做“把 stage 直接 move 给某个 child”的进一步优化

当前不支持这些组合：

- 一个 `borrow_prev_mut`，其他 child `borrow/copy`
- 一个 `consume_prev`，其他 child `borrow`
- 任意依赖 sibling 执行顺序才能解释清楚的 mixed fanout 语义

## 测试覆盖

当前测试已经覆盖：

- `prev access` 绑定规则与 task-local 互斥规则
- 三种 `fanout policy` 的 compile-time compatibility
- `consume_prev + copy_prev...` 在两种 `LayoutPolicy` 与两种 `ExecPolicy` 下行为一致
- `fanout_consume_with_copies_policy` 的非法组合会被 compile-time 拒绝
- shared stage 会在 `consume` child 执行前准备好
- 多个 `copy_prev` child 共享同一份 staged source，只发生一次 staging copy
- `abort_execution` / subtree unwind 时 staging 正常销毁，不泄漏资源

## 小结

当前模型可以概括为：

- `prev access`
  描述 child 如何访问 direct parent payload
- `fanout policy`
  描述 parent 是否允许这样分发自己的 output
- validation
  先做 task-local access 校验，再做 parent-child fanout compatibility 校验
- runtime
  默认路径不引入额外 staging，只有 `fanout_consume_with_copies_policy` 需要 shared stage

这让：

- API 语义稳定
- compile-time 报错位置更明确
- runtime 行为不依赖 sibling 顺序
- recursive 与 explicit-stack 两套 executor 保持一致
