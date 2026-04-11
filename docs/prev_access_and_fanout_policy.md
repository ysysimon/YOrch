# Prev Access 与 Fanout Policy 现状说明

## 背景

本次修改先独立引入了 `prev access` 语义，还**没有**实现显式的 `fanout policy` 类型。

当前设计把这两个层面拆开理解：

- `prev access`
  子节点想如何访问 direct parent payload
- `fanout`
  direct parent payload 在当前树结构下是否能被独占访问，还是只能共享只读访问

目前 `fanout` 仍然通过 plan 结构里的 “parent 有几个 direct child” 来近似表达：

- 单 child 路径：允许 exclusive access
- 多 child 路径：只允许 shared readonly access

这是一条**临时结构规则**，后续如果引入真正的 `fanout policy`，这里可以再替换。

## Public API

旧的 `from_prev<T>()` 已经移除，当前 public API 有四种 direct-parent access spec：

### `borrow_prev<T>()`

含义：

- 只读借用 direct parent payload
- 共享访问
- 不消费 source

绑定规则：

- 只能绑定到 `const T&`

### `borrow_prev_mut<T>()`

含义：

- 独占可变借用 direct parent payload
- 不消费 source
- 需要独占访问权

绑定规则：

- 只能绑定到 `T&`

### `copy_prev<T>()`

含义：

- 从 direct parent payload 复制出一个新的 `T` 值
- shared non-exclusive access
- 不消费 source
- 不修改 source

绑定规则：

- 只能绑定到 `T`

### `consume_prev<T>()`

含义：

- 消费 direct parent payload
- 表达 one-shot consume access
- 不会 eager destroy slot，slot 仍按现有生命周期在执行结束时统一析构

绑定规则：

- 只能绑定到 `T` 或 `T&&`

两种接收方式的区别：

- `T`
  从 parent source 中 `move` 出一个新的值对象
- `T&&`
  直接把 parent source 作为 rvalue reference 暴露给下游

## Spec 归一化

四种 spec 都会先对模板参数做 `cvref` 归一化，只保留基础类型：

```cpp
borrow_prev<const int&>()          -> borrow_prev_t<int>
borrow_prev_mut<const int&>()      -> borrow_prev_mut_t<int>
copy_prev<const int&>()            -> copy_prev_t<int>
consume_prev<volatile std::string&&>() -> consume_prev_t<std::string>
```

这里的归一化只影响 spec 自身记录的类型，不代表这些 access mode 可以绑定到任意 `const&` / `&&` 参数。真正的绑定规则仍以上一节为准。

## Task 内规则

`prev access` 的第一层校验是 **task-local validation**，也就是先不看树结构，只看一个 task 自己写得是否合法。

### 1. 参数类型必须和 access mode 精确匹配

- `borrow_prev<T>()` -> `const T&`
- `borrow_prev_mut<T>()` -> `T&`
- `copy_prev<T>()` -> `T`
- `consume_prev<T>()` -> `T` 或 `T&&`

如果参数类型不匹配，`run_plan(...)` 会在 compile time 拒绝这个 plan。

### 2. shared access 可以重复出现或混用

`borrow_prev(...)` 和 `copy_prev(...)` 都属于 shared non-exclusive access。

因此在一个 task 内：

- 多个 `borrow_prev(...)` 合法
- 多个 `copy_prev(...)` 合法
- `borrow_prev(...) + copy_prev(...)` 合法

例如：

```cpp
yorch::bind(
    [](const int& a, int b) noexcept -> yorch::step_result {
        return yorch::step_result::success();
    },
    yorch::borrow_prev<int>(),
    yorch::copy_prev<int>())
```

### 3. exclusive access 必须唯一

`borrow_prev_mut(...)` 和 `consume_prev(...)` 都属于 exclusive access。

一旦一个 task 使用了 exclusive access，它必须是这个 task 中**唯一的 prev access**。因此下面这些组合都会被拒绝：

- 多个 `borrow_prev_mut(...)`
- 多个 `consume_prev(...)`
- `borrow_prev_mut(...) + borrow_prev(...)`
- `borrow_prev_mut(...) + copy_prev(...)`
- `borrow_prev_mut(...) + consume_prev(...)`
- `consume_prev(...) + borrow_prev(...)`
- `consume_prev(...) + copy_prev(...)`

这样做的目的，是避免对同一个 direct parent source 同时声明 shared access 和 exclusive access，也避免依赖函数参数求值顺序。

## Retry 规则

`consume_prev(...)` 当前禁止和 `with_retry(...)` 组合。

原因是：

- `consume_prev(...)` 会移动 direct parent source
- `with_retry(...)` 会重复执行同一个 task

如果允许两者组合，那么第一次 consume 之后，后续 retry 将不再面对一个稳定 source，语义会变得不清晰。

因此当前规则是：

- `borrow_prev(...)` 可以用于 retry task
- `borrow_prev_mut(...)` 可以用于 retry task
- `copy_prev(...)` 可以用于 retry task
- `consume_prev(...)` 不可以用于 retry task

## Plan 结构规则

第二层校验是 **node-level structural validation**，也就是看当前 task 所在节点在 plan 结构里是否允许这种 access。

### 1. root 不能使用任何 `prev access`

因为 root 没有 direct parent payload source。

### 2. direct parent output 为 `void` 时不能使用任何 `prev access`

因为虽然有 parent 节点，但它没有可读的 payload。

### 3. 多 child fanout 下只允许 shared non-exclusive access

当前没有显式 `fanout policy`，所以用 direct child 数量近似表达 fanout 约束：

- `child_count(parent) <= 1`
  认为当前路径可以独占访问
- `child_count(parent) > 1`
  认为当前路径是 fanout 场景，只允许 shared readonly access

因此：

- 单 child 路径下：
  - `borrow_prev(...)` 合法
  - `borrow_prev_mut(...)` 合法
  - `copy_prev(...)` 合法
  - `consume_prev(...)` 合法
- 多 child 路径下：
  - `borrow_prev(...)` 合法
  - `copy_prev(...)` 合法
  - `borrow_prev_mut(...)` 非法
  - `consume_prev(...)` 非法

## 当前结论

可以把当前实现总结为：

- `borrow_prev(...)`
  shared readonly access，可重复，可用于 fanout
- `borrow_prev_mut(...)`
  exclusive mutable access，只允许单 child 路径，且 task 内唯一
- `copy_prev(...)`
  shared copy access，可重复，可用于 fanout，也可用于 retry
- `consume_prev(...)`
  exclusive consuming access，只允许单 child 路径，且 task 内唯一，并禁止与 retry 组合

## 后续演进方向

当前的 `fanout` 约束仍然是“按树结构推导 exclusivity”，还没有 public `fanout policy`。

如果后续要继续扩展，比较自然的方向是：

- 保留现有 `prev access` API 不变
- 新增显式 `fanout policy`
- 把当前 `child_count(parent) <= 1` 的临时结构规则，替换成真正的 policy compatibility check

这样可以把：

- “子节点想怎么访问上游”  
  和
- “上游是否允许这样分发”

彻底拆成两个独立维度。
