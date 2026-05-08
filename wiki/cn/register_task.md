# 注册任务

[English](../en/register_task.md) | [中文](register_task.md)

YOrch 当前把任务分成三种主要语义：普通任务、direct-output 任务和 forward-prev 任务。它们最大的区别，是和 **插槽 slot** 的关系不同

### 普通任务
普通任务是最直接的任务形式，它通过 callable 的返回值产出数据

- 如果返回普通值 `T`，YOrch 会把这个返回值移动进当前节点的 slot
- 如果返回 `void`，这个任务不会向 slot 写入数据
- 这种模型适合可以正常返回、可以移动的结果对象

### direct-output 任务
direct-output 任务不通过返回值产出主要数据，而是拿到当前节点 slot 的写入入口

- callable 可以把结果直接构造在 slot 里
- 这种模型适合不方便移动、不能移动，或希望避免中间临时对象的结果对象
- slot 是否被成功构造，由任务自己负责

### forward-prev 任务
forward-prev 任务不会为当前节点创建新的 slot 内容，而是复用 direct parent 已经存在的 slot

- callable 直接拿到上一个节点的输出对象，并在原对象上继续处理
- 当前节点的逻辑输出仍然是这个被转发的对象
- callable 不通过返回值产出新的数据
- 这种模型适合对同一个 payload 做连续加工，而不是每一步都创建一个新对象

## 插槽 slot
为什么存在三种不同的任务类型？这是因为对于每一个注册的任务，YOrch 都需要知道它和 slot 的关系：

- 普通任务：返回值先产生出来，再移动进当前节点 slot
- direct-output 任务：结果直接在当前节点 slot 中构造
- forward-prev 任务：当前节点不新建 slot 内容，而是继续使用父节点已有的 slot

对于不输出结果的任务，相关 slot 会被优化掉
  
## 任务注册 API

callable 可以直接写在 `task` 接口里，也可以先定义 lambda、自由函数或成员函数，再把它注册成任务。

```cpp
auto t = yorch::task([]() {
    return Result {0};
})();
```

### 普通任务

普通 callable 使用 `task(...)` 注册：

```cpp
struct Input {
    int value() const;
};

struct Result {
    explicit Result(int);
};

auto make_result = [](const Input& input) -> Result {
    return Result {input.value()};
};

auto t = yorch::task(make_result)(...);
```

自由函数也可以注册为普通任务：

```cpp
Result make_result(const Input& input);

auto t = yorch::task(make_result)(...);
```

成员函数使用 `task_member(...)` 注册：

成员函数任务必须额外指定一个接收对象。执行时，YOrch 会先取得这个对象，再在该对象上调用成员函数。

```cpp
struct Worker {
    Result make_result(const Input& input);
};

auto t = yorch::task_member(
    &Worker::make_result,
    ... // receiver
)(...);
```

### direct-output 任务

direct-output callable 的最后一个参数必须是 `yorch::direct_out<T>`，这里的 `T` 就是当前任务的输出类型。

除了最后的 `direct_out<T>`，前面的参数仍然按普通参数描述。

普通 callable 使用 `task_into(...)` 注册：

```cpp
auto emit_result = [](const Input& input, yorch::direct_out<Result> out) -> void {
    out.emplace(input.value());
};

auto t = yorch::task_into(emit_result)(...);
```

自由函数也可以注册为 direct-output 任务：

```cpp
void emit_result(const Input& input, yorch::direct_out<Result> out);

auto t = yorch::task_into(emit_result)(...);
```

成员函数使用 `task_into_member(...)` 注册：

direct-output 成员函数同样必须指定接收对象，并且成员函数的最后一个参数也必须是 `yorch::direct_out<T>`。

```cpp
struct Worker {
    void emit_result(const Input& input, yorch::direct_out<Result> out);
};

auto t = yorch::task_into_member(
    &Worker::emit_result,
    ... // receiver
)(...);
```

### forward-prev 任务

forward-prev 任务要求 callable 使用上一个节点的输出对象，并把这个对象继续作为当前节点的逻辑输出。

当前模型下，被转发的 payload 需要以可变引用的形式进入 callable。

这里可以先把 forward-prev 理解成“不返回新值，只修改并继续转发已有 payload”的任务形式。流程控制相关的返回值将在后面再单独说明。

普通 callable 使用 `task_forward_prev(...)` 注册：

```cpp
struct Payload {
    void update();
};

auto update_payload = [](Payload& payload) {
    payload.update();
};

auto t = yorch::task_forward_prev(update_payload)(...);
```

自由函数也可以注册为 forward-prev 任务：

```cpp
void update_payload(Payload& payload);

auto t = yorch::task_forward_prev(update_payload)(...);
```

成员函数使用 `task_forward_prev_member(...)` 注册：

forward-prev 成员函数也必须指定接收对象。被转发的 payload 可以是成员函数参数，也可以就是这个接收对象本身；但一次 forward-prev 任务只应该有一个被转发的 payload。

```cpp
struct Worker {
    void update_payload(Payload& payload);
};

auto t = yorch::task_forward_prev_member(
    &Worker::update_payload,
    ... // receiver
)(...);
```

## 参数描述

上一节里的任务注册示例还不是完整形式，其中省略的 `...`，实际描述的是 callable 的每个参数应该从哪里来。YOrch 把这种“参数来源描述”称为 **spec**

spec 本身不立刻取值，它只是记录一种来源关系。真正执行任务时，YOrch 再根据这些描述把对应对象交给 callable

### context 参数

- 表示参数来自全局执行上下文
- 适合放执行期间共享的资源、配置、服务对象或输入数据
- 这类参数不依赖父节点输出
- 心智模型是“从外部环境里按类型取一个对象”

### value 参数

- 表示参数是注册任务时一起保存下来的固定值
- 适合放常量、轻量配置或提前准备好的小对象
- 这类参数跟 task tree 的上下文和父节点输出都无关
- 心智模型是“任务自己带着一个值”

### prev 参数

- 表示参数来自 direct parent 的输出 slot
- 适合让子任务读取或继续处理父任务产出的数据
- 这类参数要求当前任务确实有 direct parent 输出可以访问
- 心智模型是“从上一个节点留下来的 slot 里取数据”

prev 参数又可以分成几种访问意图：

- 只读借用
  - 子任务只查看父节点输出，不修改也不拿走

- 可变借用
  - 子任务直接修改父节点输出对象本身

- 复制
  - 子任务从父节点输出复制出一个新值来使用

- 消费
  - 子任务接管父节点输出对象的内容
  - 适合只需要使用一次、希望转移所有权的场景

## 参数描述 API

下面的例子都使用 **普通任务** 来说明。参数描述的数量和顺序，需要与 callable 的参数一一对应。

### 从 context 取参数

使用 `yorch::from_ctx<T>()` 表示这个参数来自执行上下文中的 `T` 对象。

```cpp
struct Config {
    int base = 0;
};

auto make_value = [](const Config& config) {
    return config.base + 1;
};

auto t = yorch::task(make_value)(
    yorch::from_ctx<Config>());
```

### 使用注册时保存的值

使用 `yorch::value(x)` 表示把 `x` 保存到任务里，执行任务时再作为参数传入。

```cpp
auto add = [](int lhs, int rhs) {
    return lhs + rhs;
};

auto t = yorch::task(add)(
    yorch::value(1),
    yorch::value(2));
```

如果希望保存的是一个外部对象的引用，可以使用 `std::ref(...)`：

```cpp
int counter = 0;

auto increase = [](int& value) {
    ++value;
};

auto t = yorch::task(increase)(
    yorch::value(std::ref(counter)));
```

### 只读借用父节点输出

使用 `yorch::borrow_prev<T>()` 表示从 direct parent 的输出 slot 中取得 `const T&`。

```cpp
struct Payload {
    int value = 0;
};

auto read_payload = [](const Payload& payload) {
    return payload.value;
};

auto t = yorch::task(read_payload)(
    yorch::borrow_prev<Payload>());
```

### 可变借用父节点输出

使用 `yorch::borrow_prev_mut<T>()` 表示从 direct parent 的输出 slot 中取得 `T&`。

```cpp
auto update_payload = [](Payload& payload) {
    ++payload.value;
};

auto t = yorch::task(update_payload)(
    yorch::borrow_prev_mut<Payload>());
```

### 复制父节点输出

使用 `yorch::copy_prev<T>()` 表示从 direct parent 的输出 slot 中复制出一个新的 `T`。

```cpp
auto copy_payload = [](Payload payload) {
    return payload.value;
};

auto t = yorch::task(copy_payload)(
    yorch::copy_prev<Payload>());
```

### 消费父节点输出

使用 `yorch::consume_prev<T>()` 表示从 direct parent 的输出 slot 中移动出 `T`。

```cpp
auto consume_payload = [](Payload payload) {
    return payload.value;
};

auto t = yorch::task(consume_payload)(
    yorch::consume_prev<Payload>());
```

也可以让 callable 直接接收 `T&&`：

```cpp
auto consume_payload = [](Payload&& payload) {
    return payload.value;
};

auto t = yorch::task(consume_payload)(
    yorch::consume_prev<Payload>());
```

## 复杂案例

下面的例子把前面的任务类型和参数描述组合起来。这里只展示任务注册本身，不展开 task tree 的构建方式。

### direct-output 任务使用 context 参数和值参数

```cpp
struct Config {
    int scale = 1;
};

struct Result {
    explicit Result(int);
};

auto emit_result = [](const Config& config, int base, yorch::direct_out<Result> out) {
    out.emplace(base * config.scale);
};

auto t = yorch::task_into(emit_result)(
    yorch::from_ctx<Config>(),
    yorch::value(10));
```

这里 `emit_result` 的最后一个参数是 `yorch::direct_out<Result>`，所以它注册为 direct-output 任务。前两个参数分别来自 context 和注册时保存的值。

### direct-output 成员函数任务

```cpp
struct Config {
    int scale = 1;
};

struct Result {
    explicit Result(int);
};

struct Emitter {
    void emit(const Config& config, int base, yorch::direct_out<Result> out) {
        out.emplace(base * config.scale);
    }
};

auto t = yorch::task_into_member(
    &Emitter::emit,
    yorch::from_ctx<Emitter>())(
    yorch::from_ctx<Config>(),
    yorch::value(10));
```

这里 `yorch::from_ctx<Emitter>()` 描述的是成员函数的接收对象，后面的两个 spec 才对应 `emit(...)` 的普通参数。最后的 `direct_out<Result>` 不需要再写一个 spec。

### forward-prev 任务继续加工父节点输出

```cpp
struct Payload {
    int value = 0;
};

auto add_delta = [](Payload& payload, int delta) {
    payload.value += delta;
};

auto t = yorch::task_forward_prev(add_delta)(
    yorch::borrow_prev_mut<Payload>(),
    yorch::value(3));
```

这里 `borrow_prev_mut<Payload>()` 表示当前任务会直接修改父节点输出的 `Payload`。这个 `Payload` 会继续作为当前任务的逻辑输出向后传递。

### forward-prev 成员函数任务

```cpp
struct Payload {
    int value = 0;
};

struct Service {
    void add(Payload& payload, int delta) {
        payload.value += delta;
    }
};

auto t = yorch::task_forward_prev_member(
    &Service::add,
    yorch::from_ctx<Service>())(
    yorch::borrow_prev_mut<Payload>(),
    yorch::value(3));
```

这里成员函数的接收对象来自 context，被转发的 payload 来自父节点输出。一次 forward-prev 任务只应该有一个被转发的 payload。

### 成员函数接收对象也可以来自保存的值

```cpp
struct Worker {
    int base = 0;

    int make(int delta) {
        return base + delta;
    }
};

Worker worker {10};

auto t = yorch::task_member(
    &Worker::make,
    yorch::value(std::ref(worker)))(
    yorch::value(5));
```

这里 `std::ref(worker)` 让任务保存的是外部 `worker` 的引用。执行时会在这个对象上调用 `Worker::make`。

## 流程控制

任务的返回值除了可以产出数据，也可以表达“这一步执行得怎么样”。YOrch 会把任务返回值理解成两部分：

- 控制状态
  - 任务是否成功
  - 是否失败
  - 是否请求重试
  - 是否中止当前分支或整次执行

- 输出数据
  - 当前任务是否向 slot 写入一个值
  - 后续子任务是否可以继续读取这个值

### 不返回流程控制结果是什么意思

如果 callable 返回 `void` 或普通值 `T`，就表示这个任务没有显式表达流程控制意图。

- 返回 `void`
  - 表示任务正常完成
  - 不产出 slot 数据

- 返回普通值 `T`
  - 表示任务正常完成
  - 返回值会作为输出数据进入当前节点 slot

也就是说，不返回 `step_result` 并不是“不知道成功还是失败”，而是“没有特别说明时按成功处理”。

### 返回 `step_result`

`yorch::step_result` 只表达流程控制状态，不携带输出数据。

- `success`
  - 当前任务成功

- `failure`
  - 当前任务失败

- `retry`
  - 当前任务请求重试

- `abort_branch`
  - 停止当前分支

- `abort_execution`
  - 停止整次执行

因为 `step_result` 不携带数据，所以返回它的普通任务不会向当前节点 slot 写入输出值。

### 返回 `task_result<T>`

`yorch::task_result<T>` 同时表达流程控制和输出数据。

- 成功时
  - 携带一个 `T`
  - 这个 `T` 会进入当前节点 slot

- 非成功时
  - 只携带流程控制状态
  - 不会构造输出值

它适合“有时产出值，有时提前失败或中止”的任务。

### 普通值返回

```cpp
auto make_payload = []() {
    return Payload {1};
};

auto t = yorch::task(make_payload)();
```

这里 `Payload {1}` 会进入当前节点 slot，流程状态按成功处理。

### `void` 返回

```cpp
auto touch = [](Payload& payload) {
    payload.value += 1;
};

auto t = yorch::task(touch)(
    yorch::borrow_prev_mut<Payload>());
```

这里任务只执行动作，不产出新的 slot 数据，流程状态按成功处理。

### `step_result` 返回

```cpp
auto check = [](const Payload& payload) -> yorch::step_result {
    if (payload.value < 0) {
        return yorch::step_result::failure();
    }

    return yorch::step_result::success();
};

auto t = yorch::task(check)(
    yorch::borrow_prev<Payload>());
```

这里任务只决定成功或失败，不产出新的 slot 数据。

### `task_result<T>` 返回

```cpp
auto parse = [](const std::string& text) -> yorch::task_result<Payload> {
    if (text.empty()) {
        return yorch::task_result<Payload>::failure();
    }

    return yorch::task_result<Payload>::success(Payload {static_cast<int>(text.size())});
};

auto t = yorch::task(parse)(
    yorch::from_ctx<std::string>());
```

这里成功时会产出 `Payload` 并写入 slot；失败时只返回失败状态，不构造 `Payload`。

### direct-output 与 forward-prev

direct-output 任务的数据已经通过 `direct_out<T>` 写入 slot，所以返回值只需要表达流程控制：

如果返回 `void`，表示任务没有显式失败路径；只要 callable 正常执行完成，就按成功处理。

```cpp
auto emit = [](yorch::direct_out<Payload> out) -> yorch::step_result {
    out.emplace(1);
    return yorch::step_result::success();
};

auto t = yorch::task_into(emit)();
```

forward-prev 任务的逻辑输出来自父节点已有 slot，所以它也不通过返回值产出新数据。如果需要表达失败或中止，可以返回 `step_result`：

如果返回 `void`，同样表示正常执行完成就是成功。

```cpp
auto update = [](Payload& payload) -> yorch::step_result {
    if (payload.value < 0) {
        return yorch::step_result::failure();
    }

    payload.value += 1;
    return yorch::step_result::success();
};

auto t = yorch::task_forward_prev(update)(
    yorch::borrow_prev_mut<Payload>());
```

## 任务适配器 task adapter

task adapter 用来在不改动原始 callable 的情况下，给任务外面包上一层额外行为。当前主要有两类：

- 把异常转换成失败结果
- 根据 `retry` 状态重新执行任务

在 `task(...)` 这组注册 API 中，可以通过 `yorch::adapters(...)` 把一个或多个 adapter 一起传进去。

### 异常转失败

`adapt_catch_as_failure()` 用来捕获任务执行过程中抛出的异常，并把异常转换成失败结果。

YOrch 的主执行路径采用无异常设计：任务如果可能抛异常，或者 callable 没有提供 `noexcept` 调用面，就应该先通过这个 adapter 转成不会向外抛异常的任务。

如果不传 policy，默认规则只适合返回 `void` 或 `yorch::step_result` 的任务，因为这两类任务可以直接构造出失败状态。

```cpp
auto may_throw = []() -> yorch::step_result {
    throw std::runtime_error("boom");
    return yorch::step_result::success();
};

auto t = yorch::task(
    may_throw,
    yorch::adapters(yorch::adapt_catch_as_failure()))();
```

也可以传入自定义 catch policy。policy 接收捕获到的 `std::exception_ptr`，并且必须是 `noexcept` 调用。

如果需要针对不同异常类型做不同处理，可以在 policy 内部重新抛出这个 `exception_ptr` 再捕获分类；但异常不能逃出 policy，否则会破坏无异常调用面。

```cpp
auto fallback = [](const std::exception_ptr&) noexcept {
    return yorch::step_result::abort_branch();
};

auto t = yorch::task(
    may_throw,
    yorch::adapters(yorch::adapt_catch_as_failure(fallback)))();
```

按异常类型分类处理：

```cpp
auto fallback = [](const std::exception_ptr& ep) noexcept {
    try {
        if (ep) {
            std::rethrow_exception(ep);
        }
    } catch (const std::invalid_argument&) {
        return yorch::step_result::abort_branch();
    } catch (...) {
        return yorch::step_result::failure();
    }

    return yorch::step_result::failure();
};

auto t = yorch::task(
    may_throw,
    yorch::adapters(yorch::adapt_catch_as_failure(fallback)))();
```

如果任务返回的是普通值或 `task_result<T>`，自定义 policy 需要返回一个能转换成该任务返回类型的结果。

```cpp
auto parse = [](const std::string& text) -> yorch::task_result<int> {
    if (text == "bad") {
        throw std::runtime_error("bad input");
    }

    return yorch::task_result<int>::success(static_cast<int>(text.size()));
};

auto fallback = [](const std::exception_ptr&) noexcept {
    return yorch::task_result<int>::failure();
};

auto t = yorch::task(
    parse,
    yorch::adapters(yorch::adapt_catch_as_failure(fallback)))(
    yorch::from_ctx<std::string>());
```

direct-output 任务也可以使用异常转失败。对于 direct-output，policy 返回 `yorch::step_result`。

```cpp
auto emit = [](yorch::direct_out<Payload> out) {
    throw std::runtime_error("boom");
    out.emplace(1);
};

auto fallback = [](const std::exception_ptr&) noexcept {
    return yorch::step_result::failure();
};

auto t = yorch::task_into(
    emit,
    yorch::adapters(yorch::adapt_catch_as_failure(fallback)))();
```

### retry

`adapt_retry(policy)` 用来在任务返回 `retry` 状态时重新执行任务。它只对能表达 `retry` 的返回类型有意义，例如 `yorch::step_result` 和 `yorch::task_result<T>`。

```cpp
int attempts = 0;

auto flaky = [&]() -> yorch::step_result {
    ++attempts;

    if (attempts < 3) {
        return yorch::step_result::retry();
    }

    return yorch::step_result::success();
};

auto t = yorch::task(
    flaky,
    yorch::adapters(yorch::adapt_retry(yorch::retry_fixed_policy {2})))();
```

当前内置 retry policy 有三种：

- `retry_fixed_policy`
  - 允许固定次数的 retry
  - 次数耗尽后，把最终的 `retry` 转成 `failure`

```cpp
auto t = yorch::task(
    flaky,
    yorch::adapters(yorch::adapt_retry(yorch::retry_fixed_policy {3})))();
```

- `retry_fixed_passthrough_policy`
  - 允许固定次数的 retry
  - 次数耗尽后，保留最终的 `retry` 状态

```cpp
auto t = yorch::task(
    flaky,
    yorch::adapters(yorch::adapt_retry(yorch::retry_fixed_passthrough_policy {3})))();
```

- `retry_forever_policy`
  - 只要任务继续返回 `retry`，就持续重试

```cpp
auto t = yorch::task(
    flaky,
    yorch::adapters(yorch::adapt_retry(yorch::retry_forever_policy {})))();
```

也可以自定义 retry policy。policy 需要提供 `should_retry(retry_count)`，其中 `retry_count` 表示已经批准过多少次重试。

```cpp
struct my_retry_policy {
    bool should_retry(std::size_t retry_count) const noexcept {
        return retry_count < 5;
    }
};

auto t = yorch::task(
    flaky,
    yorch::adapters(yorch::adapt_retry(my_retry_policy {})))();
```

如果自定义 policy 没有提供“次数耗尽后如何处理”的规则，最终的 `retry` 会原样保留。也可以额外提供 `on_exhausted(...)` 来决定耗尽后的结果。

```cpp
struct fail_after_retry_policy {
    bool should_retry(std::size_t retry_count) const noexcept {
        return retry_count < 5;
    }

    static yorch::step_result on_exhausted(yorch::step_result) noexcept {
        return yorch::step_result::failure();
    }
};

auto t = yorch::task(
    flaky,
    yorch::adapters(yorch::adapt_retry(fail_after_retry_policy {})))();
```

### 组合多个 adapter

多个 adapter 可以组合使用。它们会按书写顺序依次应用：前面的 adapter 更靠近原始任务，后面的 adapter 包在更外层。

```cpp
auto t = yorch::task(
    flaky,
    yorch::adapters(
        yorch::adapt_catch_as_failure(),
        yorch::adapt_retry(yorch::retry_fixed_policy {2})))();
```

这个例子中，原始任务先被 catch adapter 包住，然后外层 retry adapter 根据结果决定是否重试。
