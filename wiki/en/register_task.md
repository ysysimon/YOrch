# Register Tasks

<a href="register_task.md">English</a> | <a href="../cn/register_task.md">中文</a>

YOrch currently groups tasks into three main semantics: regular tasks,
direct-output tasks, and forward-prev tasks. The biggest difference between them
is how they relate to a **slot**.

### Regular tasks

A regular task produces data through the callable return value:

- If it returns a normal value `T`, YOrch moves that value into the current
  node's slot.
- If it returns `void`, the task does not write data into a slot.
- This model fits results that can be returned and moved normally.

### Direct-output tasks

A direct-output task does not produce its main data through the return value.
Instead, it receives a write handle to the current node's slot:

- The callable can construct the result directly in the slot.
- This model fits results that are inconvenient to move, non-movable, or should
  avoid intermediate temporary objects.
- The task itself is responsible for whether the slot is successfully
  constructed.

### Forward-prev tasks

A forward-prev task does not create new slot content for the current node. It
reuses the existing slot from the direct parent:

- The callable receives the previous node's output object and continues
  processing it in place.
- The current node's logical output is still the forwarded object.
- The callable does not produce new data through its return value.
- This model fits continuous processing on the same payload instead of creating a
  new object at every step.

## Slot

Why are there three task types? For every registered task, YOrch needs to know
how that task relates to a slot:

- Regular task: produce a return value first, then move it into the current
  node's slot.
- Direct-output task: construct the result directly in the current node's slot.
- Forward-prev task: do not create new slot content; keep using the parent's
  existing slot.

For tasks with no output, the related slot can be optimized away.

## Task Registration API

A callable can be written directly in the `task` API, or you can define a lambda,
free function, or member function first and then register it as a task.

```cpp
auto t = yorch::task([]() {
    return Result {0};
})();
```

### Regular tasks

Use `task(...)` for regular callables:

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

Free functions can also be registered as regular tasks:

```cpp
Result make_result(const Input& input);

auto t = yorch::task(make_result)(...);
```

Use `task_member(...)` for member functions. A member function task must also
specify a receiver object. During execution, YOrch obtains that object first and
then invokes the member function on it.

```cpp
struct Worker {
    Result make_result(const Input& input);
};

auto t = yorch::task_member(
    &Worker::make_result,
    ... // receiver
)(...);
```

### Direct-output tasks

The last argument of a direct-output callable must be `yorch::direct_out<T>`,
where `T` is the current task's output type.

All arguments before the final `direct_out<T>` are still described with normal
argument specs.

Use `task_into(...)` for regular callables:

```cpp
auto emit_result = [](const Input& input, yorch::direct_out<Result> out) -> void {
    out.emplace(input.value());
};

auto t = yorch::task_into(emit_result)(...);
```

Free functions can also be registered as direct-output tasks:

```cpp
void emit_result(const Input& input, yorch::direct_out<Result> out);

auto t = yorch::task_into(emit_result)(...);
```

Use `task_into_member(...)` for member functions. A direct-output member function
also needs a receiver, and its last parameter must also be `yorch::direct_out<T>`.

```cpp
struct Worker {
    void emit_result(const Input& input, yorch::direct_out<Result> out);
};

auto t = yorch::task_into_member(
    &Worker::emit_result,
    ... // receiver
)(...);
```

### Forward-prev tasks

A forward-prev task requires the callable to use the previous node's output
object and keep that object as the current node's logical output.

In the current model, the forwarded payload enters the callable as a mutable
reference.

You can initially think of forward-prev as a task form that "does not return a
new value; it mutates and forwards an existing payload". Flow-control return
values are described later.

Use `task_forward_prev(...)` for regular callables:

```cpp
struct Payload {
    void update();
};

auto update_payload = [](Payload& payload) {
    payload.update();
};

auto t = yorch::task_forward_prev(update_payload)(...);
```

Free functions can also be registered as forward-prev tasks:

```cpp
void update_payload(Payload& payload);

auto t = yorch::task_forward_prev(update_payload)(...);
```

Use `task_forward_prev_member(...)` for member functions. A forward-prev member
function also needs a receiver. The forwarded payload can be a member-function
argument or the receiver itself, but one forward-prev task should have only one
forwarded payload.

```cpp
struct Worker {
    void update_payload(Payload& payload);
};

auto t = yorch::task_forward_prev_member(
    &Worker::update_payload,
    ... // receiver
)(...);
```

## Argument Specs

The task registration examples above omit `...`. In complete code, those
positions describe where each callable argument comes from. YOrch calls this
kind of argument-source description a **spec**.

A spec does not fetch a value immediately. It records a source relationship. At
execution time, YOrch uses these descriptions to pass the corresponding objects
to the callable.

### Context arguments

- The argument comes from the global execution context.
- This is useful for shared resources, configuration, services, or input data.
- These arguments do not depend on the parent node's output.
- The mental model is "get an object by type from the external environment".

### Value arguments

- The argument is a fixed value stored together with the registered task.
- This is useful for constants, lightweight configuration, or small objects that
  are prepared in advance.
- These arguments are independent of the task tree context and parent output.
- The mental model is "the task carries its own value".

### Prev arguments

- The argument comes from the direct parent's output slot.
- This is useful when a child task reads or continues processing data produced by
  its parent.
- These arguments require that the current task has a direct parent output to
  access.
- The mental model is "get data from the slot left by the previous node".

Prev arguments can express several access intentions:

- Read-only borrow: the child only observes the parent output.
- Mutable borrow: the child modifies the parent output object directly.
- Copy: the child copies a new value from the parent output.
- Consume: the child takes ownership of the parent output object.

## Argument Spec API

The following examples use **regular tasks**. The number and order of specs must
match the callable arguments one by one.

### Get an argument from context

Use `yorch::from_ctx<T>()` to say that this argument comes from a `T` object in
the execution context.

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

### Use a value saved at registration time

Use `yorch::value(x)` to save `x` into the task and pass it as an argument during
execution.

```cpp
auto add = [](int lhs, int rhs) {
    return lhs + rhs;
};

auto t = yorch::task(add)(
    yorch::value(1),
    yorch::value(2));
```

If you want to store a reference to an external object, use `std::ref(...)`:

```cpp
int counter = 0;

auto increase = [](int& value) {
    ++value;
};

auto t = yorch::task(increase)(
    yorch::value(std::ref(counter)));
```

### Read-only borrow of parent output

Use `yorch::borrow_prev<T>()` to get `const T&` from the direct parent's output
slot.

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

### Mutable borrow of parent output

Use `yorch::borrow_prev_mut<T>()` to get `T&` from the direct parent's output
slot.

```cpp
auto update_payload = [](Payload& payload) {
    ++payload.value;
};

auto t = yorch::task(update_payload)(
    yorch::borrow_prev_mut<Payload>());
```

### Copy parent output

Use `yorch::copy_prev<T>()` to copy a new `T` from the direct parent's output
slot.

```cpp
auto copy_payload = [](Payload payload) {
    return payload.value;
};

auto t = yorch::task(copy_payload)(
    yorch::copy_prev<Payload>());
```

### Consume parent output

Use `yorch::consume_prev<T>()` to move a `T` out of the direct parent's output
slot.

```cpp
auto consume_payload = [](Payload payload) {
    return payload.value;
};

auto t = yorch::task(consume_payload)(
    yorch::consume_prev<Payload>());
```

The callable may also receive `T&&` directly:

```cpp
auto consume_payload = [](Payload&& payload) {
    return payload.value;
};

auto t = yorch::task(consume_payload)(
    yorch::consume_prev<Payload>());
```

## Combined Examples

These examples combine the task types and argument specs. They only show task
registration, not task tree construction.

### Direct-output task with context and value arguments

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

The last argument of `emit_result` is `yorch::direct_out<Result>`, so it is
registered as a direct-output task. The first two arguments come from context and
from a saved value.

### Direct-output member function task

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

Here `yorch::from_ctx<Emitter>()` describes the member-function receiver. The two
following specs correspond to the normal parameters of `emit(...)`. The final
`direct_out<Result>` does not need a spec.

### Forward-prev task continuing parent output

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

`borrow_prev_mut<Payload>()` means this task directly mutates the parent's
`Payload`. That same `Payload` continues as the current task's logical output.

### Forward-prev member function task

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

The member-function receiver comes from context, and the forwarded payload comes
from the parent output. One forward-prev task should have only one forwarded
payload.

### Member function receiver from a saved value

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

`std::ref(worker)` makes the task store a reference to the external `worker`.
Execution invokes `Worker::make` on that object.

## Flow Control

A task return value can produce data, but it can also express "how this step
finished". YOrch interprets task return values as two parts:

- Control state: success, failure, retry request, abort current branch, or abort
  the whole execution.
- Output data: whether this task wrote a value into the slot and whether child
  tasks can read that value.

### What if a task does not return a flow-control result?

If a callable returns `void` or a normal value `T`, it has no explicit
flow-control intention.

- Returning `void` means the task completed normally and produces no slot data.
- Returning `T` means the task completed normally and the return value becomes
  output data in the current node's slot.

So not returning `step_result` does not mean "unknown success"; it means "treat
as success unless stated otherwise".

### Return `step_result`

`yorch::step_result` only expresses flow-control state. It does not carry output
data.

- `success`: the task succeeded.
- `failure`: the task failed.
- `retry`: the task requests retry.
- `abort_branch`: stop the current branch.
- `abort_execution`: stop the entire execution.

Because `step_result` carries no data, a regular task returning it does not write
output into the current node's slot.

### Return `task_result<T>`

`yorch::task_result<T>` expresses both flow control and output data.

- On success, it carries a `T`, and that `T` enters the current node's slot.
- On non-success, it carries only the flow-control state and does not construct
  an output value.

This fits tasks that sometimes produce a value and sometimes fail or abort early.

### Normal value return

```cpp
auto make_payload = []() {
    return Payload {1};
};

auto t = yorch::task(make_payload)();
```

`Payload {1}` enters the current node's slot, and the flow state is treated as
success.

### `void` return

```cpp
auto touch = [](Payload& payload) {
    payload.value += 1;
};

auto t = yorch::task(touch)(
    yorch::borrow_prev_mut<Payload>());
```

This task only performs an action and does not produce new slot data. The flow
state is treated as success.

### `step_result` return

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

This task only decides success or failure and does not produce slot data.

### `task_result<T>` return

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

On success, the task produces `Payload` and writes it into the slot. On failure,
it only returns a failure state and does not construct `Payload`.

### Direct-output and forward-prev

Direct-output tasks already write data through `direct_out<T>`, so their return
value only needs to express flow control. If they return `void`, normal callable
completion is treated as success.

```cpp
auto emit = [](yorch::direct_out<Payload> out) -> yorch::step_result {
    out.emplace(1);
    return yorch::step_result::success();
};

auto t = yorch::task_into(emit)();
```

Forward-prev tasks get their logical output from the parent's existing slot, so
they also do not produce new data through their return value. If they need to
express failure or abort, they can return `step_result`. Returning `void` means
normal completion is success.

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

## Task Adapter

A task adapter wraps extra behavior around the original callable without
changing the callable itself. The current adapters mainly cover:

- converting exceptions into failure results;
- retrying a task when it returns `retry`.

In the `task(...)` registration APIs, one or more adapters can be passed through
`yorch::adapters(...)`.

### Catch exceptions as failure

`adapt_catch_as_failure()` catches exceptions thrown during task execution and
converts them into failure results.

YOrch's main execution path is designed as a no-exception surface. If a task may
throw, or if the callable does not provide a `noexcept` call surface, it should
first be converted into a non-throwing task through this adapter.

Without a custom policy, the default rule only fits tasks returning `void` or
`yorch::step_result`, because those task kinds can directly construct a failure
state.

```cpp
auto may_throw = []() -> yorch::step_result {
    throw std::runtime_error("boom");
    return yorch::step_result::success();
};

auto t = yorch::task(
    may_throw,
    yorch::adapters(yorch::adapt_catch_as_failure()))();
```

You can also provide a custom catch policy. The policy receives the captured
`std::exception_ptr` and must be callable with `noexcept`.

If different exception types need different handling, the policy can rethrow the
`exception_ptr` internally and catch by type, but exceptions must not escape the
policy.

```cpp
auto fallback = [](const std::exception_ptr&) noexcept {
    return yorch::step_result::abort_branch();
};

auto t = yorch::task(
    may_throw,
    yorch::adapters(yorch::adapt_catch_as_failure(fallback)))();
```

Classify by exception type:

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

If the task returns a normal value or `task_result<T>`, the custom policy must
return a result convertible to that task return type.

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

Direct-output tasks can also use catch-as-failure. For direct-output, the policy
returns `yorch::step_result`.

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

### Retry

`adapt_retry(policy)` re-executes a task when it returns `retry`. It is only
meaningful for return types that can express `retry`, such as
`yorch::step_result` and `yorch::task_result<T>`.

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

The currently built-in retry policies are:

- `retry_fixed_policy`: allow a fixed number of retries; when exhausted, convert
  the final `retry` into `failure`.

```cpp
auto t = yorch::task(
    flaky,
    yorch::adapters(yorch::adapt_retry(yorch::retry_fixed_policy {3})))();
```

- `retry_fixed_passthrough_policy`: allow a fixed number of retries; when
  exhausted, keep the final `retry` state.

```cpp
auto t = yorch::task(
    flaky,
    yorch::adapters(yorch::adapt_retry(yorch::retry_fixed_passthrough_policy {3})))();
```

- `retry_forever_policy`: keep retrying as long as the task keeps returning
  `retry`.

```cpp
auto t = yorch::task(
    flaky,
    yorch::adapters(yorch::adapt_retry(yorch::retry_forever_policy {})))();
```

Custom retry policies are also supported. A policy should provide
`should_retry(retry_count)`, where `retry_count` is the number of retries already
approved.

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

If a custom policy does not provide a rule for exhaustion, the final `retry`
state is preserved. It may additionally provide `on_exhausted(...)` to decide
the exhausted result.

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

### Combine multiple adapters

Multiple adapters can be combined. They are applied in written order: earlier
adapters are closer to the original task, and later adapters wrap around the
outside.

```cpp
auto t = yorch::task(
    flaky,
    yorch::adapters(
        yorch::adapt_catch_as_failure(),
        yorch::adapt_retry(yorch::retry_fixed_policy {2})))();
```

In this example, the original task is first wrapped by the catch adapter, and the
outer retry adapter decides whether to retry based on the result.
