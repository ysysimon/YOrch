# Build a Task Tree

<a href="build_tree.md">English</a> | <a href="../cn/build_tree.md">中文</a>

YOrch uses `yorch::task_tree` to build a task tree. A task tree starts from one
root node and then appends child nodes level by level.

The most common style is to pass a `callable` directly into the tree builder and
describe its argument sources in the following `(...)`:

```cpp
auto tree = yorch::task_tree
    .root([]() noexcept -> int {
        return 7;
    })()
    .node<1>([](const int& value) noexcept {
        return value * 2;
    })(yorch::borrow_prev<int>())
    .node<2>([](const int& value) noexcept {
        return value + 1;
    })(yorch::borrow_prev<int>());
```

There are two sets of parentheses:

- `.root(callable)` / `.node<Level>(callable)` selects the `callable` placed on
  the tree.
- The following `(...)` describes where that `callable` gets its arguments.

If the `callable` has no arguments, the second set of parentheses is still
required and should be written as `()`.

## Levels

`root(...)` registers the root task. Its level is always `0`.

`node<Level>(...)` registers a normal node. `Level` is the node depth inside the
tree:

- `node<1>(...)` is a child of the root node.
- `node<2>(...)` is a child of the nearest available first-level node.
- Consecutive nodes on the same level are siblings.

```cpp
auto tree = yorch::task_tree
    .root([]() noexcept {
        // level 0
    })()
    .node<1>([]() noexcept {
        // level 1, first child of root
    })()
    .node<2>([]() noexcept {
        // level 2, child of the previous level 1 node
    })()
    .node<1>([]() noexcept {
        // level 1, second child of root
    })();
```

You cannot skip levels while building a tree. For example, when the tree only has
the root node, you cannot append `node<2>(...)` directly; a `node<1>(...)` must
exist first.

An empty `task_tree` can also register the root with `node<0>(...)`, but
`root(...)` is usually preferred because it is clearer.

## Two Registration Styles

Each node can register a task in two ways.

### Register a callable directly

This is the recommended shorthand. The tree builder internally calls the
corresponding task registration API, so the syntax is shorter and works well in
documentation and business code.

Use `root(...)` and `node<Level>(...)` for regular tasks:

```cpp
auto tree = yorch::task_tree
    .root([]() noexcept -> int {
        return 3;
    })()
    .node<1>([](const int& value) noexcept -> int {
        return value + 1;
    })(yorch::borrow_prev<int>());
```

Use `root_into(...)` and `node_into<Level>(...)` for direct-output tasks:

```cpp
auto tree = yorch::task_tree
    .root_into([](yorch::direct_out<std::string> out) noexcept -> yorch::step_result {
        return out.success("hello");
    })()
    .node_into<1>(
        [](const std::string& text,
           yorch::direct_out<int> out) noexcept -> yorch::step_result {
            return out.success(static_cast<int>(text.size()));
        })(yorch::borrow_prev<std::string>());
```

Use `node_forward_prev<Level>(...)` for forward-prev tasks:

```cpp
auto tree = yorch::task_tree
    .root([]() noexcept -> int {
        return 5;
    })()
    .node_forward_prev<1>(
        [](int& value) noexcept -> yorch::step_result {
            value += 4;
            return yorch::step_result::success();
        })(yorch::borrow_prev_mut<int>());
```

A forward-prev task depends on the direct parent's output, so in an executable
tree it is normally used as a non-root node.

### Register an already-bound task

You can also first create a task with the task registration API, bind its
arguments, and then place that task into the tree.

This style is useful when task definitions and tree structure are maintained
separately, or when a task must first be composed through adapters, factory
functions, or local variables.

```cpp
auto make_value = yorch::task([]() noexcept -> int {
    return 3;
})();

auto add_one = yorch::task([](const int& value) noexcept -> int {
    return value + 1;
})(yorch::borrow_prev<int>());

auto tree = yorch::task_tree
    .root(make_value)
    .node<1>(add_one);
```

Direct-output tasks can be bound first as well:

```cpp
auto emit_text = yorch::task_into(
    [](yorch::direct_out<std::string> out) noexcept -> yorch::step_result {
        return out.success("hello");
    })();

auto read_text = yorch::task([](const std::string& text) noexcept -> int {
    return static_cast<int>(text.size());
})(yorch::borrow_prev<std::string>());

auto tree = yorch::task_tree
    .root_into(emit_text)
    .node<1>(read_text);
```

The task already contains its argument descriptions, so `.root(task)` /
`.node<Level>(task)` do not take an extra argument list.

Member function tasks can use the same pattern. You may first create a task
object with `task_member(...)`, `task_into_member(...)`, or
`task_forward_prev_member(...)`, then place it into the tree with the normal
`.root(task)`, `.root_into(task)`, `.node<Level>(task)`, or
`.node_into<Level>(task)` APIs.

```cpp
struct Worker {
    int seed(int delta) noexcept {
        return delta * 2;
    }

    int add(const int& value, int delta) noexcept {
        return value + delta;
    }
};

Worker worker;

auto seed = yorch::task_member(
    &Worker::seed,
    yorch::value(std::ref(worker)))(
    yorch::value(3));

auto add_one = yorch::task_member(
    &Worker::add,
    yorch::value(std::ref(worker)))(
    yorch::borrow_prev<int>(),
    yorch::value(1));

auto tree = yorch::task_tree
    .root(seed)
    .node<1>(add_one);
```

## Member Function Tasks

A member function cannot be passed directly as a regular `callable` to
`root(...)` or `node<Level>(...)`, because it also needs a receiver. The tree
builder provides separate shorthand APIs for member functions.

Use `root_member(...)` and `node_member<Level>(...)` for regular member
functions:

```cpp
struct Worker {
    int base = 0;

    int seed(int delta) noexcept {
        base += delta;
        return base;
    }
};

Worker worker;

auto tree = yorch::task_tree
    .root_member(
        &Worker::seed,
        yorch::value(std::ref(worker)))(
        yorch::value(3));
```

Use `root_into_member(...)` and `node_into_member<Level>(...)` for direct-output
member functions:

```cpp
struct Worker {
    int base = 0;

    yorch::step_result emit_text(
        int delta,
        yorch::direct_out<std::string> out) noexcept {
        base += delta;
        return out.success(std::to_string(base));
    }
};

Worker worker;

auto tree = yorch::task_tree
    .root_into_member(
        &Worker::emit_text,
        yorch::value(std::ref(worker)))(
        yorch::value(3));
```

Use `node_forward_prev_member<Level>(...)` for forward-prev member functions:

```cpp
struct Payload {
    int value = 0;
};

struct Service {
    yorch::step_result bump(Payload& payload, int delta) noexcept {
        payload.value += delta;
        return yorch::step_result::success();
    }
};

auto tree = yorch::task_tree
    .root([]() noexcept -> Payload {
        return Payload {6};
    })()
    .node_forward_prev_member<1>(
        &Service::bump,
        yorch::from_ctx<Service>())(
        yorch::borrow_prev_mut<Payload>(),
        yorch::value(5));
```

The receiver is also an argument source description. It can come from
`value(...)`, `from_ctx<T>()`, `borrow_prev<T>()`, `borrow_prev_mut<T>()`,
`copy_prev<T>()`, or `consume_prev<T>()`.

## Fanout Policy

When one parent node has multiple direct children and those children access the
parent output, the fanout semantics must be considered.

The default policy is `yorch::fanout_auto_policy`. Usually it does not need to be
written explicitly:

```cpp
auto tree = yorch::task_tree
    .root([]() noexcept -> int {
        return 10;
    })()
    .node<1>([](const int& value) noexcept {
        return value + 1;
    })(yorch::borrow_prev<int>())
    .node<1>([](int value) noexcept {
        return value + 2;
    })(yorch::copy_prev<int>());
```

Its meaning is:

- If a parent has at most one direct child, that child may use any prev access
  semantic supported by the task, including `borrow_prev<T>()`,
  `borrow_prev_mut<T>()`, `copy_prev<T>()`, and `consume_prev<T>()`.
- If a parent has multiple direct children, any child that accesses the parent
  output can only use shared-read semantics by default: `borrow_prev<T>()` and
  `copy_prev<T>()`.
- In multi-child fanout, the default policy does not allow any child to use
  exclusive semantics: `borrow_prev_mut<T>()` or `consume_prev<T>()`. If one
  child should consume the parent output while other children use copies, select
  `fanout_consume_with_copies_policy` explicitly.

Use `yorch::fanout_shared_readonly_policy` when you want to explicitly state
that the parent output is shared read-only by multiple children:

```cpp
auto tree = yorch::task_tree
    .root(
        []() noexcept -> int {
            return 10;
        },
        yorch::fanout_shared_readonly_policy {})()
    .node<1>([](const int& value) noexcept {
        return value + 1;
    })(yorch::borrow_prev<int>())
    .node<1>([](int value) noexcept {
        return value + 2;
    })(yorch::copy_prev<int>());
```

Use `yorch::fanout_consume_with_copies_policy` when one child may consume the
parent output while other children use copied values:

```cpp
auto tree = yorch::task_tree
    .root(
        []() noexcept -> std::string {
            return "payload";
        },
        yorch::fanout_consume_with_copies_policy {})()
    .node<1>([](std::string text) noexcept {
        return text.size();
    })(yorch::copy_prev<std::string>())
    .node<1>([](std::string text) noexcept {
        return text.empty();
    })(yorch::consume_prev<std::string>());
```

## Adapter

When registering a `callable` directly, `adapters(...)` can also be used on the
tree builder. This is equivalent in spirit to first binding with
`yorch::task(..., yorch::adapters(...))` and then registering the bound task, but
the syntax is more centralized.

`adapters(...)` is placed in the first set of tree-builder parentheses, the
`.root(...)` / `.node<Level>(...)` layer. The following `(...)` still contains
only argument source specs.

For regular callables:

```cpp
.root(callable, adapters(...))(spec...)
.node<Level>(callable, adapters(...))(spec...)
```

If the same node also specifies a fanout policy, the order is:

```cpp
.root(callable, fanout_policy, adapters(...))(spec...)
.node<Level>(callable, fanout_policy, adapters(...))(spec...)
```

Member functions have one extra receiver argument, so `adapters(...)` is placed
after the receiver:

```cpp
.root_member(member_ptr, receiver_spec, adapters(...))(spec...)
.node_member<Level>(member_ptr, receiver_spec, adapters(...))(spec...)
```

If a member function also specifies a fanout policy, the order is:

```cpp
.root_member(member_ptr, receiver_spec, fanout_policy, adapters(...))(spec...)
.node_member<Level>(member_ptr, receiver_spec, fanout_policy, adapters(...))(spec...)
```

`root_into(...)`, `node_into<Level>(...)`, `root_into_member(...)`, and
`node_into_member<Level>(...)` follow the same parameter placement rules.

```cpp
auto tree = yorch::task_tree
    .root(
        []() -> yorch::step_result {
            throw std::runtime_error("boom");
        },
        yorch::adapters(yorch::adapt_catch_as_failure()))();
```

`adapters(...)` receives concrete adapter descriptions. A policy needed by an
adapter is passed to the adapter factory function, not as an independent
tree-builder argument.

For example, the catch adapter has two forms:

- `yorch::adapt_catch_as_failure()` uses the default catch policy.
- `yorch::adapt_catch_as_failure(policy)` uses a custom catch policy.

The custom catch policy receives `std::exception_ptr`, must be callable with
`noexcept`, and must return a value convertible to the task return type. For
direct-output tasks, it returns `yorch::step_result`.

```cpp
auto fallback = [](const std::exception_ptr&) noexcept {
    return yorch::step_result::failure();
};

auto tree = yorch::task_tree
    .root(
        []() -> yorch::step_result {
            throw std::runtime_error("boom");
        },
        yorch::adapters(yorch::adapt_catch_as_failure(fallback)))();
```

The retry adapter policy is also passed to the adapter factory:

```cpp
auto tree = yorch::task_tree
    .root(
        []() noexcept -> yorch::step_result {
            return yorch::step_result::retry();
        },
        yorch::adapters(yorch::adapt_retry(yorch::retry_fixed_policy {3})))();
```

When both fanout policy and adapter are present, the fanout policy is a parameter
of the tree node, while policies such as `retry_fixed_policy` are parameters of
the adapter itself:

```cpp
auto tree = yorch::task_tree
    .root(
        []() noexcept -> yorch::step_result {
            return yorch::step_result::success();
        },
        yorch::fanout_shared_readonly_policy {},
        yorch::adapters(yorch::adapt_retry(yorch::retry_fixed_policy {3})))();
```

Multiple adapters can be placed inside the same `adapters(...)`. They are applied
in written order: earlier adapters are closer to the original task, and later
adapters wrap around the outside.

```cpp
auto tree = yorch::task_tree
    .root(
        []() -> yorch::step_result {
            throw std::runtime_error("boom");
        },
        yorch::adapters(
            yorch::adapt_catch_as_failure(),
            yorch::adapt_retry(yorch::retry_fixed_policy {2})))();
```

If a task object is already bound and then placed into the tree, adapters should
be provided to `yorch::task(...)`, `yorch::task_into(...)`, or the corresponding
member-function task registration API. `.root(task)` / `.node<Level>(task)` do
not accept an additional `adapters(...)` argument.

## Compile and Run

A task tree is only a static structure. To execute it, compile it into a plan and
then run the plan:

```cpp
auto plan = yorch::compile_plan(tree);
const auto result = yorch::run_plan(plan);

if (!result.ok()) {
    // handle failure
}
```

If a task uses `from_ctx<T>()`, pass the context to `run_plan`:

```cpp
struct Config {
    int base = 0;
};

auto tree = yorch::task_tree
    .root([](const Config& config) noexcept {
        return config.base + 1;
    })(yorch::from_ctx<Config>());

yorch::context<Config> ctx(Config {.base = 41});

auto plan = yorch::compile_plan(tree);
const auto result = yorch::run_plan(plan, ctx);
```

## Summary

- Prefer registering callables directly on the tree builder, such as
  `.root(f)(...)` and `.node<1>(f)(...)`.
- If you already have a bound task, place it into the tree with `.root(task)` and
  `.node<Level>(task)`.
- Use `root_into(...)` / `node_into<Level>(...)` for direct-output tasks.
- Use `node_forward_prev<Level>(...)` for forward-prev tasks.
- Use `root_member(...)`, `node_member<Level>(...)`, and their `into` /
  `forward_prev` variants for member functions.
