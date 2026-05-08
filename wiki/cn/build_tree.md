# 构建任务树

[English](../en/build_tree.md) | [中文](build_tree.md)

YOrch 使用 `yorch::task_tree` 构建任务树。任务树从一个根节点开始，
再按层级逐个追加子节点。

最常用的写法是直接把 `callable` 写进 tree builder，再在后面的 `(...)`
里描述参数来源：

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

这里有两层括号：

- `.root(callable)` / `.node<Level>(callable)` 说明要把哪个 `callable` 放到树上
- 后面的 `(...)` 说明这个 `callable` 的参数应该从哪里来

如果 `callable` 没有参数，后面的括号也要保留，写成 `()`。

## 层级

`root(...)` 注册根任务，它的层级固定是 `0`。

`node<Level>(...)` 注册普通节点，`Level` 表示这个节点在树里的深度：

- `node<1>(...)` 是根节点的子节点
- `node<2>(...)` 是最近一个可用的第一层节点的子节点
- 同一层连续注册多个节点时，它们是 sibling

```cpp
auto tree = yorch::task_tree
    .root([]() noexcept {
        // level 0
    })()
    .node<1>([]() noexcept {
        // level 1, root 的第一个 child
    })()
    .node<2>([]() noexcept {
        // level 2, 上一个 level 1 节点的 child
    })()
    .node<1>([]() noexcept {
        // level 1, root 的第二个 child
    })();
```

构建时不能跳层。例如已经只有 `root` 时，不能直接追加 `node<2>(...)`；
需要先有 `node<1>(...)`。

空的 `task_tree` 也可以用 `node<0>(...)` 注册根节点，但通常推荐写
`root(...)`，因为语义更清楚。

## 两种注册方式

构建 tree 时，每个节点可以用两种方式注册任务。

### 直接注册 callable

这是推荐的语法糖写法。tree builder 会在内部调用对应的任务注册 API，
所以写法更短，也更适合文档和业务代码。

普通任务使用 `root(...)` 和 `node<Level>(...)`：

```cpp
auto tree = yorch::task_tree
    .root([]() noexcept -> int {
        return 3;
    })()
    .node<1>([](const int& value) noexcept -> int {
        return value + 1;
    })(yorch::borrow_prev<int>());
```

direct-output 任务使用 `root_into(...)` 和 `node_into<Level>(...)`：

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

forward-prev 任务使用 `node_forward_prev<Level>(...)`：

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

forward-prev 任务依赖 direct parent 的输出，所以实际可执行的 tree 中，
它通常作为非根节点使用。

### 注册已经绑定好的 task

也可以先用任务注册 API 得到一个已经绑定好参数的 task，再把这个 task
放进 tree。

这种写法适合任务定义和树结构分开维护，或一个 task 需要先被 adapter、
工厂函数、局部变量组合出来的场景。

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

direct-output 任务同样可以先绑定，再注册到 tree：

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

注意这里的 task 已经包含参数描述，所以 `.root(task)` / `.node<Level>(task)`
后面不再额外写参数括号。

成员函数任务也可以用同样的方式先绑定，再放进 tree。也就是说，可以先用
`task_member(...)`、`task_into_member(...)` 或
`task_forward_prev_member(...)` 得到 task object；放入 tree 时再使用普通的
`.root(task)`、`.root_into(task)`、`.node<Level>(task)` 或
`.node_into<Level>(task)`。

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

## 成员函数任务

成员函数不能直接作为普通 `callable` 传给 `root(...)` 或 `node<Level>(...)`，
因为它还需要一个 receiver。tree builder 为成员函数提供了单独的语法糖。

普通成员函数使用 `root_member(...)` 和 `node_member<Level>(...)`：

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

direct-output 成员函数使用 `root_into_member(...)` 和
`node_into_member<Level>(...)`：

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

forward-prev 成员函数使用 `node_forward_prev_member<Level>(...)`：

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

receiver 本身也是一个参数来源描述，可以来自 `value(...)`、`from_ctx<T>()`，
也可以来自 `borrow_prev<T>()`、`borrow_prev_mut<T>()`、`copy_prev<T>()`
或 `consume_prev<T>()`。

## fanout policy

当一个父节点有多个 direct child，而且这些 child 都要访问父节点输出时，
需要考虑 fanout 语义。

默认策略是 `yorch::fanout_auto_policy`。通常不用显式写出来：

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

它的具体含义是：

- 如果一个父节点最多只有一个 direct child，那么这个 child 可以使用任务本身
  支持的任意 prev 访问语义，包括 `borrow_prev<T>()`、
  `borrow_prev_mut<T>()`、`copy_prev<T>()` 和 `consume_prev<T>()`
- 如果一个父节点有多个 direct child，那么其中任何访问父节点输出的 child
  默认都只能使用共享读语义，也就是 `borrow_prev<T>()` 和 `copy_prev<T>()`
- 多 child fanout 下，默认策略不允许任何 child 使用独占语义，也就是
  `borrow_prev_mut<T>()` 或 `consume_prev<T>()`；如果需要“一个 child
  consume，其他 child 使用 copy”的语义，应显式选择下面的
  `fanout_consume_with_copies_policy`

如果希望明确表达“父节点输出会被多个 child 只读共享”，可以在注册父节点时
传入 `yorch::fanout_shared_readonly_policy`：

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

如果希望允许一个 child 消费父节点输出，同时其他 child 使用复制出来的值，
可以使用 `yorch::fanout_consume_with_copies_policy`：

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

## adapter

直接注册 `callable` 时，也可以在 tree builder 上使用 `adapters(...)`。
这和先用 `yorch::task(..., yorch::adapters(...))` 绑定再注册是等价思路，
只是写法更集中。

`adapters(...)` 写在 tree builder 的第一个括号里，也就是
`.root(...)` / `.node<Level>(...)` 这一层；后面的 `(...)` 仍然只写参数来源
spec。

对于普通 callable，位置是：

```cpp
.root(callable, adapters(...))(spec...)
.node<Level>(callable, adapters(...))(spec...)
```

如果同一个节点还要显式写 `fanout policy`，顺序是：

```cpp
.root(callable, fanout_policy, adapters(...))(spec...)
.node<Level>(callable, fanout_policy, adapters(...))(spec...)
```

成员函数多一个 receiver 参数，所以 `adapters(...)` 位于 receiver 后面：

```cpp
.root_member(member_ptr, receiver_spec, adapters(...))(spec...)
.node_member<Level>(member_ptr, receiver_spec, adapters(...))(spec...)
```

如果成员函数也同时指定 `fanout policy`，顺序是：

```cpp
.root_member(member_ptr, receiver_spec, fanout_policy, adapters(...))(spec...)
.node_member<Level>(member_ptr, receiver_spec, fanout_policy, adapters(...))(spec...)
```

`root_into(...)`、`node_into<Level>(...)`、`root_into_member(...)`、
`node_into_member<Level>(...)` 也使用同样的参数位置规则。

```cpp
auto tree = yorch::task_tree
    .root(
        []() -> yorch::step_result {
            throw std::runtime_error("boom");
        },
        yorch::adapters(yorch::adapt_catch_as_failure()))();
```

`adapters(...)` 接收的是具体 adapter 描述。adapter 自己需要的 policy
写在 adapter 工厂函数里面，而不是作为 tree builder 的独立参数。

例如 catch adapter 有两种写法：

- `yorch::adapt_catch_as_failure()` 使用默认 catch policy
- `yorch::adapt_catch_as_failure(policy)` 使用自定义 catch policy

自定义 catch policy 接收 `std::exception_ptr`，并且必须是 `noexcept` 调用。
policy 的返回值要能转换成任务的返回类型；对于 direct-output 任务，返回
`yorch::step_result`。

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

retry adapter 的 policy 也写在 adapter 工厂函数里，例如：

```cpp
auto tree = yorch::task_tree
    .root(
        []() noexcept -> yorch::step_result {
            return yorch::step_result::retry();
        },
        yorch::adapters(yorch::adapt_retry(yorch::retry_fixed_policy {3})))();
```

带 `fanout policy` 和 `adapter` 时，`fanout policy` 是 tree 节点自己的参数，
`retry_fixed_policy` 这类 policy 是 adapter 自己的参数。两者位置不同：

```cpp
auto tree = yorch::task_tree
    .root(
        []() noexcept -> yorch::step_result {
            return yorch::step_result::success();
        },
        yorch::fanout_shared_readonly_policy {},
        yorch::adapters(yorch::adapt_retry(yorch::retry_fixed_policy {3})))();
```

多个 adapter 可以写在同一个 `adapters(...)` 里。它们按书写顺序依次应用：
前面的 adapter 更靠近原始任务，后面的 adapter 包在更外层。

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

如果已经先注册成绑定好的 task object，再放入 tree，那么 adapter 应该在
`yorch::task(...)`、`yorch::task_into(...)` 或成员函数对应的 task 注册 API
里提供；放入 tree 的 `.root(task)` / `.node<Level>(task)` 不再额外接收
`adapters(...)`。

## 编译和执行

任务树本身只是静态结构。要执行它，需要先编译成 plan，再运行：

```cpp
auto plan = yorch::compile_plan(tree);
const auto result = yorch::run_plan(plan);

if (!result.ok()) {
    // handle failure
}
```

如果任务使用了 `from_ctx<T>()`，执行时把 context 传给 `run_plan`：

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

## 小结

- 推荐直接在 tree builder 上注册 `callable`，例如 `.root(f)(...)`
  和 `.node<1>(f)(...)`
- 如果已经有绑定好的 task，可以用 `.root(task)` 和 `.node<Level>(task)`
  直接放进 tree
- `root_into(...)` / `node_into<Level>(...)` 用于 direct-output 任务
- `node_forward_prev<Level>(...)` 用于 forward-prev 任务
- 成员函数使用 `root_member(...)`、`node_member<Level>(...)`
  以及对应的 `into` / `forward_prev` 版本
