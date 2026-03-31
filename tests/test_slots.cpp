#include "yorch/slots.hpp"

#include "yorch/bind.hpp"
#include "yorch/resolve.hpp"
#include "yorch/plan.hpp"

#include <gtest/gtest.h>

#include <string>
#include <type_traits>
#include <utility>

namespace {

struct move_only {
    int value = 0;

    move_only() = default;
    explicit move_only(int in) : value(in) {}

    move_only(const move_only&) = delete;
    move_only& operator=(const move_only&) = delete;
    move_only(move_only&&) noexcept = default;
    move_only& operator=(move_only&&) noexcept = default;

    ~move_only() = default;
};

struct lifetime_tracker {
    int live_count = 0;
    int destroyed_count = 0;
};

struct lifetime_probe {
    lifetime_tracker* tracker = nullptr;

    int value = 0;

    explicit lifetime_probe(lifetime_tracker& tracker_ref)
        : tracker(&tracker_ref) {
        ++tracker->live_count;
    }

    lifetime_probe(lifetime_tracker& tracker_ref, int in)
        : tracker(&tracker_ref),
          value(in) {
        ++tracker->live_count;
    }

    lifetime_probe(const lifetime_probe& other)
        : tracker(other.tracker),
          value(other.value) {
        ++tracker->live_count;
    }

    lifetime_probe(lifetime_probe&& other) noexcept
        : tracker(other.tracker),
          value(other.value) {
        ++tracker->live_count;
        other.value = -1;
    }

    lifetime_probe& operator=(const lifetime_probe&) = default;
    lifetime_probe& operator=(lifetime_probe&&) noexcept = default;

    ~lifetime_probe() {
        --tracker->live_count;
        ++tracker->destroyed_count;
    }
};

template <typename Slots, std::size_t I, typename... Args>
concept can_emplace_slot =
    requires(Slots& slots, Args&&... args) {
        slots.template emplace<I>(std::forward<Args>(args)...);
    };

constexpr auto make_mixed_plan() {
    auto tree = yorch::task_tree.root(yorch::bind([]() noexcept -> int {
            return 1;
        }))
        .node<1>(yorch::bind([]() noexcept -> yorch::task_result<std::string> {
            return yorch::task_result<std::string>::success("child");
        }))
            .node<2>(yorch::bind([]() noexcept -> yorch::step_result {
                return yorch::step_result::retry();
            }))
        .node<1>(yorch::bind([]() noexcept {}))
            .node<2>(yorch::bind([]() noexcept -> bool {
                return true;
            }));

    return yorch::compile_plan(tree);
}

constexpr auto make_move_only_plan() {
    auto tree = yorch::task_tree.root(yorch::bind([]() noexcept -> move_only {
        return move_only {9};
    }));

    return yorch::compile_plan(tree);
}

auto make_probe_plan(lifetime_tracker& tracker) {
    auto tree = yorch::task_tree.root(yorch::bind([&tracker]() noexcept -> lifetime_probe {
        return lifetime_probe {tracker, 7};
    }));

    return yorch::compile_plan(tree);
}

} // namespace

TEST(SlotsTest, TypedSlotTracksLifecycleAndAutoDestroysLivePayload) {
    lifetime_tracker tracker;

    {
        yorch::detail::typed_slot<lifetime_probe> slot;

        EXPECT_FALSE(slot.has_value());

        auto& probe = slot.emplace(tracker, 11);

        EXPECT_TRUE(slot.has_value());
        EXPECT_EQ(probe.value, 11);
        EXPECT_EQ(tracker.live_count, 1);
        EXPECT_EQ(tracker.destroyed_count, 0);

        slot.destroy();

        EXPECT_FALSE(slot.has_value());
        EXPECT_EQ(tracker.live_count, 0);
        EXPECT_EQ(tracker.destroyed_count, 1);

        slot.emplace(tracker, 22);
        EXPECT_TRUE(slot.has_value());
        EXPECT_EQ(probe.value, 22);
        EXPECT_EQ(tracker.live_count, 1);
    }

    EXPECT_EQ(tracker.live_count, 0);
    EXPECT_EQ(tracker.destroyed_count, 2);
}

TEST(SlotsTest, PlanSlotsExposePerNodeOutputTypesAndLifecycle) {
    auto plan = make_mixed_plan();
    using plan_t = decltype(plan);
    using slots_t = yorch::plan_slots<plan_t>;

    static_assert(std::is_same_v<typename slots_t::template output_type<0>, int>);
    static_assert(std::is_same_v<typename slots_t::template output_type<1>, std::string>);
    static_assert(std::is_same_v<typename slots_t::template output_type<2>, void>);
    static_assert(std::is_same_v<typename slots_t::template output_type<3>, void>);
    static_assert(std::is_same_v<typename slots_t::template output_type<4>, bool>);
    static_assert(can_emplace_slot<slots_t, 0, int>);
    static_assert(can_emplace_slot<slots_t, 1, std::string>);
    static_assert(!can_emplace_slot<slots_t, 2>);
    static_assert(!can_emplace_slot<slots_t, 3>);
    static_assert(can_emplace_slot<slots_t, 4, bool>);

    slots_t slots;

    EXPECT_FALSE(slots.has_value<0>());
    EXPECT_FALSE(slots.has_value<1>());
    EXPECT_FALSE(slots.has_value<2>());
    EXPECT_FALSE(slots.has_value<3>());
    EXPECT_FALSE(slots.has_value<4>());

    auto& root = slots.emplace<0>(41);
    auto& child = slots.emplace<1>(std::string("leaf"));
    auto& tail = slots.emplace<4>(true);

    EXPECT_TRUE(slots.has_value<0>());
    EXPECT_TRUE(slots.has_value<1>());
    EXPECT_FALSE(slots.has_value<2>());
    EXPECT_FALSE(slots.has_value<3>());
    EXPECT_TRUE(slots.has_value<4>());
    EXPECT_EQ(&slots.get<0>(), &root);
    EXPECT_EQ(&slots.get<1>(), &child);
    EXPECT_EQ(&slots.get<4>(), &tail);
    EXPECT_EQ(slots.get<0>(), 41);
    EXPECT_EQ(slots.get<1>(), "leaf");
    EXPECT_TRUE(slots.get<4>());

    slots.destroy<1>();
    slots.destroy<2>();

    EXPECT_FALSE(slots.has_value<1>());
    EXPECT_FALSE(slots.has_value<2>());
    EXPECT_TRUE(slots.has_value<0>());
    EXPECT_TRUE(slots.has_value<4>());
}

TEST(SlotsTest, PlanSlotsPrevViewForReturnsDirectParentViewOrNoPrev) {
    auto plan = make_mixed_plan();
    using plan_t = decltype(plan);
    using slots_t = yorch::plan_slots<plan_t>;

    static_assert(std::is_same_v<
                  decltype(std::declval<slots_t&>().template prev_view_for<0>()),
                  yorch::prev_slot_view<int>>);
    static_assert(std::is_same_v<
                  decltype(std::declval<const slots_t&>().template prev_view_for<0>()),
                  yorch::prev_slot_view<const int>>);
    static_assert(std::is_same_v<
                  decltype(std::declval<slots_t&>().template prev_view_for<2>()),
                  yorch::no_prev>);

    slots_t slots;
    auto& parent = slots.emplace<0>(7);

    auto prev = slots.prev_view_for<0>();
    yorch::exec_context<void, decltype(prev)> exec {prev};

    auto&& resolved_ref = yorch::resolve_as<int&>(yorch::from_prev<int>(), exec);
    auto copied = yorch::resolve_as<long>(yorch::from_prev<int>(), exec);

    EXPECT_EQ(&resolved_ref, &parent);
    EXPECT_EQ(copied, 7);

    const auto& const_slots = slots;
    auto const_prev = const_slots.prev_view_for<0>();
    yorch::exec_context<void, decltype(const_prev)> const_exec {const_prev};
    auto&& const_ref = yorch::resolve_as<const int&>(yorch::from_prev<int>(), const_exec);

    EXPECT_EQ(&const_ref, &parent);

    auto empty_prev = slots.prev_view_for<2>();
    static_assert(std::is_same_v<decltype(empty_prev), yorch::no_prev>);
    EXPECT_FALSE(decltype(empty_prev)::template contains<int>());
}

TEST(SlotsTest, PlanSlotsSupportMoveOnlyPayloads) {
    auto plan = make_move_only_plan();
    using slots_t = yorch::plan_slots<decltype(plan)>;

    slots_t slots;
    auto& stored = slots.emplace<0>(move_only {9});

    static_assert(std::is_same_v<decltype(stored), move_only&>);

    EXPECT_TRUE(slots.has_value<0>());
    EXPECT_EQ(stored.value, 9);
    EXPECT_EQ(slots.get<0>().value, 9);
}

TEST(SlotsTest, PlanSlotsDestructorDestroysLivePayloads) {
    lifetime_tracker tracker;

    {
        auto plan = make_probe_plan(tracker);
        yorch::plan_slots<decltype(plan)> slots;

        slots.emplace<0>(lifetime_probe {tracker, 5});

        EXPECT_TRUE(slots.has_value<0>());
        EXPECT_EQ(tracker.live_count, 1);
        // The temporary passed into emplace(...) has already been destroyed;
        // the slot-owned payload remains live here.
        EXPECT_EQ(tracker.destroyed_count, 1);
    }

    EXPECT_EQ(tracker.live_count, 0);
    EXPECT_EQ(tracker.destroyed_count, 2);
}
