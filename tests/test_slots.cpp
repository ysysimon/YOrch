#include "yorch/slots.hpp"

#include "yorch/bind.hpp"
#include "yorch/resolve.hpp"
#include "yorch/plan.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <tuple>
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

template <typename Slots, std::size_t I>
concept can_query_slot_value =
    requires(const Slots& slots) {
        { slots.template has_value<I>() } -> std::convertible_to<bool>;
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

auto make_probe_result_plan(lifetime_tracker& tracker) {
    auto tree = yorch::task_tree.root(yorch::bind([&tracker]() noexcept -> yorch::task_result<lifetime_probe> {
        return yorch::task_result<lifetime_probe>::success(lifetime_probe {tracker, 7});
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
    static_assert(slots_t::template slot_logical_policy<0> == yorch::detail::slot_logical_policy::must_payload);
    static_assert(slots_t::template slot_logical_policy<1> == yorch::detail::slot_logical_policy::maybe_payload);
    static_assert(slots_t::template slot_logical_policy<2> == yorch::detail::slot_logical_policy::none);
    static_assert(slots_t::template slot_logical_policy<3> == yorch::detail::slot_logical_policy::none);
    static_assert(slots_t::template slot_logical_policy<4> == yorch::detail::slot_logical_policy::must_payload);
    static_assert(std::tuple_size_v<typename slots_t::tuple_type> == slots_t::physical_slot_count);
    static_assert(std::is_same_v<
                  std::tuple_element_t<0, typename slots_t::tuple_type>,
                  yorch::detail::typed_slot<int, yorch::detail::slot_logical_policy::must_payload>>);
    static_assert(std::is_same_v<
                  std::tuple_element_t<1, typename slots_t::tuple_type>,
                  yorch::detail::typed_slot<std::string, yorch::detail::slot_logical_policy::maybe_payload>>);
    static_assert(std::is_same_v<
                  std::tuple_element_t<2, typename slots_t::tuple_type>,
                  yorch::detail::typed_slot<bool, yorch::detail::slot_logical_policy::must_payload>>);
    static_assert(can_emplace_slot<slots_t, 0, int>);
    static_assert(can_emplace_slot<slots_t, 1, std::string>);
    static_assert(!can_query_slot_value<slots_t, 0>);
    static_assert(can_query_slot_value<slots_t, 1>);
    static_assert(can_query_slot_value<slots_t, 2>);
    static_assert(can_query_slot_value<slots_t, 3>);
    static_assert(!can_query_slot_value<slots_t, 4>);
    static_assert(!can_emplace_slot<slots_t, 2>);
    static_assert(!can_emplace_slot<slots_t, 3>);
    static_assert(can_emplace_slot<slots_t, 4, bool>);

    slots_t slots;

    EXPECT_FALSE(slots.has_value<1>());
    EXPECT_FALSE(slots.has_value<2>());
    EXPECT_FALSE(slots.has_value<3>());

    auto& root = slots.emplace<0>(41);
    auto& child = slots.emplace<1>(std::string("leaf"));
    auto& tail = slots.emplace<4>(true);

    EXPECT_TRUE(slots.has_value<1>());
    EXPECT_FALSE(slots.has_value<2>());
    EXPECT_FALSE(slots.has_value<3>());
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
}

TEST(SlotsTest, CompactLayoutInfersPhysicalSlotReuseAndPolicies) {
    auto plan = make_mixed_plan();
    using plan_t = decltype(plan);
    using compact_slots_t =
        yorch::plan_exec_slots<plan_t, yorch::slot_layout_serial_dfs_compact_policy>;

    static_assert(compact_slots_t::physical_slot_count == 2);
    static_assert(compact_slots_t::template physical_slot_index<0> == 0);
    static_assert(compact_slots_t::template physical_slot_index<1> == 1);
    static_assert(compact_slots_t::template physical_slot_index<4> == 1);
    static_assert(compact_slots_t::template slot_physical_policy<0> ==
                  yorch::detail::slot_physical_policy::must_payload);
    static_assert(compact_slots_t::template slot_physical_policy<1> ==
                  yorch::detail::slot_physical_policy::maybe_payload);
    static_assert(std::is_same_v<
                  std::tuple_element_t<0, typename compact_slots_t::tuple_type>,
                  yorch::detail::erased_slot<
                      sizeof(int),
                      alignof(int),
                      yorch::detail::slot_physical_policy::must_payload>>);

    compact_slots_t slots;
    auto& root = slots.emplace<0>(41);
    auto& child = slots.emplace<1>(std::string("leaf"));

    EXPECT_EQ(&slots.get<0>(), &root);
    EXPECT_EQ(&slots.get<1>(), &child);
    EXPECT_EQ(slots.get<0>(), 41);
    EXPECT_EQ(slots.get<1>(), "leaf");

    const auto reused_slot_addr =
        reinterpret_cast<std::uintptr_t>(static_cast<const void*>(&slots.get<1>()));

    slots.destroy<1>();
    auto& tail = slots.emplace<4>(true);

    EXPECT_EQ(reused_slot_addr,
              reinterpret_cast<std::uintptr_t>(static_cast<const void*>(&tail)));
    EXPECT_TRUE(slots.get<4>());
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

    EXPECT_EQ(stored.value, 9);
    EXPECT_EQ(slots.get<0>().value, 9);
}

TEST(SlotsTest, PlanSlotsDestructorDestroysLivePayloads) {
    lifetime_tracker must_tracker;

    {
        auto plan = make_probe_plan(must_tracker);
        yorch::plan_slots<decltype(plan)> slots;

        slots.emplace<0>(lifetime_probe {must_tracker, 5});

        EXPECT_EQ(must_tracker.live_count, 1);
        // The temporary passed into emplace(...) has already been destroyed;
        // the slot-owned payload remains live here.
        EXPECT_EQ(must_tracker.destroyed_count, 1);
    }

    EXPECT_EQ(must_tracker.live_count, 0);
    EXPECT_EQ(must_tracker.destroyed_count, 2);

    lifetime_tracker maybe_tracker;

    {
        auto plan = make_probe_result_plan(maybe_tracker);
        yorch::plan_slots<decltype(plan)> slots;

        slots.emplace<0>(lifetime_probe {maybe_tracker, 5});

        EXPECT_TRUE(slots.has_value<0>());
        EXPECT_EQ(maybe_tracker.live_count, 1);
        // The temporary passed into emplace(...) has already been destroyed;
        // the slot-owned payload remains live here.
        EXPECT_EQ(maybe_tracker.destroyed_count, 1);
    }

    EXPECT_EQ(maybe_tracker.live_count, 0);
    EXPECT_EQ(maybe_tracker.destroyed_count, 2);
}
