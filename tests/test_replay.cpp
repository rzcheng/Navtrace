#include "navtrace/io.hpp"
#include <gtest/gtest.h>
#include <limits>

namespace navtrace {
namespace {
TEST(Replay, EqualTimestampPredictsOnlyOnce) {
  State s = State::Zero();
  s(SPEED) = 2.0;
  Replay replay(s, Covariance::Zero(), Config{}, Method::Ekf);
  ASSERT_TRUE(replay.process({10000, GyroMeasurement{0.0, 0.01}}).valid);
  ASSERT_TRUE(replay.process({10000, WheelSpeedMeasurement{2.0, 0.1}}).valid);
  // Independent one-step result: pxx stays zero; pvv=qv*h then scalar update.
  EXPECT_NEAR(replay.state()(PX), 0.02, 1e-14);
  EXPECT_DOUBLE_EQ(replay.state()(SPEED), 2.0);
  EXPECT_DOUBLE_EQ(replay.covariance()(PX, PX), 0.0);
  EXPECT_NEAR(replay.covariance()(SPEED, SPEED), 0.0025 * 0.1 / 0.1025, 1e-14);
}
TEST(Replay, InvalidDuplicateOutOfOrderAndLargeGapPreserveStateAndTime) {
  Replay replay(State::Zero(), Covariance::Identity(), Config{}, Method::Ekf);
  ASSERT_TRUE(replay.process({10000, GyroMeasurement{0.1, 0.01}}).valid);
  const State state = replay.state();
  const Covariance p = replay.covariance();
  EXPECT_FALSE(replay.process({10000, GyroMeasurement{0.1, 0.01}}).valid);
  EXPECT_FALSE(replay.process({9000, WheelSpeedMeasurement{1.0, 0.1}}).valid);
  EXPECT_FALSE(replay.process({-1, GpsMeasurement{}}).valid);
  EXPECT_FALSE(replay.process({20000, WheelSpeedMeasurement{1.0, -0.1}}).valid);
  EXPECT_FALSE(replay.process({20000, GyroMeasurement{std::numeric_limits<double>::infinity(), 0.1}}).valid);
  EXPECT_FALSE(replay.process({3000000, GpsMeasurement{}}).valid);
  EXPECT_EQ(replay.timestamp_us(), 10000);
  EXPECT_EQ(replay.state(), state);
  EXPECT_EQ(replay.covariance(), p);
  EXPECT_TRUE(replay.process({10000, WheelSpeedMeasurement{1.0, 0.1}}).valid);
}
TEST(Replay, WrongSimultaneousOrderRejected) {
  Replay replay(State::Zero(), Covariance::Identity(), Config{}, Method::Ekf);
  ASSERT_TRUE(replay.process({0, GpsMeasurement{}}).valid);
  EXPECT_FALSE(replay.process({0, WheelSpeedMeasurement{1.0, 0.1}}).valid);
  EXPECT_EQ(replay.accepted_gps(), 1U);
}
TEST(Replay, RejectedGpsKeepsPredictionAtNewTimestamp) {
  State s = State::Zero();
  s(SPEED) = 2.0;
  Replay replay(s, Covariance::Identity(), Config{}, Method::GatedEkf);
  Ekf predicted(s, Covariance::Identity());
  GpsMeasurement gps;
  gps.position_m << 1000.0, 1000.0;
  const auto result = replay.process({100000, gps});
  ASSERT_TRUE(result.valid);
  ASSERT_EQ(result.update.status, UpdateStatus::Rejected);
  ASSERT_TRUE(predicted.predict(0.1).ok);
  EXPECT_TRUE(replay.state().isApprox(predicted.state(), 1e-14));
  EXPECT_TRUE(replay.covariance().isApprox(predicted.covariance(), 1e-14));
  EXPECT_EQ(replay.timestamp_us(), 100000);
  EXPECT_EQ(replay.rejected_gps(), 1U);
}
TEST(Replay, OutageHasNoFabricatedGpsAndReplayIsDeterministic) {
  Replay first(State::Zero(), Covariance::Identity(), Config{}, Method::Ekf);
  Replay second = first;
  ASSERT_TRUE(first.process({0, GpsMeasurement{}}).valid);
  ASSERT_TRUE(second.process({0, GpsMeasurement{}}).valid);
  for (int i = 1; i <= 300; ++i) {
    const SensorEvent gyro{10000LL * i, GyroMeasurement{0.02, 0.001}};
    const SensorEvent wheel{10000LL * i, WheelSpeedMeasurement{1.0, 0.01}};
    ASSERT_TRUE(first.process(gyro).valid);
    ASSERT_TRUE(first.process(wheel).valid);
    ASSERT_TRUE(second.process(gyro).valid);
    ASSERT_TRUE(second.process(wheel).valid);
    EXPECT_TRUE(first.state().isApprox(second.state(), 1e-14));
    EXPECT_TRUE(first.covariance().isApprox(second.covariance(), 1e-14));
  }
  EXPECT_EQ(first.accepted_gps(), 1U);
  EXPECT_EQ(first.rejected_gps(), 0U);
  EXPECT_GT(first.state()(PX), 2.5);
}
TEST(Replay, DeadReckoningIsCausalAndIgnoresGps) {
  Replay replay(State::Zero(), Covariance::Identity(), Config{}, Method::DeadReckoning);
  ASSERT_TRUE(replay.process({0, WheelSpeedMeasurement{2.0, 0.1}}).valid);
  ASSERT_TRUE(replay.process({1000000, WheelSpeedMeasurement{10.0, 0.1}}).valid);
  EXPECT_NEAR(replay.state()(PX), 2.0, 1e-12);  // New speed cannot affect past interval.
  GpsMeasurement gps;
  gps.position_m << -100.0, 30.0;
  ASSERT_TRUE(replay.process({1000000, gps}).valid);
  EXPECT_NEAR(replay.state()(PX), 2.0, 1e-12);
  EXPECT_EQ(replay.accepted_gps(), 0U);
  EXPECT_EQ(replay.ignored_gps(), 1U);
}
TEST(Replay, GpsSnapshotsBracketCorrectionAndInvalidEventsDoNotAdvanceHistory) {
  State state = State::Zero();
  state(SPEED) = 1.0;
  Replay replay(state, Covariance::Identity(), Config{}, Method::GatedEkf);
  GpsMeasurement measurement;
  measurement.position_m << 0.2, 0.1;
  const auto first = replay.process({0, measurement});
  ASSERT_TRUE(first.valid);
  ASSERT_TRUE(first.gps.has_value());
  EXPECT_EQ(first.gps->predicted_state, state);
  EXPECT_EQ(first.gps->posterior_state, replay.state());
  EXPECT_FALSE(first.gps->since_accepted_us.has_value());
  EXPECT_FALSE(first.gps->since_available_us.has_value());
  EXPECT_NE(first.gps->predicted_covariance, first.gps->posterior_covariance);
  EXPECT_NEAR(first.update.nis, 0.025, 1e-14);  // S=2I BEFORE correction.
  measurement.position_m << 100.0, 100.0;
  const auto rejected = replay.process({200000, measurement});
  ASSERT_TRUE(rejected.valid);
  ASSERT_EQ(rejected.update.status, UpdateStatus::Rejected);
  EXPECT_EQ(rejected.gps->predicted_state, rejected.gps->posterior_state);
  EXPECT_EQ(rejected.gps->predicted_covariance, rejected.gps->posterior_covariance);
  EXPECT_EQ(rejected.gps->since_accepted_us.value(), 200000);
  EXPECT_EQ(rejected.gps->rejection_duration_us, 0);
  EXPECT_FALSE(replay.process({200000, measurement}).valid);
  GpsMeasurement invalid_measurement = measurement;
  invalid_measurement.covariance_m2(0, 0) = -1.0;
  EXPECT_FALSE(replay.process({300000, invalid_measurement}).valid);
  const auto again = replay.process({400000, measurement});
  ASSERT_TRUE(again.valid);
  EXPECT_EQ(again.gps->since_available_us.value(), 200000);
  EXPECT_EQ(again.gps->since_accepted_us.value(), 400000);
  EXPECT_EQ(again.gps->rejection_duration_us, 200000);
}
TEST(Input, StrictMalformedCsvChecks) {
  EXPECT_NO_THROW(parse_event("10000,gyro,0.1,0,0.001,0,0"));
  EXPECT_NO_THROW(parse_event("10000,gps,1,2,1,0.1,1"));
  EXPECT_THROW(parse_event("1.5,gyro,0.1,0,1,0,0"), std::invalid_argument);
  EXPECT_THROW(parse_event("-1,gyro,0.1,0,1,0,0"), std::invalid_argument);
  EXPECT_THROW(parse_event("999999999999999999999,gyro,0,0,1,0,0"), std::invalid_argument);
  EXPECT_THROW(parse_event("10,gyro,nan,0,1,0,0"), std::invalid_argument);
  EXPECT_THROW(parse_event("10,gyro,1junk,0,1,0,0"), std::invalid_argument);
  EXPECT_THROW(parse_event("10,gyro,1,2,1,0,0"), std::invalid_argument);
  EXPECT_THROW(parse_event("10,gps,1,2,1,2,1"), std::invalid_argument);
  EXPECT_THROW(parse_event("10,unknown,1,0,1,0,0"), std::invalid_argument);
  EXPECT_THROW(parse_event("10,wheel,1,0,1,0,0,"), std::invalid_argument);
}
}  // namespace
}  // namespace navtrace
