#include "navtrace/ekf.hpp"
#include <gtest/gtest.h>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <limits>

namespace navtrace {
namespace {
constexpr double pi = 3.14159265358979323846;
Config noiseless() {
  Config c;
  c.speed_diffusion_m2_per_s3 = 0.0;
  c.yaw_rate_diffusion_rad2_per_s3 = 0.0;
  c.bias_diffusion_rad2_per_s3 = 0.0;
  return c;
}
TEST(Angles, WrapAndCompareAcrossBoundary) {
  EXPECT_DOUBLE_EQ(wrap_angle(pi), -pi);
  EXPECT_NEAR(wrap_angle(-pi - 0.1), pi - 0.1, 1e-14);
  EXPECT_NEAR(wrap_angle(7.0 * pi), -pi, 1e-14);
  EXPECT_NEAR(wrap_angle((pi - 0.01) - (-pi + 0.01)), -0.02, 1e-14);
}
TEST(Prediction, StationaryAndStraight) {
  State s = State::Zero();
  Ekf stationary(s, Covariance::Zero(), noiseless());
  ASSERT_TRUE(stationary.predict(1.0).ok);
  EXPECT_EQ(stationary.state(), s);
  EXPECT_EQ(stationary.covariance(), Covariance::Zero());
  s(SPEED) = -2.0;
  s(YAW) = pi / 3.0;
  Ekf straight(s, Covariance::Identity(), noiseless());
  ASSERT_TRUE(straight.predict(1.0).ok);
  EXPECT_NEAR(straight.state()(PX), -1.0, 1e-12);
  EXPECT_NEAR(straight.state()(PY), -std::sqrt(3.0), 1e-12);
  EXPECT_DOUBLE_EQ(straight.state()(YAW), s(YAW));
}
TEST(Prediction, ConstantTurnAgainstAnalyticalArc) {
  State s = State::Zero();
  s(SPEED) = 2.0;
  s(OMEGA) = 0.4;
  Config config = noiseless();
  config.max_step_s = 0.001;
  Ekf filter(s, Covariance::Identity(), config);
  ASSERT_TRUE(filter.predict(2.0).ok);
  const double exact_x = 5.0 * std::sin(0.8);
  const double exact_y = 5.0 * (1.0 - std::cos(0.8));
  // Left-endpoint integration error <= v*|omega|*T*h/2 = 0.0008 m.
  EXPECT_LT(std::hypot(filter.state()(PX) - exact_x, filter.state()(PY) - exact_y), 0.00081);
  EXPECT_NEAR(filter.state()(YAW), 0.8, 1e-12);
}
TEST(Jacobians, TransitionFiniteDifferencesWithWrappedOutput) {
  State state;
  state << 2.0, -1.0, pi - 0.002, -1.7, 0.5, 0.03;
  constexpr double dt = 0.01;
  constexpr double epsilon = 1e-6;
  Covariance numerical;
  for (int i = 0; i < 6; ++i) {
    State plus = state;
    State minus = state;
    plus(i) += epsilon;
    minus(i) -= epsilon;
    State difference = propagate_mean(plus, dt) - propagate_mean(minus, dt);
    difference(YAW) = wrap_angle(difference(YAW));
    numerical.col(i) = difference / (2.0 * epsilon);
  }
  EXPECT_TRUE(numerical.isApprox(transition_jacobian(state, dt), 1e-8));
}
TEST(Jacobians, MeasurementModelsFiniteDifferences) {
  State state;
  state << 3.0, 4.0, 0.6, 2.1, -0.2, 0.01;
  Eigen::Matrix<double, 4, 6> numerical;
  auto observations = [](const State& s) {
    Eigen::Vector4d z;
    z << s(PX), s(PY), s(SPEED), s(OMEGA) + s(BIAS);
    return z;
  };
  for (int i = 0; i < 6; ++i) {
    State plus = state, minus = state;
    plus(i) += 1e-6;
    minus(i) -= 1e-6;
    numerical.col(i) = (observations(plus) - observations(minus)) / 2e-6;
  }
  Eigen::Matrix<double, 4, 6> analytical;
  analytical.topRows<2>() = gps_jacobian();
  analytical.row(2) = wheel_jacobian();
  analytical.row(3) = gyro_jacobian();
  EXPECT_TRUE(numerical.isApprox(analytical, 1e-8));
}
TEST(Update, IndependentHandComputedGps) {
  State state = State::Zero();
  state(PX) = 10.0;
  Covariance p = Covariance::Identity();
  p(PX, PX) = 4.0;
  Ekf filter(state, p);
  GpsMeasurement gps;
  gps.position_m << 12.0, 0.0;
  gps.covariance_m2 = 9.0 * Eigen::Matrix2d::Identity();
  const auto result = filter.update_gps(gps);
  ASSERT_EQ(result.status, UpdateStatus::Accepted);
  EXPECT_NEAR(filter.state()(PX), 10.0 + 8.0 / 13.0, 1e-12);
  EXPECT_NEAR(filter.covariance()(PX, PX), 36.0 / 13.0, 1e-12);
  EXPECT_NEAR(result.nis, 4.0 / 13.0, 1e-12);
  EXPECT_DOUBLE_EQ(result.innovation_covariance_m2(0, 0), 13.0);
}
TEST(Update, CrossCovariancesArePreservedAndUsed) {
  Covariance p = Covariance::Identity();
  p(PX, SPEED) = p(SPEED, PX) = 0.5;
  Ekf filter(State::Zero(), p);
  ASSERT_EQ(filter.update_wheel_speed({2.0, 1.0}).status, UpdateStatus::Accepted);
  EXPECT_NEAR(filter.state()(PX), 0.5, 1e-12);
  EXPECT_NEAR(filter.covariance()(PX, SPEED), 0.25, 1e-12);
  EXPECT_NEAR(filter.covariance()(PX, PX), 0.875, 1e-12);
}
TEST(Update, GyroObservesRatePlusBias) {
  State s = State::Zero();
  s(OMEGA) = 0.3;
  s(BIAS) = 0.1;
  Ekf filter(s, Covariance::Identity());
  ASSERT_EQ(filter.update_gyro({0.4, 0.01}).status, UpdateStatus::Accepted);
  EXPECT_TRUE(filter.state().isApprox(s, 1e-14));
  EXPECT_LT(filter.covariance()(OMEGA, BIAS), 0.0);
}
TEST(Covariance, SymmetricPsdAfterRepeatedPredictionAndUpdates) {
  Ekf filter(State::Zero(), Covariance::Identity());
  for (int i = 0; i < 300; ++i) {
    ASSERT_TRUE(filter.predict(0.01).ok);
    ASSERT_EQ(filter.update_wheel_speed({1.5, 0.0025}).status, UpdateStatus::Accepted);
    ASSERT_EQ(filter.update_gyro({0.12, 0.000064}).status, UpdateStatus::Accepted);
    if (i % 20 == 0) {
      GpsMeasurement gps;
      gps.position_m << 0.015 * i, 0.0;
      ASSERT_EQ(filter.update_gps(gps).status, UpdateStatus::Accepted);
    }
    const auto& p = filter.covariance();
    EXPECT_LT((p - p.transpose()).cwiseAbs().maxCoeff(), 1e-12);
    Eigen::SelfAdjointEigenSolver<Covariance> eigenvalues(p);
    EXPECT_GE(eigenvalues.eigenvalues().minCoeff(), -1e-10);
  }
}
TEST(Covariance, ProcessNoiseScalesWithElapsedTime) {
  Config config;
  config.speed_diffusion_m2_per_s3 = 2.0;
  config.yaw_rate_diffusion_rad2_per_s3 = 3.0;
  config.bias_diffusion_rad2_per_s3 = 4.0;
  Ekf short_step(State::Zero(), Covariance::Zero(), config);
  Ekf long_step = short_step;
  ASSERT_TRUE(short_step.predict(0.1).ok);
  ASSERT_TRUE(long_step.predict(0.2).ok);
  EXPECT_NEAR(short_step.covariance()(SPEED, SPEED), 0.2, 1e-12);
  EXPECT_NEAR(long_step.covariance()(SPEED, SPEED), 0.4, 1e-12);
  EXPECT_NEAR(long_step.covariance()(OMEGA, OMEGA), 0.6, 1e-12);
  EXPECT_NEAR(long_step.covariance()(BIAS, BIAS), 0.8, 1e-12);
  const Covariance before = long_step.covariance();
  ASSERT_TRUE(long_step.predict(0.0).ok);
  EXPECT_EQ(long_step.covariance(), before);
}
TEST(Validation, InvalidConstructionAndConfiguration) {
  Covariance p = Covariance::Identity();
  p(0, 0) = -0.1;
  EXPECT_THROW(Ekf(State::Zero(), p), std::invalid_argument);
  p = Covariance::Identity();
  p(0, 1) = 0.1;
  EXPECT_THROW(Ekf(State::Zero(), p), std::invalid_argument);
  State state = State::Zero();
  state(YAW) = std::numeric_limits<double>::infinity();
  EXPECT_THROW(Ekf(state, Covariance::Identity()), std::invalid_argument);
  Config config;
  config.speed_diffusion_m2_per_s3 = -1.0;
  EXPECT_THROW(Ekf(State::Zero(), Covariance::Identity(), config), std::invalid_argument);
  config = Config{};
  config.max_step_s = 0.0;
  EXPECT_THROW(Ekf(State::Zero(), Covariance::Identity(), config), std::invalid_argument);
  config = Config{};
  config.gps_nis_threshold = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(Ekf(State::Zero(), Covariance::Identity(), config), std::invalid_argument);
}
TEST(Validation, InvalidOperationsAreTransactional) {
  Ekf filter(State::Zero(), Covariance::Identity());
  const State state = filter.state();
  const Covariance p = filter.covariance();
  EXPECT_FALSE(filter.predict(-1.0).ok);
  EXPECT_FALSE(filter.predict(2.01).ok);
  EXPECT_FALSE(filter.predict(std::numeric_limits<double>::infinity()).ok);
  EXPECT_EQ(filter.update_wheel_speed({1.0, 0.0}).status, UpdateStatus::Invalid);
  EXPECT_EQ(filter.update_gyro({std::numeric_limits<double>::quiet_NaN(), 1.0}).status, UpdateStatus::Invalid);
  GpsMeasurement gps;
  gps.covariance_m2 << 1.0, 2.0, 2.0, 1.0;
  EXPECT_EQ(filter.update_gps(gps).status, UpdateStatus::Invalid);
  gps.covariance_m2 << 1.0, 0.1, 0.0, 1.0;
  EXPECT_EQ(filter.update_gps(gps).status, UpdateStatus::Invalid);
  gps.covariance_m2 = Eigen::Matrix2d::Identity();
  gps.position_m(0) = std::numeric_limits<double>::infinity();
  EXPECT_EQ(filter.update_gps(gps).status, UpdateStatus::Invalid);
  EXPECT_EQ(filter.state(), state);
  EXPECT_EQ(filter.covariance(), p);
}
TEST(Validation, NumericalOverflowDoesNotCommit) {
  State s = State::Zero();
  s(SPEED) = std::numeric_limits<double>::max();
  Ekf filter(s, Covariance::Identity());
  EXPECT_FALSE(filter.predict(2.0).ok);
  EXPECT_EQ(filter.state(), s);
  EXPECT_EQ(filter.covariance(), Covariance::Identity());
  GpsMeasurement gps;
  gps.position_m.setConstant(std::numeric_limits<double>::max());
  EXPECT_EQ(filter.update_gps(gps).status, UpdateStatus::Invalid);
  EXPECT_EQ(filter.state(), s);
}
TEST(Gate, NominalAcceptanceAndOutlierWithoutCorrection) {
  Config config;
  config.gps_gate_enabled = true;
  Ekf filter(State::Zero(), Covariance::Identity(), config);
  GpsMeasurement gps;
  gps.position_m << 0.1, -0.1;
  ASSERT_EQ(filter.update_gps(gps).status, UpdateStatus::Accepted);
  const State state = filter.state();
  const Covariance p = filter.covariance();
  gps.position_m << 100.0, -100.0;
  const auto result = filter.update_gps(gps);
  EXPECT_EQ(result.status, UpdateStatus::Rejected);
  EXPECT_GT(result.nis, result.threshold);
  EXPECT_EQ(result.reason, "GPS NIS exceeds threshold");
  EXPECT_EQ(filter.state(), state);
  EXPECT_EQ(filter.covariance(), p);
}
TEST(Gate, SwitchAndConfigurableSquaredDistanceThreshold) {
  GpsMeasurement gps;
  gps.position_m << 4.0, 0.0;  // S=2I, NIS=8, distance=sqrt(8).
  Config config;
  config.gps_gate_enabled = true;
  config.gps_nis_threshold = 7.0;
  Ekf gated(State::Zero(), Covariance::Identity(), config);
  const auto result = gated.update_gps(gps);
  EXPECT_EQ(result.status, UpdateStatus::Rejected);
  EXPECT_NEAR(result.nis, 8.0, 1e-12);
  config.gps_gate_enabled = false;
  Ekf basic(State::Zero(), Covariance::Identity(), config);
  EXPECT_EQ(basic.update_gps(gps).status, UpdateStatus::Accepted);
}
}  // namespace
}  // namespace navtrace
