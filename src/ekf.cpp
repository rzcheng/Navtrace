#include "navtrace/ekf.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace navtrace {
namespace {
constexpr double kPi = 3.14159265358979323846;
bool nonnegative_finite(double x) { return std::isfinite(x) && x >= 0.0; }
bool valid_config(const Config& c) {
  return nonnegative_finite(c.speed_diffusion_m2_per_s3) &&
         nonnegative_finite(c.yaw_rate_diffusion_rad2_per_s3) &&
         nonnegative_finite(c.bias_diffusion_rad2_per_s3) &&
         std::isfinite(c.max_step_s) && c.max_step_s >= 1e-5 && c.max_step_s <= 0.1 &&
         std::isfinite(c.max_gap_s) && c.max_gap_s >= c.max_step_s && c.max_gap_s <= 60.0 &&
         std::isfinite(c.gps_nis_threshold) && c.gps_nis_threshold > 0.0;
}
UpdateResult invalid(std::string reason) {
  UpdateResult result;
  result.reason = std::move(reason);
  return result;
}
}  // namespace

double wrap_angle(double angle_rad) {
  double wrapped = std::remainder(angle_rad, 2.0 * kPi);
  if (wrapped >= kPi) wrapped -= 2.0 * kPi;
  return wrapped;
}

State propagate_mean(const State& state, double dt_s) {
  State next = state;
  next(PX) += dt_s * state(SPEED) * std::cos(state(YAW));
  next(PY) += dt_s * state(SPEED) * std::sin(state(YAW));
  next(YAW) = wrap_angle(state(YAW) + dt_s * state(OMEGA));
  return next;
}

Covariance transition_jacobian(const State& state, double dt_s) {
  Covariance f = Covariance::Identity();
  f(PX, YAW) = -dt_s * state(SPEED) * std::sin(state(YAW));
  f(PX, SPEED) = dt_s * std::cos(state(YAW));
  f(PY, YAW) = dt_s * state(SPEED) * std::cos(state(YAW));
  f(PY, SPEED) = dt_s * std::sin(state(YAW));
  f(YAW, OMEGA) = dt_s;
  return f;
}

GpsJacobian gps_jacobian() {
  GpsJacobian h = GpsJacobian::Zero();
  h(0, PX) = 1.0;
  h(1, PY) = 1.0;
  return h;
}
ScalarJacobian wheel_jacobian() {
  ScalarJacobian h = ScalarJacobian::Zero();
  h(SPEED) = 1.0;
  return h;
}
ScalarJacobian gyro_jacobian() {
  ScalarJacobian h = ScalarJacobian::Zero();
  h(OMEGA) = 1.0;
  h(BIAS) = 1.0;
  return h;
}

bool valid_covariance(const Covariance& covariance) {
  if (!covariance.allFinite()) return false;
  const double tolerance = 1e-10 * std::max(1.0, covariance.cwiseAbs().maxCoeff());
  if ((covariance - covariance.transpose()).cwiseAbs().maxCoeff() > tolerance) return false;
  const Covariance symmetric = 0.5 * (covariance + covariance.transpose());
  Eigen::SelfAdjointEigenSolver<Covariance> eigenvalues(symmetric, Eigen::EigenvaluesOnly);
  return eigenvalues.info() == Eigen::Success && eigenvalues.eigenvalues().minCoeff() >= -tolerance;
}

bool valid_measurement(const GpsMeasurement& measurement) {
  const auto& r = measurement.covariance_m2;
  if (!measurement.position_m.allFinite() || !r.allFinite()) return false;
  // Reject asymmetric inputs rather than silently selecting one triangle.
  if ((r - r.transpose()).cwiseAbs().maxCoeff() > 1e-12 * std::max(1.0, r.cwiseAbs().maxCoeff())) return false;
  Eigen::LLT<Eigen::Matrix2d> decomposition(0.5 * (r + r.transpose()));
  return decomposition.info() == Eigen::Success;
}
bool valid_measurement(const WheelSpeedMeasurement& measurement) {
  return std::isfinite(measurement.speed_m_per_s) &&
         std::isfinite(measurement.variance_m2_per_s2) && measurement.variance_m2_per_s2 > 0.0;
}
bool valid_measurement(const GyroMeasurement& measurement) {
  return std::isfinite(measurement.yaw_rate_rad_per_s) &&
         std::isfinite(measurement.variance_rad2_per_s2) && measurement.variance_rad2_per_s2 > 0.0;
}

Ekf::Ekf(State initial_state, Covariance initial_covariance, Config config)
    : state_(std::move(initial_state)), covariance_(std::move(initial_covariance)), config_(config) {
  if (!state_.allFinite() || !valid_covariance(covariance_) || !valid_config(config_)) {
    throw std::invalid_argument("invalid initial state, covariance, or filter configuration");
  }
  state_(YAW) = wrap_angle(state_(YAW));
  covariance_ = (0.5 * (covariance_ + covariance_.transpose())).eval();
}

OperationResult Ekf::predict(double dt_s) {
  if (!std::isfinite(dt_s) || dt_s < 0.0 || dt_s > config_.max_gap_s) {
    return {false, "prediction dt must be finite, nonnegative, and <= max_gap_s"};
  }
  if (dt_s == 0.0) return {true, "zero dt"};
  const int steps = static_cast<int>(std::ceil(dt_s / config_.max_step_s));
  const double h = dt_s / static_cast<double>(steps);
  Covariance q = Covariance::Zero();
  q(SPEED, SPEED) = config_.speed_diffusion_m2_per_s3 * h;
  q(OMEGA, OMEGA) = config_.yaw_rate_diffusion_rad2_per_s3 * h;
  q(BIAS, BIAS) = config_.bias_diffusion_rad2_per_s3 * h;
  State next_state = state_;
  Covariance next_covariance = covariance_;
  for (int step = 0; step < steps; ++step) {
    const Covariance f = transition_jacobian(next_state, h);
    next_state = propagate_mean(next_state, h);
    next_covariance = (f * next_covariance * f.transpose() + q).eval();
    next_covariance = (0.5 * (next_covariance + next_covariance.transpose())).eval();
    if (!next_state.allFinite() || !valid_covariance(next_covariance)) {
      return {false, "nonfinite state or invalid covariance during prediction"};
    }
  }
  state_ = next_state;
  covariance_ = next_covariance;
  return {true, "predicted"};
}

UpdateResult Ekf::update_gps(const GpsMeasurement& measurement) {
  if (!valid_measurement(measurement)) return invalid("invalid GPS position/covariance");
  const GpsJacobian h = gps_jacobian();
  const Eigen::Matrix2d r = 0.5 * (measurement.covariance_m2 + measurement.covariance_m2.transpose());
  UpdateResult result;
  result.threshold = config_.gps_nis_threshold;
  result.residual_m = measurement.position_m - h * state_;
  result.innovation_covariance_m2 = h * covariance_ * h.transpose() + r;
  if (!result.innovation_covariance_m2.allFinite()) return invalid("nonfinite GPS innovation covariance");
  Eigen::LLT<Eigen::Matrix2d> decomposition(result.innovation_covariance_m2);
  if (decomposition.info() != Eigen::Success) return invalid("GPS innovation LLT failed");
  const Eigen::Vector2d solved_residual = decomposition.solve(result.residual_m);
  result.nis = result.residual_m.dot(solved_residual);
  if (decomposition.info() != Eigen::Success || !solved_residual.allFinite() || !std::isfinite(result.nis) || result.nis < 0.0) {
    return invalid("GPS innovation solve failed");
  }
  if (config_.gps_gate_enabled && result.nis > result.threshold) {
    result.status = UpdateStatus::Rejected;
    result.reason = "GPS NIS exceeds threshold";
    return result;
  }
  const Eigen::Matrix<double, 6, 2> gain = decomposition.solve((covariance_ * h.transpose()).transpose()).transpose();
  State next_state = state_ + gain * result.residual_m;
  next_state(YAW) = wrap_angle(next_state(YAW));
  const Covariance a = Covariance::Identity() - gain * h;
  Covariance next_covariance = a * covariance_ * a.transpose() + gain * r * gain.transpose();
  next_covariance = (0.5 * (next_covariance + next_covariance.transpose())).eval();
  if (decomposition.info() != Eigen::Success || !next_state.allFinite() || !valid_covariance(next_covariance)) {
    return invalid("invalid GPS posterior");
  }
  state_ = next_state;
  covariance_ = next_covariance;
  result.status = UpdateStatus::Accepted;
  result.reason = "GPS accepted";
  return result;
}

UpdateResult Ekf::update_scalar(double measurement, double variance, const ScalarJacobian& h) {
  if (!std::isfinite(measurement) || !std::isfinite(variance) || variance <= 0.0) return invalid("invalid scalar measurement/variance");
  Eigen::Matrix<double, 1, 1> s;
  s(0, 0) = (h * covariance_ * h.transpose())(0, 0) + variance;
  if (!s.allFinite()) return invalid("nonfinite scalar innovation covariance");
  Eigen::LLT<Eigen::Matrix<double, 1, 1>> decomposition(s);
  if (decomposition.info() != Eigen::Success) return invalid("scalar innovation LLT failed");
  const State gain = decomposition.solve((covariance_ * h.transpose()).transpose()).transpose();
  State next_state = state_ + gain * (measurement - (h * state_)(0));
  next_state(YAW) = wrap_angle(next_state(YAW));
  const Covariance a = Covariance::Identity() - gain * h;
  Covariance next_covariance = a * covariance_ * a.transpose() + variance * gain * gain.transpose();
  next_covariance = (0.5 * (next_covariance + next_covariance.transpose())).eval();
  if (decomposition.info() != Eigen::Success || !next_state.allFinite() || !valid_covariance(next_covariance)) return invalid("invalid scalar posterior");
  state_ = next_state;
  covariance_ = next_covariance;
  UpdateResult result;
  result.status = UpdateStatus::Accepted;
  result.reason = "scalar accepted";
  return result;
}
UpdateResult Ekf::update_wheel_speed(const WheelSpeedMeasurement& measurement) {
  return update_scalar(measurement.speed_m_per_s, measurement.variance_m2_per_s2, wheel_jacobian());
}
UpdateResult Ekf::update_gyro(const GyroMeasurement& measurement) {
  return update_scalar(measurement.yaw_rate_rad_per_s, measurement.variance_rad2_per_s2, gyro_jacobian());
}
}  // namespace navtrace
