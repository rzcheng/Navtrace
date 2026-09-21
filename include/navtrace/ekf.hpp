#pragma once

#include <Eigen/Core>
#include <string>

namespace navtrace {
constexpr int kStateSize = 6;
using State = Eigen::Matrix<double, kStateSize, 1>;
using Covariance = Eigen::Matrix<double, kStateSize, kStateSize>;
using GpsJacobian = Eigen::Matrix<double, 2, kStateSize>;
using ScalarJacobian = Eigen::Matrix<double, 1, kStateSize>;
enum StateIndex { PX = 0, PY = 1, YAW = 2, SPEED = 3, OMEGA = 4, BIAS = 5 };

struct Config {
  double speed_diffusion_m2_per_s3 = 0.25;
  double yaw_rate_diffusion_rad2_per_s3 = 0.01;
  double bias_diffusion_rad2_per_s3 = 1e-7;
  double max_step_s = 0.01;
  double max_gap_s = 2.0;
  bool gps_gate_enabled = false;
  double gps_nis_threshold = 9.210340371976184;
};

struct GpsMeasurement {
  Eigen::Vector2d position_m = Eigen::Vector2d::Zero();
  Eigen::Matrix2d covariance_m2 = Eigen::Matrix2d::Identity();
};
struct WheelSpeedMeasurement { double speed_m_per_s; double variance_m2_per_s2; };
struct GyroMeasurement { double yaw_rate_rad_per_s; double variance_rad2_per_s2; };

enum class UpdateStatus { Accepted, Rejected, Invalid };
struct UpdateResult {
  UpdateStatus status = UpdateStatus::Invalid;
  std::string reason;
  // GPS diagnostics; unused for scalar updates.
  Eigen::Vector2d residual_m = Eigen::Vector2d::Zero();
  Eigen::Matrix2d innovation_covariance_m2 = Eigen::Matrix2d::Zero();
  double nis = 0.0;
  double threshold = 0.0;
};
struct OperationResult { bool ok; std::string reason; };

double wrap_angle(double angle_rad);
State propagate_mean(const State& state, double dt_s);
Covariance transition_jacobian(const State& state, double dt_s);
GpsJacobian gps_jacobian();
ScalarJacobian wheel_jacobian();
ScalarJacobian gyro_jacobian();
bool valid_covariance(const Covariance& covariance);
bool valid_measurement(const GpsMeasurement& measurement);
bool valid_measurement(const WheelSpeedMeasurement& measurement);
bool valid_measurement(const GyroMeasurement& measurement);

class Ekf {
 public:
  // Invalid construction is a programming/configuration error (invalid_argument).
  Ekf(State initial_state, Covariance initial_covariance, Config config = {});
  OperationResult predict(double dt_s);
  UpdateResult update_gps(const GpsMeasurement& measurement);
  UpdateResult update_wheel_speed(const WheelSpeedMeasurement& measurement);
  UpdateResult update_gyro(const GyroMeasurement& measurement);
  const State& state() const { return state_; }
  const Covariance& covariance() const { return covariance_; }
  const Config& config() const { return config_; }

 private:
  UpdateResult update_scalar(double measurement, double variance, const ScalarJacobian& h);
  State state_;
  Covariance covariance_;
  Config config_;
};
}  // namespace navtrace
