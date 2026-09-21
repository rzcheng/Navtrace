#include "navtrace/replay.hpp"
#include <cmath>

namespace navtrace {
namespace {
Config method_config(Config config, Method method) {
  config.gps_gate_enabled = method == Method::GatedEkf;
  return config;
}
}  // namespace

Replay::Replay(State initial_state, Covariance initial_covariance, Config config, Method method)
    : ekf_(initial_state, initial_covariance, method_config(config, method)),
      dr_state_(ekf_.state()), method_(method) {}

ReplayResult Replay::process(const SensorEvent& event) {
  ReplayResult result;
  const std::size_t rank = event.measurement.index();
  const unsigned bit = 1U << rank;
  // Validate before prediction. Copy candidates also makes numerical failure atomic.
  if (event.timestamp_us < 0 || event.timestamp_us < timestamp_us_) {
    result.reason = "negative or out-of-order timestamp";
    return result;
  }
  if (has_event_ && event.timestamp_us == timestamp_us_) {
    if ((sensors_at_timestamp_ & bit) != 0U) {
      result.reason = "duplicate sensor at timestamp";
      return result;
    }
    if (rank < last_sensor_rank_) {
      result.reason = "simultaneous order must be gyro, wheel, GPS";
      return result;
    }
  }
  if (!std::visit([](const auto& measurement) { return valid_measurement(measurement); }, event.measurement)) {
    result.reason = "invalid measurement or covariance";
    return result;
  }
  const double dt_s = static_cast<double>(event.timestamp_us - timestamp_us_) * 1e-6;
  if (dt_s > ekf_.config().max_gap_s) {
    result.reason = "event gap exceeds max_gap_s";
    return result;
  }
  Ekf candidate = ekf_;
  State dr_candidate = dr_state_;
  if (method_ == Method::DeadReckoning) {
    if (dt_s > 0.0) {
      const int steps = static_cast<int>(std::ceil(dt_s / ekf_.config().max_step_s));
      for (int i = 0; i < steps; ++i) dr_candidate = propagate_mean(dr_candidate, dt_s / static_cast<double>(steps));
    }
    if (const auto* gyro = std::get_if<GyroMeasurement>(&event.measurement)) {
      dr_candidate(OMEGA) = gyro->yaw_rate_rad_per_s - dr_candidate(BIAS);
    } else if (const auto* wheel = std::get_if<WheelSpeedMeasurement>(&event.measurement)) {
      dr_candidate(SPEED) = wheel->speed_m_per_s;
    }
    if (!dr_candidate.allFinite()) {
      result.reason = "nonfinite dead-reckoning state";
      return result;
    }
    result.update.status = UpdateStatus::Accepted;
    result.update.reason = rank == 2U ? "GPS ignored by dead reckoning" : "dead-reckoning hold updated";
  } else {
    const auto predicted = candidate.predict(dt_s);
    if (!predicted.ok) {
      result.reason = predicted.reason;
      return result;
    }
    if (const auto* gyro = std::get_if<GyroMeasurement>(&event.measurement)) {
      result.update = candidate.update_gyro(*gyro);
    } else if (const auto* wheel = std::get_if<WheelSpeedMeasurement>(&event.measurement)) {
      result.update = candidate.update_wheel_speed(*wheel);
    } else {
      result.gps.emplace();
      result.gps->predicted_state = candidate.state();
      result.gps->predicted_covariance = candidate.covariance();
      result.update = candidate.update_gps(std::get<GpsMeasurement>(event.measurement));
    }
    if (result.update.status == UpdateStatus::Invalid) {
      result.reason = result.update.reason;
      return result;
    }
  }
  ekf_ = candidate;
  dr_state_ = dr_candidate;
  if (!has_event_ || event.timestamp_us != timestamp_us_) sensors_at_timestamp_ = 0;
  sensors_at_timestamp_ |= bit;
  last_sensor_rank_ = rank;
  timestamp_us_ = event.timestamp_us;
  has_event_ = true;
  if (rank == 2U) {
    if (method_ == Method::DeadReckoning) ++ignored_gps_;
    else if (result.update.status == UpdateStatus::Accepted) ++accepted_gps_;
    else ++rejected_gps_;
    if (result.gps) {
      auto& gps = *result.gps;
      gps.posterior_state = candidate.state();
      gps.posterior_covariance = candidate.covariance();
      if (last_available_gps_us_) gps.since_available_us = timestamp_us_ - *last_available_gps_us_;
      if (last_accepted_gps_us_) gps.since_accepted_us = timestamp_us_ - *last_accepted_gps_us_;
      if (result.update.status == UpdateStatus::Accepted) {
        last_accepted_gps_us_ = timestamp_us_;
        rejection_start_us_.reset();
      } else {
        if (!rejection_start_us_) rejection_start_us_ = timestamp_us_;
        gps.rejection_duration_us = timestamp_us_ - *rejection_start_us_;
      }
      last_available_gps_us_ = timestamp_us_;
    }
  }
  result.valid = true;
  result.reason = result.update.reason;
  return result;
}
}  // namespace navtrace
