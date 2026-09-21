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
  const std::size_t sensor_order = event.measurement.index();
  const unsigned sensor_bit = 1U << sensor_order;
  // Check the event before moving forward. Work on copies so a failed
  // update leaves the last good state alone.
  if (event.timestamp_us < 0 || event.timestamp_us < timestamp_us_) {
    result.reason = "negative or out-of-order timestamp";
    return result;
  }
  if (has_event_ && event.timestamp_us == timestamp_us_) {
    if ((sensors_at_timestamp_ & sensor_bit) != 0U) {
      result.reason = "duplicate sensor at timestamp";
      return result;
    }
    if (sensor_order < last_sensor_rank_) {
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
  Ekf next_filter = ekf_;
  State next_dr_state = dr_state_;
  if (method_ == Method::DeadReckoning) {
    if (dt_s > 0.0) {
      const int steps = static_cast<int>(std::ceil(dt_s / ekf_.config().max_step_s));
      for (int step = 0; step < steps; ++step) {
        next_dr_state = propagate_mean(next_dr_state, dt_s / static_cast<double>(steps));
      }
    }
    if (const auto* gyro = std::get_if<GyroMeasurement>(&event.measurement)) {
      next_dr_state(OMEGA) = gyro->yaw_rate_rad_per_s - next_dr_state(BIAS);
    } else if (const auto* wheel = std::get_if<WheelSpeedMeasurement>(&event.measurement)) {
      next_dr_state(SPEED) = wheel->speed_m_per_s;
    }
    if (!next_dr_state.allFinite()) {
      result.reason = "nonfinite dead-reckoning state";
      return result;
    }
    result.update.status = UpdateStatus::Accepted;
    result.update.reason = sensor_order == 2U ? "GPS ignored by dead reckoning" : "dead-reckoning hold updated";
  } else {
    const auto prediction = next_filter.predict(dt_s);
    if (!prediction.ok) {
      result.reason = prediction.reason;
      return result;
    }
    if (const auto* gyro = std::get_if<GyroMeasurement>(&event.measurement)) {
      result.update = next_filter.update_gyro(*gyro);
    } else if (const auto* wheel = std::get_if<WheelSpeedMeasurement>(&event.measurement)) {
      result.update = next_filter.update_wheel_speed(*wheel);
    } else {
      result.gps.emplace();
      result.gps->predicted_state = next_filter.state();
      result.gps->predicted_covariance = next_filter.covariance();
      result.update = next_filter.update_gps(std::get<GpsMeasurement>(event.measurement));
    }
    if (result.update.status == UpdateStatus::Invalid) {
      result.reason = result.update.reason;
      return result;
    }
  }
  // The event is valid, so keep the new state and time.
  ekf_ = next_filter;
  dr_state_ = next_dr_state;
  if (!has_event_ || event.timestamp_us != timestamp_us_) sensors_at_timestamp_ = 0;
  sensors_at_timestamp_ |= sensor_bit;
  last_sensor_rank_ = sensor_order;
  timestamp_us_ = event.timestamp_us;
  has_event_ = true;
  if (sensor_order == 2U) {
    if (method_ == Method::DeadReckoning) ++ignored_gps_;
    else if (result.update.status == UpdateStatus::Accepted) ++accepted_gps_;
    else ++rejected_gps_;
    if (result.gps) {
      auto& gps = *result.gps;
      gps.posterior_state = next_filter.state();
      gps.posterior_covariance = next_filter.covariance();
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
