#pragma once

#include "navtrace/ekf.hpp"
#include <cstdint>
#include <optional>
#include <variant>

namespace navtrace {
// At the same timestamp, process gyro, then wheel speed, then GPS.
using Measurement = std::variant<GyroMeasurement, WheelSpeedMeasurement, GpsMeasurement>;
struct SensorEvent { std::int64_t timestamp_us; Measurement measurement; };
// Save the state just before and after the GPS update. Prediction and any
// same-time gyro/wheel updates have already happened. These are only for logging.
struct GpsEventDiagnostics {
  State predicted_state;
  Covariance predicted_covariance;
  State posterior_state;
  Covariance posterior_covariance;
  std::optional<std::int64_t> since_available_us;
  std::optional<std::int64_t> since_accepted_us;
  std::int64_t rejection_duration_us = 0;
};
struct ReplayResult {
  bool valid = false;
  std::string reason;
  UpdateResult update;
  std::optional<GpsEventDiagnostics> gps;
};
enum class Method { DeadReckoning, Ekf, GatedEkf };

class Replay {
 public:
  Replay(State initial_state, Covariance initial_covariance, Config config, Method method);
  ReplayResult process(const SensorEvent& event);
  const State& state() const { return method_ == Method::DeadReckoning ? dr_state_ : ekf_.state(); }
  const Covariance& covariance() const { return ekf_.covariance(); }
  std::int64_t timestamp_us() const { return timestamp_us_; }
  std::size_t accepted_gps() const { return accepted_gps_; }
  std::size_t rejected_gps() const { return rejected_gps_; }
  std::size_t ignored_gps() const { return ignored_gps_; }

 private:
  Ekf ekf_;
  State dr_state_;
  Method method_;
  std::int64_t timestamp_us_ = 0;
  bool has_event_ = false;
  unsigned sensors_at_timestamp_ = 0;
  std::size_t last_sensor_rank_ = 0;
  std::size_t accepted_gps_ = 0;
  std::size_t rejected_gps_ = 0;
  std::size_t ignored_gps_ = 0;
  std::optional<std::int64_t> last_available_gps_us_;
  std::optional<std::int64_t> last_accepted_gps_us_;
  std::optional<std::int64_t> rejection_start_us_;
};
}  // namespace navtrace
