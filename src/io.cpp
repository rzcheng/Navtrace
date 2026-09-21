#include "navtrace/io.hpp"
#include <charconv>
#include <cmath>
#include <istream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace navtrace {
namespace {
double number(const std::string& text) {
  std::size_t end = 0;
  const double value = std::stod(text, &end);
  if (end != text.size() || !std::isfinite(value)) throw std::invalid_argument("invalid finite number: " + text);
  return value;
}
std::vector<std::string> split(const std::string& text, char separator) {
  std::istringstream input(text);
  std::string field;
  std::vector<std::string> fields;
  while (std::getline(input, field, separator)) fields.push_back(field);
  if (!text.empty() && text.back() == separator) fields.emplace_back();
  return fields;
}
}  // namespace

PriorConfig read_config(std::istream& input) {
  std::map<std::string, double> values;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line.front() == '#') continue;
    const auto fields = split(line, '=');
    if (fields.size() != 2 || !values.emplace(fields[0], number(fields[1])).second) throw std::invalid_argument("malformed/duplicate config entry");
  }
  auto take = [&values](const std::string& key) {
    const auto found = values.find(key);
    if (found == values.end()) throw std::invalid_argument("missing config key: " + key);
    const double value = found->second;
    values.erase(found);
    return value;
  };
  PriorConfig result;
  const std::string names[] = {"px", "py", "yaw", "speed", "yaw_rate", "gyro_bias"};
  result.covariance = Covariance::Zero();
  for (int i = 0; i < kStateSize; ++i) {
    result.state(i) = take("initial_" + names[i]);
    result.covariance(i, i) = take("variance_" + names[i]);
  }
  result.filter.speed_diffusion_m2_per_s3 = take("speed_diffusion_m2_per_s3");
  result.filter.yaw_rate_diffusion_rad2_per_s3 = take("yaw_rate_diffusion_rad2_per_s3");
  result.filter.bias_diffusion_rad2_per_s3 = take("bias_diffusion_rad2_per_s3");
  result.filter.max_step_s = take("max_step_s");
  result.filter.max_gap_s = take("max_gap_s");
  result.filter.gps_nis_threshold = take("gps_nis_threshold");
  if (!values.empty()) throw std::invalid_argument("unknown config key: " + values.begin()->first);
  // Check the starting state and settings before reading sensor events.
  const Ekf validated(result.state, result.covariance, result.filter);
  return result;
}

SensorEvent parse_event(const std::string& line) {
  const auto fields = split(line, ',');
  if (fields.size() != 7) throw std::invalid_argument("sensor CSV requires exactly seven fields");
  std::int64_t timestamp = -1;
  const auto parsed = std::from_chars(fields[0].data(), fields[0].data() + fields[0].size(), timestamp);
  if (parsed.ec != std::errc() || parsed.ptr != fields[0].data() + fields[0].size() || timestamp < 0) throw std::invalid_argument("invalid integer timestamp_us");
  const double z1 = number(fields[2]);
  const double z2 = number(fields[3]);
  const double r11 = number(fields[4]);
  const double r12 = number(fields[5]);
  const double r22 = number(fields[6]);
  SensorEvent event{timestamp, GyroMeasurement{0.0, 1.0}};
  if (fields[1] == "gps") {
    GpsMeasurement gps;
    gps.position_m << z1, z2;
    gps.covariance_m2 << r11, r12, r12, r22;
    event.measurement = gps;
  } else {
    if (z2 != 0.0 || r12 != 0.0 || r22 != 0.0) throw std::invalid_argument("unused scalar CSV fields must be zero");
    if (fields[1] == "gyro") event.measurement = GyroMeasurement{z1, r11};
    else if (fields[1] == "wheel") event.measurement = WheelSpeedMeasurement{z1, r11};
    else throw std::invalid_argument("unknown sensor: " + fields[1]);
  }
  if (!std::visit([](const auto& m) { return valid_measurement(m); }, event.measurement)) throw std::invalid_argument("invalid measurement/covariance");
  return event;
}
}  // namespace navtrace
