#pragma once

#include "navtrace/replay.hpp"
#include <iosfwd>

namespace navtrace {
struct PriorConfig { State state; Covariance covariance; Config filter; };
PriorConfig read_config(std::istream& input);
SensorEvent parse_event(const std::string& line);
constexpr const char* kSensorHeader = "timestamp_us,sensor,value1,value2,cov11,cov12,cov22";
}  // namespace navtrace
