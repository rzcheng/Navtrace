// Offline analysis helper: nominal trajectories never enter navtrace_replay.
#include "navtrace/ekf.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr const char* kHeader =
    "timestamp_us,px_m,py_m,yaw_rad,speed_m_per_s,yaw_rate_rad_per_s,gyro_bias_rad_per_s";
struct NominalSample {
  std::int64_t timestamp_us;
  navtrace::State state;
};

std::vector<NominalSample> read_nominal(const std::filesystem::path& input) {
  std::ifstream stream(input);
  if (!stream) throw std::runtime_error("cannot open nominal input");
  std::string line;
  if (!std::getline(stream, line) || line != kHeader) {
    throw std::runtime_error("unexpected nominal CSV header");
  }
  std::vector<NominalSample> samples;
  std::size_t line_number = 1;
  while (std::getline(stream, line)) {
    ++line_number;
    try {
      std::istringstream fields(line);
      std::string field;
      if (!std::getline(fields, field, ',') || field.empty() ||
          field.find_first_not_of("0123456789") != std::string::npos) {
        throw std::runtime_error("timestamp must be a nonnegative integer");
      }
      NominalSample sample;
      sample.timestamp_us = std::stoll(field);
      const std::int64_t expected_us = static_cast<std::int64_t>(samples.size()) * 10000;
      if (sample.timestamp_us != expected_us || sample.timestamp_us > 60000000) {
        throw std::runtime_error("nominal grid must start at zero, use 10000 us steps, and end by 60 s");
      }
      for (int column = 0; column < navtrace::kStateSize; ++column) {
        if (!std::getline(fields, field, ',') || field.empty()) {
          throw std::runtime_error("missing state field");
        }
        std::size_t consumed = 0;
        sample.state(column) = std::stod(field, &consumed);
        if (consumed != field.size()) throw std::runtime_error("invalid state number");
      }
      if (std::getline(fields, field, ',') || (!line.empty() && line.back() == ',')) {
        throw std::runtime_error("extra state field");
      }
      if (!sample.state.allFinite()) throw std::runtime_error("nonfinite nominal state");
      samples.push_back(sample);
    } catch (const std::exception& error) {
      throw std::runtime_error("nominal line " + std::to_string(line_number) + ": " + error.what());
    }
  }
  if (stream.bad()) throw std::runtime_error("nominal input read failed");
  if (samples.empty()) throw std::runtime_error("nominal input has no samples");
  return samples;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3) throw std::runtime_error("usage: navtrace_linearize NOMINAL.csv JACOBIANS.csv");
    const std::filesystem::path input(argv[1]);
    const std::filesystem::path output(argv[2]);
    if (std::filesystem::exists(output) || std::filesystem::is_symlink(output)) {
      throw std::runtime_error("refusing to overwrite output");
    }
    // Validate the complete nominal input before creating any output.
    const auto samples = read_nominal(input);
    std::ofstream stream(output);
    if (!stream) throw std::runtime_error("cannot create Jacobian output");
    stream << "timestamp_us";
    for (int row = 0; row < navtrace::kStateSize; ++row) {
      for (int column = 0; column < navtrace::kStateSize; ++column) {
        stream << ",f" << row << column;
      }
    }
    stream << '\n' << std::setprecision(17);
    for (std::size_t index = 0; index < samples.size(); ++index) {
      navtrace::Covariance transition = navtrace::Covariance::Identity();
      if (index > 0) {
        const double dt_s = static_cast<double>(samples[index].timestamp_us -
                                               samples[index - 1].timestamp_us) / 1e6;
        transition = navtrace::transition_jacobian(samples[index - 1].state, dt_s);
      }
      if (!transition.allFinite()) throw std::runtime_error("nonfinite transition Jacobian");
      stream << samples[index].timestamp_us;
      for (int row = 0; row < navtrace::kStateSize; ++row) {
        for (int column = 0; column < navtrace::kStateSize; ++column) {
          stream << ',' << transition(row, column);
        }
      }
      stream << '\n';
    }
    stream.flush();
    if (!stream) throw std::runtime_error("Jacobian output write failed");
    stream.close();
    if (!stream) throw std::runtime_error("Jacobian output close failed");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
