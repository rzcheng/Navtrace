#include "navtrace/io.hpp"
#include <Eigen/Eigenvalues>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
  using namespace navtrace;
  try {
    if (argc != 6) throw std::invalid_argument("usage: navtrace_replay INPUT.csv FILTER.conf dr|ekf|gated OUTPUT.csv GPS_DIAGNOSTICS.csv");
    const std::string method_name = argv[3];
    Method method;
    if (method_name == "dr") method = Method::DeadReckoning;
    else if (method_name == "ekf") method = Method::Ekf;
    else if (method_name == "gated") method = Method::GatedEkf;
    else throw std::invalid_argument("method must be dr, ekf, or gated");
    if (std::filesystem::weakly_canonical(argv[4]) == std::filesystem::weakly_canonical(argv[5]) ||
        std::filesystem::exists(argv[4]) || std::filesystem::exists(argv[5])) {
      throw std::invalid_argument("output paths must be distinct and must not already exist");
    }
    std::ifstream config_file(argv[2]);
    if (!config_file) throw std::runtime_error("cannot open filter config");
    const PriorConfig prior = read_config(config_file);
    Replay replay(prior.state, prior.covariance, prior.filter, method);
    std::ifstream sensors(argv[1]);
    std::string line;
    if (!sensors || !std::getline(sensors, line) || line != kSensorHeader) throw std::invalid_argument("missing/invalid sensor CSV header");
    std::ofstream output(argv[4]);
    std::ofstream diagnostics(argv[5]);
    if (!output || !diagnostics) throw std::runtime_error("cannot create output files");
    output << std::setprecision(17);
    diagnostics << std::setprecision(17);
    output << "timestamp_us,sensor,px_m,py_m,yaw_rad,speed_m_per_s,yaw_rate_rad_per_s,gyro_bias_rad_per_s";
    for (int row = 0; row < 6; ++row) for (int col = 0; col < 6; ++col) output << ",p" << row << col;
    output << ",covariance_min_eigenvalue,covariance_symmetry_error\n";
    diagnostics << "timestamp_us,status,residual_x_m,residual_y_m,s00,s01,s11,nis,threshold,reason";
    for (const char* prefix : {"pre", "post"}) {
      for (int i = 0; i < 6; ++i) diagnostics << ',' << prefix << "_x" << i;
      for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) diagnostics << ',' << prefix << "_p" << i << j;
    }
    diagnostics << ",z0,z1,r00,r01,r11,since_available_s,since_accepted_s,rejection_duration_s";
    for (int i = 0; i < 6; ++i) diagnostics << ",delta_x" << i;
    diagnostics << '\n';
    const char* sensor_names[] = {"gyro", "wheel", "gps"};
    std::size_t line_number = 1;
    std::size_t events = 0;
    while (std::getline(sensors, line)) {
      ++line_number;
      try {
        const SensorEvent event = parse_event(line);
        const ReplayResult result = replay.process(event);
        if (!result.valid) throw std::invalid_argument(result.reason);
        ++events;
        output << event.timestamp_us << ',' << sensor_names[event.measurement.index()];
        for (int i = 0; i < 6; ++i) output << ',' << replay.state()(i);
        if (method == Method::DeadReckoning) {
          for (int i = 0; i < 38; ++i) output << ',';  // No fictitious DR covariance.
        } else {
          const auto& p = replay.covariance();
          for (int row = 0; row < 6; ++row) for (int col = 0; col < 6; ++col) output << ',' << p(row, col);
          Eigen::SelfAdjointEigenSolver<Covariance> eigenvalues(p, Eigen::EigenvaluesOnly);
          if (eigenvalues.info() != Eigen::Success) throw std::runtime_error("output covariance eigensolve failed");
          output << ',' << eigenvalues.eigenvalues().minCoeff() << ',' << (p - p.transpose()).cwiseAbs().maxCoeff();
        }
        output << '\n';
        if (std::holds_alternative<GpsMeasurement>(event.measurement)) {
          const auto& u = result.update;
          if (method == Method::DeadReckoning) diagnostics << event.timestamp_us << ",ignored,,,,,,,,GPS ignored by dead reckoning";
          else diagnostics << event.timestamp_us << ',' << (u.status == UpdateStatus::Accepted ? "accepted" : "rejected")
            << ',' << u.residual_m(0) << ',' << u.residual_m(1) << ',' << u.innovation_covariance_m2(0, 0)
            << ',' << u.innovation_covariance_m2(0, 1) << ',' << u.innovation_covariance_m2(1, 1)
            << ',' << u.nis << ',' << u.threshold << ',' << u.reason;
          if (result.gps) {
            const auto& gps = *result.gps;
            for (int i = 0; i < 6; ++i) diagnostics << ',' << gps.predicted_state(i);
            for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) diagnostics << ',' << gps.predicted_covariance(i, j);
            for (int i = 0; i < 6; ++i) diagnostics << ',' << gps.posterior_state(i);
            for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) diagnostics << ',' << gps.posterior_covariance(i, j);
            const auto& measurement = std::get<GpsMeasurement>(event.measurement);
            diagnostics << ',' << measurement.position_m(0) << ',' << measurement.position_m(1)
              << ',' << measurement.covariance_m2(0, 0) << ',' << measurement.covariance_m2(0, 1)
              << ',' << measurement.covariance_m2(1, 1) << ',';
            if (gps.since_available_us) diagnostics << static_cast<double>(*gps.since_available_us) * 1e-6;
            diagnostics << ',';
            if (gps.since_accepted_us) diagnostics << static_cast<double>(*gps.since_accepted_us) * 1e-6;
            diagnostics << ',' << static_cast<double>(gps.rejection_duration_us) * 1e-6;
            State change = gps.posterior_state - gps.predicted_state;
            change(YAW) = wrap_angle(change(YAW));
            for (int i = 0; i < 6; ++i) diagnostics << ',' << change(i);
          } else {
            for (int i = 0; i < 98; ++i) diagnostics << ',';
          }
          diagnostics << '\n';
        }
      } catch (const std::exception& error) {
        throw std::runtime_error("sensor line " + std::to_string(line_number) + ": " + error.what());
      }
    }
    if (sensors.bad() || events == 0) throw std::runtime_error("empty sensor stream or read failure");
    output.flush();
    diagnostics.flush();
    if (!output || !diagnostics) throw std::runtime_error("output write failure");
    std::cout << method_name << ": " << events << " events; GPS accepted=" << replay.accepted_gps()
              << " rejected=" << replay.rejected_gps() << " ignored=" << replay.ignored_gps() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "navtrace_replay: " << error.what() << '\n';
    return 1;
  }
}
