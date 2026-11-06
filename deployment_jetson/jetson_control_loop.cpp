// =============================================================================
// jetson_control_loop.cpp - BB-8 real-time closed-loop controller (Jetson Nano)
// =============================================================================
// Sensors : Arduino Nano 33 IoT IMU (I2C slave 0x42) + 3x AS5600 encoders behind a
//           TCA9548A multiplexer (0x70) on /dev/i2c-1 (40-pin header pins 3/5).
// Policy  : default  -> exported zero-dependency bb8_policy_inference.cpp (compiled in)
//           -DUSE_ONNXRUNTIME=ON -> ONNX Runtime C++ API on best_policy.onnx
// Motors  : default  -> TMC2208 UART VACTUAL velocity mode (driver's internal step generator)
//           -DUSE_STEP_DIR=ON -> STEP/DIR pulses from a SCHED_FIFO DDS thread via libgpiod
//
// Build:   mkdir build && cd build && cmake .. -DPOLICY_DIR=/path/to/export && make -j4
// Run:     sudo ./bb8_jetson_runner ../robot_hw_config.json [seconds]
// =============================================================================
#include <atomic>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <linux/i2c.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#ifdef USE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#else
#include "bb8_policy_inference.h"
#endif
#ifdef USE_STEP_DIR
#include <gpiod.h>
#endif

static std::atomic<bool> g_run{true};
static void on_signal(int) { g_run = false; }

// ----------------------------------------------------------------------------
// Minimal flat-JSON reader (numbers, strings, arrays of either)
// ----------------------------------------------------------------------------
class FlatJson {
 public:
  explicit FlatJson(const std::string& path) {
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    std::stringstream ss; ss << f.rdbuf(); txt_ = ss.str();
  }
  double num(const std::string& k, double def = 0) const {
    size_t p = find(k); if (p == std::string::npos) return def;
    return std::strtod(txt_.c_str() + p, nullptr);
  }
  std::string str(const std::string& k, const std::string& def = "") const {
    size_t p = find(k); if (p == std::string::npos || txt_[p] != '"') return def;
    size_t e = txt_.find('"', p + 1); return txt_.substr(p + 1, e - p - 1);
  }
  std::vector<double> nums(const std::string& k) const {
    std::vector<double> v; size_t p = find(k); if (p == std::string::npos) return v;
    size_t e = txt_.find(']', p); std::string body = txt_.substr(p + 1, e - p - 1);
    std::stringstream ss(body); std::string item;
    while (std::getline(ss, item, ',')) v.push_back(std::strtod(item.c_str(), nullptr));
    return v;
  }
  std::vector<std::string> strs(const std::string& k) const {
    std::vector<std::string> v; size_t p = find(k); if (p == std::string::npos) return v;
    size_t e = txt_.find(']', p);
    for (size_t q = txt_.find('"', p); q < e; q = txt_.find('"', q + 1)) {
      size_t r = txt_.find('"', q + 1); v.push_back(txt_.substr(q + 1, r - q - 1)); q = r;
    }
    return v;
  }
 private:
  size_t find(const std::string& k) const {
    size_t p = txt_.find("\"" + k + "\""); if (p == std::string::npos) return p;
    p = txt_.find(':', p) + 1;
    while (p < txt_.size() && (txt_[p] == ' ' || txt_[p] == '\n' || txt_[p] == '\t')) ++p;
    return p;
  }
  std::string txt_;
};

// ----------------------------------------------------------------------------
// I2C helpers
// ----------------------------------------------------------------------------
class I2CBus {
 public:
  explicit I2CBus(int bus) {
    char dev[32]; snprintf(dev, sizeof dev, "/dev/i2c-%d", bus);
    fd_ = open(dev, O_RDWR);
    if (fd_ < 0) { perror(dev); exit(1); }
  }
  ~I2CBus() { if (fd_ >= 0) close(fd_); }
  bool read(uint8_t addr, uint8_t* buf, uint16_t n) {
    i2c_msg m{addr, I2C_M_RD, n, buf}; i2c_rdwr_ioctl_data d{&m, 1};
    return ioctl(fd_, I2C_RDWR, &d) >= 0;
  }
  bool write(uint8_t addr, const uint8_t* buf, uint16_t n) {
    i2c_msg m{addr, 0, n, const_cast<uint8_t*>(buf)}; i2c_rdwr_ioctl_data d{&m, 1};
    return ioctl(fd_, I2C_RDWR, &d) >= 0;
  }
  bool write_read(uint8_t addr, uint8_t reg, uint8_t* buf, uint16_t n) {
    i2c_msg m[2] = {{addr, 0, 1, &reg}, {addr, I2C_M_RD, n, buf}}; i2c_rdwr_ioctl_data d{m, 2};
    return ioctl(fd_, I2C_RDWR, &d) >= 0;
  }
 private:
  int fd_ = -1;
};

struct ImuSample { float rpy[3], gyro[3], acc[3]; bool ok; uint8_t seq; };

static bool read_imu(I2CBus& bus, uint8_t addr, ImuSample& s) {
  uint8_t b[24];
  if (!bus.read(addr, b, 24)) return false;
  uint8_t c = 0; for (int i = 0; i < 23; ++i) c ^= b[i];
  if (b[0] != 0xB8 || c != b[23]) return false;
  int16_t v[9]; memcpy(v, b + 2, sizeof v);  // little-endian on both ends
  for (int i = 0; i < 3; ++i) {
    s.rpy[i] = v[i] * 1e-4f; s.gyro[i] = v[3 + i] * 1e-3f; s.acc[i] = v[6 + i] * 1e-3f;
  }
  s.ok = (b[20] & 1) != 0; s.seq = b[1];
  return true;
}

static bool read_encoders(I2CBus& bus, uint8_t mux, const std::vector<double>& ch, uint8_t addr,
                          const std::vector<double>& sign, float out[3]) {
  for (int i = 0; i < 3; ++i) {
    uint8_t sel = (uint8_t)(1u << (int)ch[i]);
    uint8_t raw[2];
    if (!bus.write(mux, &sel, 1) || !bus.write_read(addr, 0x0C, raw, 2)) return false;
    out[i] = (float)sign[i] * (float)(((raw[0] & 0x0F) << 8) | raw[1]) * (2.f * (float)M_PI / 4096.f);
  }
  return true;
}

static float wrap_pi(float a) { a = std::fmod(a + (float)M_PI, 2.f * (float)M_PI); if (a < 0) a += 2.f * (float)M_PI; return a - (float)M_PI; }

// ----------------------------------------------------------------------------
// TMC2208 UART (VACTUAL velocity mode)
// ----------------------------------------------------------------------------
class TMC2208 {
 public:
  TMC2208(const std::string& port, int baud, double fclk, int microsteps, int irun, int ihold) : fclk_(fclk) {
    fd_ = open(port.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
    if (fd_ < 0) { perror(port.c_str()); exit(1); }
    termios t{}; tcgetattr(fd_, &t); cfmakeraw(&t);
    speed_t sp = baud == 230400 ? B230400 : baud == 57600 ? B57600 : B115200;
    cfsetispeed(&t, sp); cfsetospeed(&t, sp);
    t.c_cflag |= (CLOCAL | CREAD); t.c_cflag &= ~CSTOPB; t.c_cc[VMIN] = 0; t.c_cc[VTIME] = 0;
    tcsetattr(fd_, TCSANOW, &t);
    int mres = 0; for (int m = 256; m > microsteps && mres < 8; m >>= 1) ++mres;
    write_reg(0x00, (1u << 2) | (1u << 6) | (1u << 7) | (1u << 8));        // GCONF: spreadCycle, UART, MRES reg
    write_reg(0x6C, (0x10000053u & ~(0xFu << 24)) | ((uint32_t)mres << 24));  // CHOPCONF
    write_reg(0x10, (uint32_t)(ihold & 31) | ((uint32_t)(irun & 31) << 8) | (4u << 16));  // IHOLD_IRUN
    set_velocity(0);
  }
  ~TMC2208() { set_velocity(0); if (fd_ >= 0) close(fd_); }
  void set_velocity(double usteps_per_s) {
    double v = usteps_per_s * 16777216.0 / fclk_;
    long iv = std::lround(std::fmax(-8388607.0, std::fmin(8388607.0, v)));
    write_reg(0x22, (uint32_t)iv & 0xFFFFFFu);
  }
 private:
  static uint8_t crc8(const uint8_t* d, int n) {
    uint8_t crc = 0;
    for (int i = 0; i < n; ++i) { uint8_t b = d[i];
      for (int j = 0; j < 8; ++j) { crc = ((crc >> 7) ^ (b & 1)) ? (uint8_t)((crc << 1) ^ 0x07) : (uint8_t)(crc << 1); b >>= 1; } }
    return crc;
  }
  void write_reg(uint8_t reg, uint32_t val) {
    uint8_t d[8] = {0x05, 0x00, (uint8_t)(reg | 0x80), (uint8_t)(val >> 24), (uint8_t)(val >> 16),
                    (uint8_t)(val >> 8), (uint8_t)val, 0};
    d[7] = crc8(d, 7);
    if (::write(fd_, d, 8) != 8) perror("tmc write");
    tcdrain(fd_); tcflush(fd_, TCIFLUSH);   // single-wire UART: drop our own echo
  }
  int fd_ = -1; double fclk_;
};

#ifdef USE_STEP_DIR
// ----------------------------------------------------------------------------
// STEP/DIR software DDS (20 kHz SCHED_FIFO thread). Max rate = 10 k microsteps/s
// per motor -> use 8 microsteps for the full 45 rad/s range, or prefer UART mode.
// ----------------------------------------------------------------------------
class StepDirGenerator {
 public:
  StepDirGenerator(const std::string& chip, const std::vector<double>& step, const std::vector<double>& dir, int en) {
    chip_ = gpiod_chip_open_by_name(chip.c_str());
    if (!chip_) { perror("gpiod_chip_open_by_name"); exit(1); }
    for (int i = 0; i < 3; ++i) {
      st_[i] = gpiod_chip_get_line(chip_, (unsigned)step[i]); dr_[i] = gpiod_chip_get_line(chip_, (unsigned)dir[i]);
      gpiod_line_request_output(st_[i], "bb8_step", 0); gpiod_line_request_output(dr_[i], "bb8_dir", 0);
    }
    en_ = gpiod_chip_get_line(chip_, (unsigned)en); gpiod_line_request_output(en_, "bb8_en", 0);  // ENN low = enabled
    th_ = std::thread([this] { loop(); });
  }
  ~StepDirGenerator() {
    stop_ = true; th_.join(); gpiod_line_set_value(en_, 1); gpiod_chip_close(chip_);
  }
  void set_rates(const float r[3]) { for (int i = 0; i < 3; ++i) rate_[i].store(r[i]); }
 private:
  void loop() {
    sched_param sp{}; sp.sched_priority = 90; pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);
    const double f = 20000.0; const long period_ns = (long)(1e9 / f);
    double phase[3] = {0, 0, 0}; bool high[3] = {false, false, false}; int dir_state[3] = {0, 0, 0};
    timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    while (!stop_) {
      for (int i = 0; i < 3; ++i) {
        if (high[i]) { gpiod_line_set_value(st_[i], 0); high[i] = false; continue; }
        float r = rate_[i].load(); int d = r < 0 ? 1 : 0;
        if (d != dir_state[i]) { gpiod_line_set_value(dr_[i], d); dir_state[i] = d; continue; }  // DIR setup time
        phase[i] += std::fabs(r) / f;
        if (phase[i] >= 1.0) { phase[i] -= 1.0; gpiod_line_set_value(st_[i], 1); high[i] = true; }
      }
      t.tv_nsec += period_ns; if (t.tv_nsec >= 1000000000L) { t.tv_nsec -= 1000000000L; ++t.tv_sec; }
      clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &t, nullptr);
    }
  }
  gpiod_chip* chip_; gpiod_line* st_[3]; gpiod_line* dr_[3]; gpiod_line* en_;
  std::atomic<float> rate_[3]{{0.f}, {0.f}, {0.f}}; std::atomic<bool> stop_{false}; std::thread th_;
};
#endif

// ----------------------------------------------------------------------------
// Odometry (no-slip): wheel_speeds = J * w_rel  (see mjcf_generator.omniwheel_jacobian)
// ----------------------------------------------------------------------------
struct Odometry {
  double Jinv[3][3]; double R; double x = 0, y = 0;
  Odometry(double ball_r, double wheel_r, double zenith_deg, const std::vector<double>& az) : R(ball_r) {
    double J[3][3]; double th = zenith_deg * M_PI / 180.0;
    for (int i = 0; i < 3; ++i) { double a = az[i] * M_PI / 180.0;
      J[i][0] = -(ball_r / wheel_r) * cos(th) * cos(a); J[i][1] = -(ball_r / wheel_r) * cos(th) * sin(a);
      J[i][2] = (ball_r / wheel_r) * sin(th); }
    double det = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) - J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) +
                 J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);
    for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) {
      int r1 = (c + 1) % 3, r2 = (c + 2) % 3, c1 = (r + 1) % 3, c2 = (r + 2) % 3;
      Jinv[r][c] = (J[r1][c1] * J[r2][c2] - J[r1][c2] * J[r2][c1]) / det; }
  }
  void update(const float w[3], const float gyro[3], float yaw, double dt) {
    double wr[3]; for (int i = 0; i < 3; ++i) wr[i] = Jinv[i][0] * w[0] + Jinv[i][1] * w[1] + Jinv[i][2] * w[2];
    double wbx = gyro[0] - wr[0], wby = gyro[1] - wr[1];
    double vx = wby * R, vy = -wbx * R, c = cos(yaw), s = sin(yaw);
    x += dt * (c * vx - s * vy); y += dt * (s * vx + c * vy);
  }
};

#ifdef USE_ONNXRUNTIME
static const char* kAllObs[20] = {"imu_roll_rad", "imu_pitch_rad", "imu_yaw_rad", "imu_gx_rads", "imu_gy_rads",
  "imu_gz_rads", "imu_ax_ms2", "imu_ay_ms2", "imu_az_ms2", "enc1_vel_rads", "enc2_vel_rads", "enc3_vel_rads",
  "enc1_pos_rad", "enc2_pos_rad", "enc3_pos_rad", "target_dx_m", "target_dy_m", "prev_motor1_cmd",
  "prev_motor2_cmd", "prev_motor3_cmd"};
#endif

int main(int argc, char** argv) {
  const std::string cfg_path = argc > 1 ? argv[1] : "robot_hw_config.json";
  const double run_seconds = argc > 2 ? atof(argv[2]) : 0.0;
  FlatJson cfg(cfg_path);
  signal(SIGINT, on_signal); signal(SIGTERM, on_signal);
  mlockall(MCL_CURRENT | MCL_FUTURE);
  sched_param sp{}; sp.sched_priority = 80;
  if (sched_setscheduler(0, SCHED_FIFO, &sp) != 0) fprintf(stderr, "warning: no SCHED_FIFO (run with sudo)\n");

  const double hz = cfg.num("control_hz", 100), dt = 1.0 / hz;
  const double max_w = cfg.num("max_wheel_speed_rads", 45);
  const double steps_per_rad = cfg.num("full_steps_per_rev", 200) * cfg.num("microsteps", 16) / (2 * M_PI);
  const double cutoff = cfg.num("tilt_cutoff_deg", 35) * M_PI / 180.0;
  const double deadband = cfg.num("deadband_rads", 0), alpha = cfg.num("encoder_velocity_lpf_alpha", 0.5);
  const bool target_on = cfg.num("target_enabled", 0) != 0;
  const double tx = cfg.num("target_x_m"), ty = cfg.num("target_y_m");
  const auto msign = cfg.nums("motor_direction_sign");

  I2CBus bus((int)cfg.num("i2c_bus", 1));
  const uint8_t imu_addr = (uint8_t)cfg.num("imu_i2c_addr", 0x42);
  const uint8_t mux = (uint8_t)cfg.num("encoder_mux_addr", 0x70), enc_addr = (uint8_t)cfg.num("encoder_i2c_addr", 0x36);
  const auto mux_ch = cfg.nums("encoder_mux_channels"); const auto esign = cfg.nums("encoder_direction_sign");
  { uint8_t c = 0x01; bus.write(imu_addr, &c, 1); usleep(1300000); c = 0x02; bus.write(imu_addr, &c, 1); }  // gyro calib

#ifdef USE_STEP_DIR
  StepDirGenerator motors(cfg.str("gpio_chip", "gpiochip0"), cfg.nums("step_lines"), cfg.nums("dir_lines"),
                          (int)cfg.num("enable_line"));
#else
  std::vector<TMC2208*> tmc;
  for (const auto& p : cfg.strs("tmc_uart_ports"))
    tmc.push_back(new TMC2208(p, (int)cfg.num("tmc_uart_baud", 115200), cfg.num("tmc_fclk_hz", 12e6),
                              (int)cfg.num("microsteps", 16), (int)cfg.num("tmc_irun", 20), (int)cfg.num("tmc_ihold", 10)));
#endif

#ifdef USE_ONNXRUNTIME
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "bb8");
  Ort::SessionOptions so; so.SetIntraOpNumThreads(1);
  Ort::Session session(env, cfg.str("policy_onnx", "best_policy.onnx").c_str(), so);
  Ort::AllocatorWithDefaultOptions alloc;
  auto meta = session.GetModelMetadata();
#if ORT_API_VERSION >= 13
  std::string obs_json = meta.LookupCustomMetadataMapAllocated("obs_names", alloc).get();
  std::string act_json = meta.LookupCustomMetadataMapAllocated("action_names", alloc).get();
#else
  std::string obs_json = meta.LookupCustomMetadataMap("obs_names", alloc);
  std::string act_json = meta.LookupCustomMetadataMap("action_names", alloc);
#endif
  std::vector<int> obs_idx, act_idx;
  for (size_t q = obs_json.find('"'); q != std::string::npos; q = obs_json.find('"', q + 1)) {
    size_t r = obs_json.find('"', q + 1); std::string n = obs_json.substr(q + 1, r - q - 1); q = r;
    for (int i = 0; i < 20; ++i) if (n == kAllObs[i]) obs_idx.push_back(i);
  }
  for (size_t q = act_json.find('"'); q != std::string::npos; q = act_json.find('"', q + 1)) {
    size_t r = act_json.find('"', q + 1); act_idx.push_back(act_json[r - 1] - '1'); q = r;
  }
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  const char* in_names[] = {"obs_raw"}; const char* out_names[] = {"action"};
#endif

  Odometry odo(cfg.num("ball_radius_m"), cfg.num("wheel_radius_m"), cfg.num("zenith_angle_deg"),
               cfg.nums("wheel_azimuths_deg"));
  float prev_ang[3] = {0, 0, 0}, unwrapped[3] = {0, 0, 0}, enc_vel[3] = {0, 0, 0}, prev_cmd[3] = {0, 0, 0};
  bool first = true; long k = 0, overruns = 0, io_errors = 0;
  FILE* log = fopen(cfg.str("log_csv", "jetson_run_log.csv").c_str(), "w");
  if (log) fprintf(log, "t,loop_us,roll,pitch,yaw,gx,gy,gz,enc1,enc2,enc3,cmd1,cmd2,cmd3,odo_x,odo_y\n");
  timespec next; clock_gettime(CLOCK_MONOTONIC, &next);
  timespec t0 = next;

  while (g_run) {
    timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    ImuSample imu{}; float ang[3];
    if (!read_imu(bus, imu_addr, imu) || !read_encoders(bus, mux, mux_ch, enc_addr, esign, ang)) {
      if (++io_errors > 5) { fprintf(stderr, "I2C failure - stopping\n"); break; }
      continue;
    }
    io_errors = 0;
    float enc_pos[3];
    for (int i = 0; i < 3; ++i) {
      if (first) { prev_ang[i] = unwrapped[i] = ang[i]; }
      float d = wrap_pi(ang[i] - prev_ang[i]); prev_ang[i] = ang[i]; unwrapped[i] += d;
      enc_vel[i] = first ? 0.f : (float)(alpha * d / dt + (1 - alpha) * enc_vel[i]);
      enc_pos[i] = wrap_pi(unwrapped[i]);
    }
    first = false;
    odo.update(enc_vel, imu.gyro, imu.rpy[2], dt);
    float tdx = 0, tdy = 0;
    if (target_on) {
      double ex = tx - odo.x, ey = ty - odo.y, n = std::hypot(ex, ey);
      if (n > 2.0) { ex *= 2.0 / n; ey *= 2.0 / n; }
      double c = cos(imu.rpy[2]), s = sin(imu.rpy[2]);
      tdx = (float)(c * ex + s * ey); tdy = (float)(-s * ex + c * ey);
    }
    float w[3] = {0, 0, 0}, a[3] = {0, 0, 0};
#ifdef USE_ONNXRUNTIME
    float full[20] = {imu.rpy[0], imu.rpy[1], imu.rpy[2], imu.gyro[0], imu.gyro[1], imu.gyro[2], imu.acc[0],
                      imu.acc[1], imu.acc[2], enc_vel[0], enc_vel[1], enc_vel[2], enc_pos[0], enc_pos[1], enc_pos[2],
                      tdx, tdy, prev_cmd[0], prev_cmd[1], prev_cmd[2]};
    std::vector<float> obs; for (int i : obs_idx) obs.push_back(full[i]);
    int64_t shape[2] = {1, (int64_t)obs.size()};
    Ort::Value in = Ort::Value::CreateTensor<float>(mem, obs.data(), obs.size(), shape, 2);
    auto out = session.Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 1);
    const float* o = out[0].GetTensorData<float>();
    for (size_t j = 0; j < act_idx.size(); ++j) { a[act_idx[j]] = std::fmax(-1.f, std::fmin(1.f, o[j])); w[act_idx[j]] = a[act_idx[j]] * (float)max_w; }
#else
    float sr[3];
    bb8::compute_stepper_commands(imu.rpy[0], imu.rpy[1], imu.rpy[2], imu.gyro[0], imu.gyro[1], imu.gyro[2],
                                  imu.acc[0], imu.acc[1], imu.acc[2], enc_vel[0], enc_vel[1], enc_vel[2],
                                  enc_pos[0], enc_pos[1], enc_pos[2], tdx, tdy, prev_cmd[0], prev_cmd[1], prev_cmd[2],
                                  &sr[0], &sr[1], &sr[2], &w[0], &w[1], &w[2]);
    for (int i = 0; i < 3; ++i) a[i] = (float)(w[i] / max_w);
#endif
    const double tilt = std::acos(std::fmax(-1.0, std::fmin(1.0, cos(imu.rpy[0]) * cos(imu.rpy[1]))));
    for (int i = 0; i < 3; ++i) if (std::fabs(w[i]) < deadband) w[i] = 0;
    if (tilt > cutoff || !imu.ok) {
      fprintf(stderr, "SAFETY STOP tilt=%.1f deg imu_ok=%d\n", tilt * 180 / M_PI, imu.ok);
      break;
    }
    float rates[3];
    for (int i = 0; i < 3; ++i) rates[i] = (float)(msign[i] * w[i] * steps_per_rad);
#ifdef USE_STEP_DIR
    motors.set_rates(rates);
#else
    for (int i = 0; i < 3 && i < (int)tmc.size(); ++i) tmc[i]->set_velocity(rates[i]);
#endif
    for (int i = 0; i < 3; ++i) prev_cmd[i] = a[i];

    timespec te; clock_gettime(CLOCK_MONOTONIC, &te);
    const double loop_us = (te.tv_sec - ts.tv_sec) * 1e6 + (te.tv_nsec - ts.tv_nsec) / 1e3;
    const double t = (ts.tv_sec - t0.tv_sec) + (ts.tv_nsec - t0.tv_nsec) / 1e9;
    if (log) fprintf(log, "%.4f,%.1f,%.5f,%.5f,%.5f,%.4f,%.4f,%.4f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.4f,%.4f\n", t, loop_us,
                     imu.rpy[0], imu.rpy[1], imu.rpy[2], imu.gyro[0], imu.gyro[1], imu.gyro[2], enc_vel[0], enc_vel[1],
                     enc_vel[2], w[0], w[1], w[2], odo.x, odo.y);
    if (++k % (long)hz == 0)
      printf("t=%6.1fs tilt=%5.2f deg loop=%6.0f us odo=(%+.2f,%+.2f) overruns=%ld\n", t, tilt * 180 / M_PI, loop_us,
             odo.x, odo.y, overruns);
    if (run_seconds > 0 && t > run_seconds) break;
    next.tv_nsec += (long)(dt * 1e9);
    while (next.tv_nsec >= 1000000000L) { next.tv_nsec -= 1000000000L; ++next.tv_sec; }
    timespec now; clock_gettime(CLOCK_MONOTONIC, &now);
    if (now.tv_sec > next.tv_sec || (now.tv_sec == next.tv_sec && now.tv_nsec > next.tv_nsec)) { ++overruns; next = now; }
    else clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, nullptr);
  }
#ifndef USE_STEP_DIR
  for (auto* d : tmc) delete d;  // destructor sets VACTUAL = 0
#endif
  if (log) fclose(log);
  printf("stopped after %ld cycles (%ld overruns)\n", k, overruns);
  return 0;
}
