/*
 * BB-8 head IMU node - Arduino Nano 33 IoT (SAMD21 + LSM6DS3)
 * ============================================================
 * Reads the on-board 6-DoF LSM6DS3 IMU at ~416 Hz, runs a Mahony AHRS filter
 * (no magnetometer -> yaw is gyro-integrated and drifts slowly, exactly like the
 * simulator's IMU model), and serves the latest sample to the Jetson Nano as an
 * I2C *slave* at address 0x42.
 *
 * WHY A SECOND I2C BUS: on the Nano 33 IoT the LSM6DS3 sits on the main Wire bus
 * (SERCOM4, pins A4/A5) where the SAMD21 must be the bus MASTER. The link to the
 * Jetson therefore uses a second hardware I2C (SERCOM1) on D11 = SDA, D13 = SCL
 * where the SAMD21 is a SLAVE. (D11/D13 are the SPI header pins; SPI is not used.)
 *
 * Wiring (3.3 V logic only - the Nano 33 IoT is NOT 5 V tolerant):
 *   Nano D11 (SDA, SERCOM1 PAD0)  ->  Jetson 40-pin header pin 3  (I2C1_SDA, /dev/i2c-1)
 *   Nano D13 (SCL, SERCOM1 PAD1)  ->  Jetson 40-pin header pin 5  (I2C1_SCL)
 *   Nano GND                      ->  Jetson pin 6 (GND)
 *   Nano VIN (5 V) or USB         ->  Jetson pin 2/4 (5 V)  [or separate 5 V rail]
 *   The Jetson carrier already has 2.2 kOhm pull-ups to 3.3 V on pins 3/5.
 *
 * Packet returned on every I2C read (24 bytes, little-endian):
 *   [0]     magic 0xB8
 *   [1]     sequence counter (uint8, increments per IMU sample)
 *   [2..7]  roll, pitch, yaw         int16, rad * 10000
 *   [8..13] gx, gy, gz               int16, rad/s * 1000   (head frame, x fwd, y left, z up)
 *   [14..19] ax, ay, az              int16, m/s^2 * 1000   (specific force, +9.81 on z when level)
 *   [20..21] status flags            uint16 (bit0 imu_ok, bit1 calibrated, bit2 saturated)
 *   [22]    sample age               uint8, ms since the sample was taken (clamped 255)
 *   [23]    XOR checksum of bytes 0..22
 *
 * Commands written by the Jetson (first byte = command):
 *   0x01  re-calibrate gyro bias (robot must be still, ~1 s)
 *   0x02  zero yaw
 *   0x03 <kp:float32> <ki:float32>  set Mahony gains
 *
 * Libraries: "Arduino_LSM6DS3" (Library Manager). Board: "Arduino SAMD Boards" -> Nano 33 IoT.
 */
#include <Arduino.h>
#include <Wire.h>
#include "wiring_private.h"   // pinPeripheral()
#include <Arduino_LSM6DS3.h>

// ------------------------------------------------------------------ settings
static const uint8_t  kJetsonI2CAddr = 0x42;
static const float    kG             = 9.80665f;
static const float    kDeg2Rad       = 0.01745329252f;
// Mounting: rotate the IMU frame into the BB-8 head frame (x forward/eye, y left, z up).
// Each output axis = sign * input axis index. Default: board flat, USB pointing backwards.
static const int8_t   kAxisMap[3]    = {0, 1, 2};
static const float    kAxisSign[3]   = {1.f, 1.f, 1.f};

// ------------------------------------------------------------- second I2C bus
TwoWire jetsonWire(&sercom1, 11, 13);   // SDA = D11 (PA16), SCL = D13 (PA17)
extern "C" void SERCOM1_Handler(void) { jetsonWire.onService(); }

// ------------------------------------------------------------------ AHRS state
static float q0 = 1.f, q1 = 0.f, q2 = 0.f, q3 = 0.f;
static float ix = 0.f, iy = 0.f, iz = 0.f;          // Mahony integral feedback
static float kp = 2.0f, ki = 0.02f;
static float gbias[3] = {0, 0, 0};
static volatile bool calibrateRequest = true, zeroYawRequest = false;
static float yawOffset = 0.f;

// --------------------------------------------------------- shared packet buffer
static volatile uint8_t txPacket[24];
static uint8_t seqCounter = 0;
static uint32_t sampleMillis = 0;
static uint16_t statusFlags = 0;

static int16_t clamp16(float v) {
  if (v > 32767.f) { statusFlags |= 0x4; return 32767; }
  if (v < -32768.f) { statusFlags |= 0x4; return -32768; }
  return (int16_t)lroundf(v);
}

static void mahonyUpdate(float gx, float gy, float gz, float ax, float ay, float az, float dt) {
  float n = sqrtf(ax * ax + ay * ay + az * az);
  if (n > 1e-3f) {
    ax /= n; ay /= n; az /= n;
    // estimated gravity direction
    float vx = 2.f * (q1 * q3 - q0 * q2);
    float vy = 2.f * (q0 * q1 + q2 * q3);
    float vz = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3;
    float ex = (ay * vz - az * vy), ey = (az * vx - ax * vz), ez = (ax * vy - ay * vx);
    // trust accelerometer less during strong dynamic accelerations
    float w = fabsf(n - kG) < 0.25f * kG ? 1.f : 0.1f;
    ix += ki * ex * dt * w; iy += ki * ey * dt * w; iz += ki * ez * dt * w;
    gx += kp * ex * w + ix; gy += kp * ey * w + iy; gz += kp * ez * w + iz;
  }
  float hq0 = q0, hq1 = q1, hq2 = q2;
  q0 += 0.5f * (-hq1 * gx - hq2 * gy - q3 * gz) * dt;
  q1 += 0.5f * (hq0 * gx + hq2 * gz - q3 * gy) * dt;
  q2 += 0.5f * (hq0 * gy - hq1 * gz + q3 * gx) * dt;
  q3 += 0.5f * (hq0 * gz + hq1 * gy - hq2 * gx) * dt;
  n = sqrtf(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
  q0 /= n; q1 /= n; q2 /= n; q3 /= n;
}

static void calibrateGyro() {
  float s[3] = {0, 0, 0};
  int n = 0;
  uint32_t t0 = millis();
  while (millis() - t0 < 1000) {
    if (IMU.gyroscopeAvailable()) {
      float g[3];
      IMU.readGyroscope(g[0], g[1], g[2]);
      for (int i = 0; i < 3; ++i) s[i] += g[i];
      ++n;
    }
  }
  if (n > 10) for (int i = 0; i < 3; ++i) gbias[i] = s[i] / n;
  q0 = 1; q1 = q2 = q3 = 0; ix = iy = iz = 0; yawOffset = 0;
  statusFlags |= 0x2;
}

// ------------------------------------------------------------- I2C callbacks
static void onJetsonRequest() {
  uint8_t buf[24];
  noInterrupts();
  for (int i = 0; i < 24; ++i) buf[i] = txPacket[i];
  interrupts();
  uint32_t age = millis() - sampleMillis;
  buf[22] = age > 255 ? 255 : (uint8_t)age;
  uint8_t c = 0;
  for (int i = 0; i < 23; ++i) c ^= buf[i];
  buf[23] = c;
  jetsonWire.write(buf, 24);
}

static void onJetsonReceive(int n) {
  if (n < 1) return;
  uint8_t cmd = jetsonWire.read();
  if (cmd == 0x01) calibrateRequest = true;
  else if (cmd == 0x02) zeroYawRequest = true;
  else if (cmd == 0x03 && n >= 9) {
    uint8_t b[8];
    for (int i = 0; i < 8; ++i) b[i] = jetsonWire.read();
    memcpy((void*)&kp, b, 4);
    memcpy((void*)&ki, b + 4, 4);
  }
  while (jetsonWire.available()) jetsonWire.read();
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_BUILTIN, OUTPUT);
  if (!IMU.begin()) {
    statusFlags = 0;
    while (true) { digitalWrite(LED_BUILTIN, millis() / 100 % 2); }   // fast blink = IMU failure
  }
  statusFlags |= 0x1;
  // Arduino_LSM6DS3 defaults: accel +-4 g @ 104 Hz, gyro +-2000 dps @ 104 Hz.
  // Raise ODR to 416 Hz for lower latency (CTRL1_XL / CTRL2_G on the internal Wire bus, addr 0x6A).
  Wire.beginTransmission(0x6A); Wire.write(0x10); Wire.write(0x6A); Wire.endTransmission();  // XL 416 Hz, +-4 g
  Wire.beginTransmission(0x6A); Wire.write(0x11); Wire.write(0x6C); Wire.endTransmission();  // G 416 Hz, 2000 dps

  jetsonWire.begin(kJetsonI2CAddr);
  pinPeripheral(11, PIO_SERCOM);
  pinPeripheral(13, PIO_SERCOM);
  jetsonWire.onRequest(onJetsonRequest);
  jetsonWire.onReceive(onJetsonReceive);
}

void loop() {
  static uint32_t lastMicros = micros();
  if (calibrateRequest) { digitalWrite(LED_BUILTIN, HIGH); calibrateGyro(); calibrateRequest = false; digitalWrite(LED_BUILTIN, LOW); }
  if (!(IMU.gyroscopeAvailable() && IMU.accelerationAvailable())) return;
  float graw[3], araw[3];
  IMU.readGyroscope(graw[0], graw[1], graw[2]);       // deg/s
  IMU.readAcceleration(araw[0], araw[1], araw[2]);    // g
  float g[3], a[3];
  for (int i = 0; i < 3; ++i) {
    g[i] = kAxisSign[i] * (graw[kAxisMap[i]] - gbias[kAxisMap[i]]) * kDeg2Rad;
    a[i] = kAxisSign[i] * araw[kAxisMap[i]] * kG;
  }
  uint32_t now = micros();
  float dt = (now - lastMicros) * 1e-6f;
  lastMicros = now;
  if (dt <= 0.f || dt > 0.05f) dt = 1.f / 416.f;
  mahonyUpdate(g[0], g[1], g[2], a[0], a[1], a[2], dt);

  // ZYX Euler (identical to the simulator's quat_to_rpy)
  float roll = atan2f(2.f * (q0 * q1 + q2 * q3), 1.f - 2.f * (q1 * q1 + q2 * q2));
  float sp = 2.f * (q0 * q2 - q3 * q1);
  float pitch = asinf(sp > 1.f ? 1.f : (sp < -1.f ? -1.f : sp));
  float yaw = atan2f(2.f * (q0 * q3 + q1 * q2), 1.f - 2.f * (q2 * q2 + q3 * q3));
  if (zeroYawRequest) { yawOffset = yaw; zeroYawRequest = false; }
  yaw -= yawOffset;
  if (yaw > PI) yaw -= TWO_PI;
  if (yaw < -PI) yaw += TWO_PI;

  statusFlags &= ~0x4;
  int16_t v[9] = {clamp16(roll * 1e4f), clamp16(pitch * 1e4f), clamp16(yaw * 1e4f),
                  clamp16(g[0] * 1e3f), clamp16(g[1] * 1e3f), clamp16(g[2] * 1e3f),
                  clamp16(a[0] * 1e3f), clamp16(a[1] * 1e3f), clamp16(a[2] * 1e3f)};
  uint8_t pkt[24];
  pkt[0] = 0xB8;
  pkt[1] = ++seqCounter;
  memcpy(pkt + 2, v, sizeof(v));   // SAMD21 is little-endian
  pkt[20] = statusFlags & 0xFF;
  pkt[21] = statusFlags >> 8;
  pkt[22] = 0;
  pkt[23] = 0;
  noInterrupts();
  for (int i = 0; i < 24; ++i) txPacket[i] = pkt[i];
  sampleMillis = millis();
  interrupts();

  static uint32_t lastPrint = 0;
  if (Serial && millis() - lastPrint > 200) {  // USB debug output
    lastPrint = millis();
    Serial.print("rpy[deg] "); Serial.print(roll * 57.2958f); Serial.print(' ');
    Serial.print(pitch * 57.2958f); Serial.print(' '); Serial.println(yaw * 57.2958f);
  }
}
