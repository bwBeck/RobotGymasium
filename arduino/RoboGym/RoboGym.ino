/*
  RoboGym.ino - robot side of the RoboGym gymnasium environment
  =============================================================

  Target: ELEGOO Conqueror Robot Tank (Arduino UNO + ConquerorCar shield).

  How the pieces talk:

    PC (robotgym.RoboGym) ==WiFi==> ESP32 camera module, running ELEGOO's STOCK firmware
       |  TCP :100   "{...}" frames  <-->  UART 9600 baud  <-->  this sketch on the UNO
       |  HTTP :81/stream (MJPEG), :80/capture  -->  camera image, straight to the PC

  The camera is wired to the ESP32, not to the UNO, and one JPEG frame is tens of
  kilobytes while the UNO has 2 KB of RAM - so the image never passes through this
  sketch. The Python env fetches it from the ESP32 and merges it with the sensor
  packet sent from here.

  What this sketch reports:
    * heading         MPU6050 gyro Z - measured. Bias calibrated at boot and on reset,
                      and tracked continuously while the robot is parked.
    * motor turns     ESTIMATED. The stock motors have no encoders, so sprocket turns are
                      integrated from a motor model (PWM, battery voltage, deadband,
                      first-order lag). Its constants are sent by the PC (C frame), so
                      calibration never needs a re-flash.
    * position        dead reckoning: estimated track speed + gyro heading.
    * distance        HC-SR04 ultrasonic, interrupt driven (never blocks the loop).
    * time            micros() timestamp of every packet; the PC differences them.

  The robot sends cumulative values since the last reset (R frame). The PC computes the
  per-step changes, so a lost packet can never silently drop motion.

  Upload: set the shield's "Upload-Cam" switch to "Upload", board "Arduino Uno", upload,
  then set the switch back to "Cam" so the ESP32 can reach the UNO. Only the built-in
  Wire and Servo libraries are needed.

  Serial protocol v1 - ASCII frames "{...}", comma-separated integers, no spaces
  (the ESP32 bridge strips spaces and flushes on '}').

    PC -> robot
      {A,seq,left,right,dur_ms}  drive: track PWM -255..255 (sign = direction). After
                                 dur_ms the robot replies with an O frame. The action keeps
                                 running until the next command, or until hold_ms passes
                                 after the reply with no new command (safety stop).
      {R,seq,calib}              reset: stop, recalibrate the gyro if calib=1 (robot must be
                                 still, takes ~0.5 s), zero odometry and turn counters, reply O.
      {C,seq,hold_ms,rpm_x10,deadband_pwm,tau_ms,sprocket_x10mm,track_mm,gyro_sign}
                                 configure the safety timeout and motor model, reply O.
      {P,seq}                    poll: reply O immediately, change nothing.

    robot -> PC
      {O,seq,t_us,x,y,th,turnsL,turnsR,dist_mm,batt_cV,flags}
          t_us    micros() when the packet was built (wraps every ~71 minutes)
          x, y    dead-reckoned position since the last R, 0.1 mm (x forward, y left at reset)
          th      heading since the last R, 1e-4 rad, counter-clockwise positive, unwrapped
          turns   cumulative sprocket turns since the last R, 1e-3 turn (estimated)
          dist    ultrasonic range in mm, -1 = no valid reading
          batt    battery voltage in 0.01 V
          flags   1 IMU ok | 2 watchdog stopped the motors since the last packet
                  | 4 no echo (nothing in range) | 8 first packet since boot
                  | 16 gyro calibration failed (robot was moving)
      {H,RoboGym,1}              sent once at boot
      {E,seq,code}               command rejected (1 = malformed)

    The ESP32 sends {"N":100} when the PC disconnects (treated as STOP), plus chatter of
    its own ({Factory}, {WA_NO}, {WA_OK}) which is ignored.
*/

#include <Arduino.h>
#include <Servo.h>
#include <Wire.h>
#include <avr/wdt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ================================ configuration ================================

#define MOTOR_DRIVER_TB6612 1   // shield V1.1 (most kits since 2022, see ReadMeFirst.pdf)
#define MOTOR_DRIVER_DRV8835 2  // shield V1.0
#ifndef MOTOR_DRIVER
#define MOTOR_DRIVER MOTOR_DRIVER_TB6612  // <-- match the driver chip on your shield
#endif

#define SERIAL_BAUD 9600         // fixed by the stock ESP32 bridge firmware
#define INVERT_LEFT_MOTOR 0      // set to 1 if a track runs backwards when commanded forward
#define INVERT_RIGHT_MOTOR 0
#define CENTER_SERVOS_AT_BOOT 1  // point the camera / ultrasonic mount straight ahead at boot
#define PAN_SERVO_ANGLE 90
#define TILT_SERVO_ANGLE 90

// Defaults for the runtime-configurable values. The Python env overwrites all of them
// with a C frame when it connects, so calibrate in Python (RobotCalibration), not here.
#define DEFAULT_HOLD_TIMEOUT_MS 500
#define DEFAULT_MAX_RPM 180.0f  // sprocket RPM at PWM 255 and NOMINAL_BATTERY_V, on the ground
#define DEFAULT_DEADBAND_PWM 40
#define DEFAULT_TAU_S 0.08f
#define DEFAULT_SPROCKET_DIAMETER_M 0.043f
#define DEFAULT_TRACK_WIDTH_M 0.15f
#define DEFAULT_GYRO_SIGN 1

#define NOMINAL_BATTERY_V 7.4f
#define MIN_MOTOR_BATTERY_V 5.0f  // below this the battery is off (USB power only): motors can't turn

const uint8_t PROTOCOL_VERSION = 1;

// ==================================== pins =====================================

#if MOTOR_DRIVER == MOTOR_DRIVER_TB6612
// Motor A = right track, motor B = left track. AIN2/BIN2 are generated from AIN1/BIN1
// by an SN74LVC2G14 inverter on the shield, so one direction pin per motor suffices.
const uint8_t PIN_PWM_RIGHT = 5;  // PWMA
const uint8_t PIN_PWM_LEFT = 6;   // PWMB
const uint8_t PIN_DIR_RIGHT = 7;  // AIN1
const uint8_t PIN_DIR_LEFT = 8;   // BIN1
#define HAS_STBY 1
const uint8_t PIN_STBY = 3;
const uint8_t FORWARD_LEVEL_RIGHT = HIGH;
const uint8_t FORWARD_LEVEL_LEFT = HIGH;
#elif MOTOR_DRIVER == MOTOR_DRIVER_DRV8835
// DRV8835 in PHASE/ENABLE mode, no standby pin.
const uint8_t PIN_PWM_RIGHT = 5;  // AENBL
const uint8_t PIN_PWM_LEFT = 6;   // BENBL
const uint8_t PIN_DIR_RIGHT = 8;  // APHASE
const uint8_t PIN_DIR_LEFT = 7;   // BPHASE
#define HAS_STBY 0
const uint8_t FORWARD_LEVEL_RIGHT = LOW;
const uint8_t FORWARD_LEVEL_LEFT = HIGH;
#else
#error "MOTOR_DRIVER must be MOTOR_DRIVER_TB6612 or MOTOR_DRIVER_DRV8835"
#endif

const uint8_t PIN_TRIG = 13;
const uint8_t PIN_ECHO = 12;  // PB4 / PCINT4
const uint8_t PIN_BATTERY = A3;
const uint8_t PIN_SERVO_PAN = 10;
const uint8_t PIN_SERVO_TILT = 11;

// ================================== constants ==================================

const uint8_t MPU_ADDR = 0x68;
const float GYRO_LSB_PER_DPS = 65.5f;  // +-500 deg/s full scale
const float GYRO_RAD_PER_LSB = (float)(PI / 180.0) / GYRO_LSB_PER_DPS;
const float GYRO_DEADBAND_RAD_S = 0.005f;  // ~0.3 deg/s
const uint16_t GYRO_CALIB_SAMPLES = 400;
const int16_t GYRO_CALIB_MAX_SPREAD = 200;  // LSB; a bigger spread means the robot moved
const float GYRO_BIAS_TRACK_TAU_S = 5.0f;
const int16_t GYRO_BIAS_TRACK_WINDOW = 131;     // only track the bias when |rate| < ~2 deg/s
const uint32_t PARKED_SETTLE_US = 500000UL;     // motors off this long = robot is still
const uint8_t GYRO_MAX_FAILED_READS = 25;       // ~100 ms of I2C errors -> give up on the IMU

const uint32_t TICK_US = 4000;              // 250 Hz model / gyro integration
const uint32_t PING_PERIOD_US = 60000;      // HC-SR04 wants >= 60 ms between pings
const uint32_t ECHO_RISE_TIMEOUT_US = 30000;
const uint32_t ECHO_MAX_US = 23500;         // ~4 m round trip
const int16_t RANGE_MAX_MM = 4000;
const float MM_PER_ECHO_US = 0.1715f;       // speed of sound / 2 at ~20 C
const uint32_t BATTERY_PERIOD_US = 50000;
const long MAX_ACTION_DURATION_MS = 10000;

const uint8_t FLAG_IMU_OK = 1;
const uint8_t FLAG_WATCHDOG = 2;
const uint8_t FLAG_NO_ECHO = 4;
const uint8_t FLAG_BOOT = 8;
const uint8_t FLAG_CALIB_FAILED = 16;

const int8_t ERR_MALFORMED = 1;

// ==================================== types ====================================

// Integer + fractional accumulator. A bare float stops resolving small increments once
// the total grows; this stays exact over any range.
struct Accum {
  long whole;
  float frac;
  void add(float d) {
    frac += d;
    long w = (long)frac;
    whole += w;
    frac -= (float)w;
  }
  void clear() {
    whole = 0;
    frac = 0.0f;
  }
};

enum EchoState : uint8_t { ECHO_IDLE, ECHO_WAIT_RISE, ECHO_WAIT_FALL, ECHO_DONE };

// =================================== globals ===================================

// Runtime configuration (C frame).
uint32_t holdTimeoutUs = DEFAULT_HOLD_TIMEOUT_MS * 1000UL;
float maxRps = DEFAULT_MAX_RPM / 60.0f;
float deadbandPwm = DEFAULT_DEADBAND_PWM;
float tauS = DEFAULT_TAU_S;
float sprocketCircM = (float)PI * DEFAULT_SPROCKET_DIAMETER_M;
float trackWidthM = DEFAULT_TRACK_WIDTH_M;
int8_t gyroSign = DEFAULT_GYRO_SIGN;

// Motors and motor model.
int16_t cmdLeft = 0, cmdRight = 0;  // applied PWM, -255..255
float rpsLeft = 0.0f, rpsRight = 0.0f;  // estimated sprocket speed, rev/s
uint32_t motorsZeroSinceUs = 0;

// IMU.
bool imuOk = false;
float gyroBias = 0.0f;  // LSB
float lastOmega = 0.0f;  // rad/s
uint8_t gyroFailedReads = 0;

// Odometry since the last reset: 0.1 mm, 0.1 mm, 1e-4 rad, 1e-3 turn, 1e-3 turn.
Accum odoX, odoY, odoTh, turnsLeft, turnsRight;
float heading = 0.0f;  // wrapped to [-pi, pi], used for the x/y projection

// Ultrasonic (written by the pin-change ISR).
volatile uint8_t echoState = ECHO_IDLE;
volatile uint32_t echoRiseUs = 0, echoFallUs = 0;
uint32_t pingStartUs = 0;
int16_t rangeMm = -1;
bool rangeNoEcho = false;

float batteryV = 0.0f;

// Protocol.
char rxBuf[64];
uint8_t rxLen = 0;
bool rxInFrame = false;
bool replyPending = false;
uint16_t pendingSeq = 0;
uint32_t pendingStartUs = 0, pendingDurUs = 0;
uint32_t lastReplyUs = 0, lastTickUs = 0, lastBatteryUs = 0;
uint8_t latchedFlags = FLAG_BOOT;

// =================================== helpers ===================================

// Wrap-safe "has `dur` passed since `since`?". Signed, so a timestamp written by an ISR
// just after `now` was read counts as "not yet" instead of underflowing to "long ago".
// The flip side: a timestamp must never age past ~35 minutes, so every one of them is
// initialised in setup() and the long-lived ones are kept fresh where they are used.
static inline bool elapsed(uint32_t now, uint32_t since, uint32_t dur) {
  return (int32_t)(now - since) >= (int32_t)dur;
}

static long clampLong(long v, long lo, long hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

static float wrapPi(float a) {
  while (a > (float)PI) a -= (float)TWO_PI;
  while (a < -(float)PI) a += (float)TWO_PI;
  return a;
}

// =================================== motors ====================================

static void writeMotor(uint8_t pinPwm, uint8_t pinDir, uint8_t forwardLevel, bool inverted,
                       int16_t pwm) {
  bool forward = (pwm >= 0) != inverted;
  uint8_t backwardLevel = (forwardLevel == HIGH) ? LOW : HIGH;
  digitalWrite(pinDir, forward ? forwardLevel : backwardLevel);
  analogWrite(pinPwm, pwm >= 0 ? pwm : -pwm);
}

// PWM 0 short-brakes on both driver chips.
static void setMotors(int16_t left, int16_t right) {
  if (left == 0 && right == 0 && (cmdLeft != 0 || cmdRight != 0)) motorsZeroSinceUs = micros();
  cmdLeft = left;
  cmdRight = right;
#if HAS_STBY
  digitalWrite(PIN_STBY, HIGH);
#endif
  writeMotor(PIN_PWM_LEFT, PIN_DIR_LEFT, FORWARD_LEVEL_LEFT, INVERT_LEFT_MOTOR, left);
  writeMotor(PIN_PWM_RIGHT, PIN_DIR_RIGHT, FORWARD_LEVEL_RIGHT, INVERT_RIGHT_MOTOR, right);
}

// Stop and, on the TB6612, release the outputs (coast, lowest power).
static void disableMotors() {
  setMotors(0, 0);
#if HAS_STBY
  digitalWrite(PIN_STBY, LOW);
#endif
}

// Steady-state sprocket speed (rev/s) the model expects for a PWM command. Speed is
// taken as proportional to the effective drive (PWM scaled by battery voltage) above a
// friction deadband.
static float targetRps(int16_t pwm) {
  if (pwm == 0 || batteryV < MIN_MOTOR_BATTERY_V) return 0.0f;
  float effective = (pwm >= 0 ? pwm : -pwm) * (batteryV / NOMINAL_BATTERY_V);
  if (effective <= deadbandPwm) return 0.0f;
  float rps = maxRps * (effective - deadbandPwm) / (255.0f - deadbandPwm);
  return pwm > 0 ? rps : -rps;
}

// ==================================== IMU ======================================

static bool mpuWrite(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}

static int mpuRead(uint8_t reg) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission() != 0) return -1;
  if (Wire.requestFrom(MPU_ADDR, (uint8_t)1) != 1) return -1;
  return Wire.read();
}

static bool readGyroZ(int16_t &out) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write((uint8_t)0x47);  // GYRO_ZOUT_H
  if (Wire.endTransmission() != 0) return false;
  if (Wire.requestFrom(MPU_ADDR, (uint8_t)2) != 2) return false;
  uint8_t hi = Wire.read();  // separate statements: operand evaluation order is unspecified
  uint8_t lo = Wire.read();
  out = (int16_t)(((uint16_t)hi << 8) | lo);
  return true;
}

static bool initImu() {
  Wire.begin();
  Wire.setClock(400000);
#if defined(WIRE_HAS_TIMEOUT)
  Wire.setWireTimeout(3000, true);  // never hang the loop on a stuck bus
#endif
  int id = mpuRead(0x75);  // WHO_AM_I; clones answer with various ids, like ELEGOO
  if (id <= 0 || id == 0xFF) return false;  // accept anything but 0x00 / 0xFF
  mpuWrite(0x6B, 0x80);  // device reset
  delay(100);
  bool ok = mpuWrite(0x6B, 0x01)    // wake up, clock = PLL on gyro X
            && mpuWrite(0x1A, 0x04)  // DLPF: gyro ~20 Hz bandwidth
            && mpuWrite(0x19, 0x00)  // sample rate divider: 1 kHz
            && mpuWrite(0x1B, 0x08); // gyro full scale +-500 deg/s
  delay(50);
  return ok;
}

// Averages the gyro at rest. Fails (keeping the old bias) if the robot moved.
static bool calibrateGyro() {
  long sum = 0;
  int16_t lo = 32767, hi = -32768;
  uint16_t n = 0;
  for (uint16_t i = 0; i < GYRO_CALIB_SAMPLES; i++) {
    int16_t g;
    if (readGyroZ(g)) {
      sum += g;
      if (g < lo) lo = g;
      if (g > hi) hi = g;
      n++;
    }
    delayMicroseconds(1000);
    if ((i & 63) == 0) wdt_reset();
  }
  if (n < GYRO_CALIB_SAMPLES / 2 || (int32_t)hi - (int32_t)lo > GYRO_CALIB_MAX_SPREAD) return false;
  gyroBias = (float)sum / n;
  return true;
}

// ================================== battery ====================================

static float readBatteryV() {
  return analogRead(PIN_BATTERY) * (0.0375f * 1.08f);  // ELEGOO's divider constant + 8 %
}

// ================================= ultrasonic ==================================

ISR(PCINT0_vect) {
  uint32_t t = micros();
  bool high = digitalRead(PIN_ECHO) == HIGH;
  if (echoState == ECHO_WAIT_RISE && high) {
    echoRiseUs = t;
    echoState = ECHO_WAIT_FALL;
  } else if (echoState == ECHO_WAIT_FALL && !high) {
    echoFallUs = t;
    echoState = ECHO_DONE;
  }
}

static void initUltrasonic() {
  pinMode(PIN_TRIG, OUTPUT);
  digitalWrite(PIN_TRIG, LOW);
  pinMode(PIN_ECHO, INPUT);
  PCICR |= _BV(PCIE0);    // pin-change interrupts for D8..D13
  PCMSK0 |= _BV(PCINT4);  // ...but only D12 (echo)
}

// Non-blocking ping state machine; the ISR timestamps the echo edges.
static void serviceUltrasonic(uint32_t now) {
  uint8_t state = echoState;
  if (state == ECHO_DONE) {
    uint32_t width = echoFallUs - echoRiseUs;
    rangeNoEcho = width > ECHO_MAX_US;
    rangeMm = rangeNoEcho ? RANGE_MAX_MM : (int16_t)(width * MM_PER_ECHO_US);
    if (rangeMm > RANGE_MAX_MM) rangeMm = RANGE_MAX_MM;
    echoState = ECHO_IDLE;
  } else if (state == ECHO_WAIT_RISE && elapsed(now, pingStartUs, ECHO_RISE_TIMEOUT_US)) {
    rangeMm = -1;  // the sensor never answered
    rangeNoEcho = false;
    echoState = ECHO_IDLE;
  } else if (state == ECHO_WAIT_FALL && elapsed(now, echoRiseUs, ECHO_MAX_US)) {
    rangeMm = RANGE_MAX_MM;  // nothing within range; some clones hold echo high ~38 ms
    rangeNoEcho = true;
    echoState = ECHO_IDLE;
  } else if (state == ECHO_IDLE && elapsed(now, pingStartUs, PING_PERIOD_US)) {
    pingStartUs = now;
    if (digitalRead(PIN_ECHO) == HIGH) return;  // still busy from the last ping: next period
    echoState = ECHO_WAIT_RISE;
    digitalWrite(PIN_TRIG, HIGH);
    delayMicroseconds(10);
    digitalWrite(PIN_TRIG, LOW);
  }
}

// ================================== odometry ===================================

static void resetOdometry() {
  odoX.clear();
  odoY.clear();
  odoTh.clear();
  turnsLeft.clear();
  turnsRight.clear();
  heading = 0.0f;
  rpsLeft = rpsRight = 0.0f;
  lastTickUs = micros();
}

static void tick(uint32_t now, float dt) {
  bool parked = cmdLeft == 0 && cmdRight == 0 && elapsed(now, motorsZeroSinceUs, PARKED_SETTLE_US);
  if (parked) motorsZeroSinceUs = now - PARKED_SETTLE_US;  // keep the timestamp fresh

  // Motor model: first-order lag towards the speed the PWM should produce.
  float a = (tauS > 0.0f) ? dt / (tauS + dt) : 1.0f;
  rpsLeft += a * (targetRps(cmdLeft) - rpsLeft);
  rpsRight += a * (targetRps(cmdRight) - rpsRight);
  float vLeft = rpsLeft * sprocketCircM;  // m/s
  float vRight = rpsRight * sprocketCircM;

  // Yaw rate: gyro when available, otherwise the skid-steer estimate.
  float omega;
  if (imuOk) {
    int16_t raw;
    if (readGyroZ(raw)) {
      gyroFailedReads = 0;
      float centered = raw - gyroBias;
      if (parked && fabs(centered) < GYRO_BIAS_TRACK_WINDOW) {
        gyroBias += centered * (dt / GYRO_BIAS_TRACK_TAU_S);  // slow drift compensation
        centered = raw - gyroBias;
      }
      omega = gyroSign * centered * GYRO_RAD_PER_LSB;
      if (fabs(omega) < GYRO_DEADBAND_RAD_S) omega = 0.0f;
      lastOmega = omega;
    } else {
      omega = lastOmega;  // ride out a glitch
      if (++gyroFailedReads > GYRO_MAX_FAILED_READS) imuOk = false;
    }
  } else {
    omega = (vRight - vLeft) / trackWidthM;
  }

  float dTh = omega * dt;
  float ds = 0.5f * (vLeft + vRight) * dt;  // m
  if (ds != 0.0f) {
    float mid = heading + 0.5f * dTh;  // midpoint heading
    odoX.add(ds * cosf(mid) * 1e4f);
    odoY.add(ds * sinf(mid) * 1e4f);
  }
  odoTh.add(dTh * 1e4f);
  heading = wrapPi(heading + dTh);
  turnsLeft.add(rpsLeft * dt * 1e3f);
  turnsRight.add(rpsRight * dt * 1e3f);
}

// ================================== protocol ===================================

static void sendObservation(uint16_t seq) {
  uint8_t flags = latchedFlags;
  if (imuOk) flags |= FLAG_IMU_OK;
  if (rangeNoEcho) flags |= FLAG_NO_ECHO;
  char buf[112];
  int n = snprintf(buf, sizeof(buf), "{O,%u,%lu,%ld,%ld,%ld,%ld,%ld,%d,%d,%u}", (unsigned)seq,
                   (unsigned long)micros(), (long)odoX.whole, (long)odoY.whole, (long)odoTh.whole,
                   (long)turnsLeft.whole, (long)turnsRight.whole, (int)rangeMm,
                   (int)(batteryV * 100.0f + 0.5f), (unsigned)flags);
  if (n > 0) Serial.write((const uint8_t *)buf, n < (int)sizeof(buf) ? n : (int)sizeof(buf) - 1);
  latchedFlags = 0;
  lastReplyUs = micros();
}

static void sendError(uint16_t seq, int8_t code) {
  char buf[24];
  int n = snprintf(buf, sizeof(buf), "{E,%u,%d}", (unsigned)seq, (int)code);
  if (n > 0) Serial.write((const uint8_t *)buf, n);
}

static void sendHello() {
  char buf[24];
  int n = snprintf(buf, sizeof(buf), "{H,RoboGym,%u}", (unsigned)PROTOCOL_VERSION);
  if (n > 0) Serial.write((const uint8_t *)buf, n);
}

// Parses ",<int>,<int>,..." into out[]. Returns the count, or -1 if malformed.
static int8_t parseFields(const char *s, long *out, int8_t maxN) {
  int8_t n = 0;
  while (*s) {
    if (*s != ',' || n >= maxN) return -1;
    s++;
    char *end;
    long v = strtol(s, &end, 10);
    if (end == s) return -1;
    out[n++] = v;
    s = end;
  }
  return n;
}

static void applyConfig(const long *v) {
  holdTimeoutUs = (uint32_t)clampLong(v[1], 50, 60000) * 1000UL;
  maxRps = clampLong(v[2], 1, 100000) / 600.0f;  // rpm x10 -> rev/s
  deadbandPwm = (float)clampLong(v[3], 0, 200);
  tauS = clampLong(v[4], 0, 2000) / 1000.0f;
  sprocketCircM = (float)PI * clampLong(v[5], 10, 2000) / 10000.0f;  // 0.1 mm -> m
  trackWidthM = clampLong(v[6], 10, 1000) / 1000.0f;
  gyroSign = v[7] < 0 ? -1 : 1;
}

static void handleFrame(const char *s) {
  if (strcmp(s, "\"N\":100") == 0) {  // ESP32: the PC disconnected -> stop
    disableMotors();
    replyPending = false;
    return;
  }
  char type = s[0];
  if (type != 'A' && type != 'R' && type != 'C' && type != 'P') return;  // ESP32 chatter

  long v[8];
  int8_t n = parseFields(s + 1, v, 8);
  uint16_t seq = n > 0 ? (uint16_t)v[0] : 0;
  replyPending = false;  // a new command supersedes an unanswered one

  if (type == 'A' && n == 4) {
    setMotors((int16_t)clampLong(v[1], -255, 255), (int16_t)clampLong(v[2], -255, 255));
    pendingSeq = seq;
    pendingStartUs = micros();
    pendingDurUs = (uint32_t)clampLong(v[3], 0, MAX_ACTION_DURATION_MS) * 1000UL;
    replyPending = true;  // loop() replies once the duration has passed
  } else if (type == 'R' && n == 2) {
    disableMotors();
    if (v[1] != 0) {
      if (!imuOk) imuOk = initImu();  // second chance for a loose I2C cable
      if (!imuOk || !calibrateGyro()) latchedFlags |= FLAG_CALIB_FAILED;
      gyroFailedReads = 0;
    }
    resetOdometry();
    sendObservation(seq);
  } else if (type == 'C' && n == 8) {
    applyConfig(v);
    sendObservation(seq);
  } else if (type == 'P' && n == 1) {
    sendObservation(seq);
  } else {
    sendError(seq, ERR_MALFORMED);
  }
}

static void pollSerial() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '{') {
      rxInFrame = true;
      rxLen = 0;
    } else if (!rxInFrame || c == ' ' || c == '\t' || c == '\r' || c == '\n') {
      // outside a frame, or whitespace (also lets you type frames in the Serial Monitor)
    } else if (c == '}') {
      rxBuf[rxLen] = '\0';
      rxInFrame = false;
      handleFrame(rxBuf);
    } else if (rxLen < sizeof(rxBuf) - 1) {
      rxBuf[rxLen++] = c;
    } else {
      rxInFrame = false;  // oversized: drop it
    }
  }
}

// =================================== servos ====================================

#if CENTER_SERVOS_AT_BOOT
static void centerServos() {
  Servo servo;
  servo.attach(PIN_SERVO_PAN);
  servo.write(PAN_SERVO_ANGLE);
  delay(500);
  servo.detach();
  servo.attach(PIN_SERVO_TILT);
  servo.write(TILT_SERVO_ANGLE);
  delay(500);
  servo.detach();
}
#endif

// ================================ setup / loop =================================

void setup() {
  wdt_disable();
  pinMode(PIN_PWM_LEFT, OUTPUT);
  pinMode(PIN_PWM_RIGHT, OUTPUT);
  pinMode(PIN_DIR_LEFT, OUTPUT);
  pinMode(PIN_DIR_RIGHT, OUTPUT);
#if HAS_STBY
  pinMode(PIN_STBY, OUTPUT);
#endif
  disableMotors();

  Serial.begin(SERIAL_BAUD);
  initUltrasonic();
  pinMode(PIN_BATTERY, INPUT);
  batteryV = readBatteryV();

#if CENTER_SERVOS_AT_BOOT
  centerServos();
#endif

  imuOk = initImu();
  bool calibrated = false;
  for (uint8_t attempt = 0; imuOk && !calibrated && attempt < 3; attempt++) {
    calibrated = calibrateGyro();
  }
  if (imuOk && !calibrated) latchedFlags |= FLAG_CALIB_FAILED;

  resetOdometry();  // also sets lastTickUs
  uint32_t now = micros();
  motorsZeroSinceUs = now;
  lastReplyUs = now;
  lastBatteryUs = now;
  pingStartUs = now - PING_PERIOD_US;  // first ping right away
  sendHello();
  wdt_enable(WDTO_2S);
}

void loop() {
  wdt_reset();
  pollSerial();

  uint32_t now = micros();
  if (elapsed(now, lastTickUs, TICK_US)) {
    float dt = (now - lastTickUs) * 1e-6f;
    lastTickUs = now;
    tick(now, dt);
  }
  serviceUltrasonic(now);
  if (elapsed(now, lastBatteryUs, BATTERY_PERIOD_US)) {
    lastBatteryUs = now;
    batteryV += 0.2f * (readBatteryV() - batteryV);  // smooth out motor-induced sag
  }

  if (replyPending) {
    if (elapsed(now, pendingStartUs, pendingDurUs)) {
      replyPending = false;
      sendObservation(pendingSeq);
    }
  } else if ((cmdLeft != 0 || cmdRight != 0) && elapsed(now, lastReplyUs, holdTimeoutUs)) {
    disableMotors();  // the PC went quiet: safety stop
    latchedFlags |= FLAG_WATCHDOG;
  }
}
