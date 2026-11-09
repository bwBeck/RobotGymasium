// Runs arduino/RoboGym/RoboGym.ino as a normal program for tests.
//
// Serial is stdin/stdout. Around the sketch sits a tiny "robot" built from the shield's
// wiring (TB6612 build): the gyro reports a yaw rate proportional to the difference of
// the track commands (positive = counter-clockwise, as on the kit), the ultrasonic
// sensor echoes an obstacle 1.000 m away and the battery reads 7.90 V. micros() starts
// 3 s before its 32-bit wrap so wrap handling is exercised in every run.
#include <Arduino.h>
#include <Wire.h>
#include <fcntl.h>
#include <time.h>
#include <unistd.h>

void setup();
void loop();
void PCINT0_vect_handler();

uint8_t PCICR = 0;
uint8_t PCMSK0 = 0;
HardwareSerial Serial;
TwoWire Wire;

static const unsigned long MICROS_START = 0xFFFFFFFFUL - 3000000UL;
static const double OBSTACLE_MM = 1000.0;
static const int GYRO_BIAS_LSB = 37;
static const double MAX_YAW_RATE = 3.0;  // rad/s for full opposite track commands

static struct timespec t0;
static uint8_t pinState[20];
static int pwmValue[20];
static bool stdinEof = false;
static uint8_t rxBuf[256];
static int rxHead = 0, rxTail = 0;

static double nowUs() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (t.tv_sec - t0.tv_sec) * 1e6 + (t.tv_nsec - t0.tv_nsec) / 1e3;
}

unsigned long micros() { return (MICROS_START + (unsigned long)nowUs()) & 0xFFFFFFFFUL; }
unsigned long millis() { return micros() / 1000UL; }
void delay(unsigned long ms) { usleep(ms * 1000); }
void delayMicroseconds(unsigned int us) { usleep(us); }

// ---- ultrasonic: echo pulse scheduled on the trigger's falling edge ----
static double echoRiseAt = -1, echoFallAt = -1;

static void setEcho(uint8_t level) {
  pinState[12] = level;
  if ((PCICR & _BV(PCIE0)) && (PCMSK0 & _BV(PCINT4))) PCINT0_vect_handler();
}

void pinMode(uint8_t, uint8_t) {}

void digitalWrite(uint8_t pin, uint8_t val) {
  if (pin == 13 && pinState[13] == HIGH && val == LOW) {
    echoRiseAt = nowUs() + 400;
    echoFallAt = echoRiseAt + OBSTACLE_MM / 0.1715;
  }
  if (pin < 20) pinState[pin] = val;
}

int digitalRead(uint8_t pin) { return pin < 20 ? pinState[pin] : LOW; }
void analogWrite(uint8_t pin, int val) {
  if (pin < 20) pwmValue[pin] = val;
}
int analogRead(uint8_t pin) { return pin == A3 ? 195 : 0; }  // 195 * 0.0405 = 7.90 V

// ---- motors (TB6612 wiring) and the gyro they drive ----
static double trackCommand(uint8_t pwmPin, uint8_t dirPin) {
  if (pinState[3] != HIGH) return 0.0;  // STBY low: outputs off
  double magnitude = pwmValue[pwmPin] / 255.0;
  return pinState[dirPin] == HIGH ? magnitude : -magnitude;
}

static int16_t gyroZRaw() {
  double left = trackCommand(6, 8), right = trackCommand(5, 7);
  double yawRate = 0.5 * MAX_YAW_RATE * (right - left);  // rad/s, CCW positive
  double raw = GYRO_BIAS_LSB + yawRate * (180.0 / PI) * 65.5;
  if (raw > 32767) raw = 32767;
  if (raw < -32768) raw = -32768;
  return (int16_t)raw;
}

// ---- Wire: an MPU6050 at 0x68 ----
static uint8_t wireReg = 0, wireOut[8];
static int wireOutLen = 0, wireOutPos = 0, wireTxCount = 0;

void TwoWire::begin() {}
void TwoWire::setClock(uint32_t) {}
void TwoWire::setWireTimeout(uint32_t, bool) {}
void TwoWire::beginTransmission(uint8_t) { wireTxCount = 0; }
size_t TwoWire::write(uint8_t value) {
  if (wireTxCount++ == 0) wireReg = value;  // first byte = register address
  return 1;
}
uint8_t TwoWire::endTransmission(bool) { return 0; }
uint8_t TwoWire::requestFrom(uint8_t address, uint8_t quantity) {
  if (address != 0x68) return 0;
  wireOutLen = wireOutPos = 0;
  for (uint8_t i = 0; i < quantity && i < sizeof(wireOut); i++) {
    uint8_t reg = wireReg + i, value = 0;
    int16_t g = gyroZRaw();
    if (reg == 0x75) value = 0x68;  // WHO_AM_I
    if (reg == 0x47) value = (uint8_t)((uint16_t)g >> 8);
    if (reg == 0x48) value = (uint8_t)(g & 0xFF);
    wireOut[wireOutLen++] = value;
  }
  return (uint8_t)wireOutLen;
}
int TwoWire::read() { return wireOutPos < wireOutLen ? wireOut[wireOutPos++] : -1; }

// ---- Serial over stdin / stdout ----
void HardwareSerial::begin(unsigned long) {}
int HardwareSerial::available() {
  if (rxHead == rxTail) {
    rxHead = rxTail = 0;
    ssize_t n = ::read(0, rxBuf, sizeof(rxBuf));
    if (n == 0) stdinEof = true;
    if (n > 0) rxTail = (int)n;
  }
  return rxTail - rxHead;
}
int HardwareSerial::read() { return available() > 0 ? rxBuf[rxHead++] : -1; }
size_t HardwareSerial::write(const uint8_t *buf, size_t n) { return fwrite(buf, 1, n, stdout); }

int main() {
  clock_gettime(CLOCK_MONOTONIC, &t0);
  fcntl(0, F_SETFL, fcntl(0, F_GETFL) | O_NONBLOCK);
  setvbuf(stdout, NULL, _IONBF, 0);
  setup();
  while (!stdinEof) {
    double t = nowUs();
    if (echoRiseAt >= 0 && t >= echoRiseAt) {
      echoRiseAt = -1;
      setEcho(HIGH);
    }
    if (echoFallAt >= 0 && echoRiseAt < 0 && t >= echoFallAt) {
      echoFallAt = -1;
      setEcho(LOW);
    }
    loop();
    usleep(50);
  }
  return 0;
}
