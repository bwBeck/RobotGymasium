// Runs arduino/UltrasonicTest/UltrasonicTest.ino as a normal program for tests, against a
// fake HC-SR04 with an injectable fault, on a simulated clock (instant and repeatable).
//
//   ultrasonic_native <mode> [serial input] [seconds]
//
// Modes: ok (a wall 0.5 m away), out_of_range (a clone's 38 ms no-echo pulse),
// trigger_open (the sensor never sees the trigger), echo_open (nothing drives echo: the
// pull-up reads high), flapping (the trigger contact drops every 4th ping), bouncing (every
// other echo is a 40 us blip), stuck (the first echo never ends), weak_high (the pulse's
// high level only reads high with the pull-up on). The serial input is what
// the sketch reads from Serial, e.g. "s" for the pan sweep. Runs 2.3 simulated seconds
// (20+ pings, so one summary) unless told otherwise.
#include <Arduino.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void setup();
void loop();

HardwareSerial Serial;
uint8_t PCICR = 0;  // declared by the shim for RoboGym.ino
uint8_t PCMSK0 = 0;

static uint64_t nowUs = 0;  // every Arduino call takes a few us, as on an UNO
static const char *mode = "ok";
static const char *input = "";
static uint8_t trigLevel = LOW;
static bool echoPullup = false;
static uint32_t triggers = 0;
static int64_t riseAt = -1, fallAt = -1;  // the current echo pulse, -1 = none

static bool is(const char *name) { return strcmp(mode, name) == 0; }

// The trigger's falling edge: the sensor sends its burst and schedules the echo pulse.
static void onTrigger() {
  triggers++;
  if (is("trigger_open") || (is("flapping") && triggers % 4 == 0)) return;
  int64_t width = 2915;  // 500 mm
  if (is("out_of_range")) width = 38000;
  if (is("bouncing") && triggers % 2 == 0) width = 40;
  riseAt = (int64_t)nowUs + 450;
  fallAt = is("stuck") ? INT64_MAX : riseAt + width;
}

static int echoLevel() {
  if (is("echo_open")) return echoPullup ? HIGH : LOW;  // floating without it; LOW stands in
  int64_t t = (int64_t)nowUs;
  bool pulse = riseAt >= 0 && t >= riseAt && t < fallAt;
  if (is("weak_high")) pulse = pulse && echoPullup;  // below the UNO's 3 V threshold without it
  return pulse ? HIGH : LOW;
}

unsigned long micros() {
  nowUs += 4;
  return (unsigned long)(uint32_t)nowUs;
}
unsigned long millis() {
  nowUs += 4;
  return (unsigned long)(uint32_t)(nowUs / 1000);
}
void delay(unsigned long ms) { nowUs += ms * 1000ULL; }
void delayMicroseconds(unsigned int us) { nowUs += us; }
void pinMode(uint8_t pin, uint8_t mode) {
  if (pin == 12) echoPullup = mode == INPUT_PULLUP;
}
void digitalWrite(uint8_t pin, uint8_t val) {
  nowUs += 4;
  if (pin == 13) {
    if (trigLevel == HIGH && val == LOW) onTrigger();
    trigLevel = val;
  }
}
int digitalRead(uint8_t pin) {
  nowUs += 4;
  return pin == 12 ? echoLevel() : LOW;
}
int analogRead(uint8_t) { return 0; }
void analogWrite(uint8_t, int) {}

void HardwareSerial::begin(unsigned long) {}
int HardwareSerial::available() { return (int)strlen(input); }
int HardwareSerial::read() { return *input ? *input++ : -1; }
size_t HardwareSerial::write(const uint8_t *buf, size_t n) { return fwrite(buf, 1, n, stdout); }

int main(int argc, char **argv) {
  if (argc > 1) mode = argv[1];
  if (argc > 2) input = argv[2];
  uint64_t endUs = (uint64_t)((argc > 3 ? atof(argv[3]) : 2.3) * 1e6);
  setup();
  while (nowUs < endUs) loop();
  return 0;
}
