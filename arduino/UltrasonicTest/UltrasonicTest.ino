/*
  UltrasonicTest.ino - bench test for the Conqueror's HC-SR04 ultrasonic sensor.

  Flash this to the UNO in place of RoboGym.ino to find out whether the sensor, its cable
  or its connectors are at fault. It pings 10 times a second, prints every result to the
  Serial Monitor with the reason when a ping fails, and adds a summary with a verdict
  every 2 s.

  How to use
    1. Shield's Upload-Cam switch on "Upload", USB cable in, board "Arduino Uno", upload.
    2. Tools > Serial Monitor at 115200 baud. Keep the USB cable in: it is the only link.
    3. Point the sensor at a wall 0.3-2 m away. Every line should say OK, with a steady
       distance.
    4. Wiggle the cable and press on each plug and header, one spot at a time, and watch
       for failures that follow your hand. That spot is the bad contact.
    5. When done, re-flash arduino/RoboGym/RoboGym.ino and set the switch back to "Cam".

  Send in the Serial Monitor
    s  pan sweep on/off: the head turns 60-120 degrees while pinging, bending the cable
       the way driving does. Failures at certain angles point at the cable or its plugs;
       failures at every angle only while sweeping point at the 5 V supply (servo current).
    p  echo pull-up on/off (starts on). RoboGym.ino before the 2026-11-22 fix read echo
       without one. If pings only succeed with the pull-up on, the echo pulse's high level
       is weak: a worn contact in the echo or 5 V line. With it off, an open echo wire
       floats, so the results below can't single out a wire.
    f  ping every 60 ms, as RoboGym.ino does, or every 100 ms (the start)
    r  reset the counters

  What the results mean (with the pull-up on). The echo pin uses the UNO's internal pull-up, so a powered,
  connected sensor holds it low while idle and a broken wire reads high instead of noise.
    OK          echo pulse 116 us - 23.5 ms: a range of 2 cm - 4 m.
    NO ANSWER   echo idled low (so the sensor has power and the echo wire is fine) but
                never went high within 30 ms of the trigger: the trigger wire is open, or
                the sensor is dead. RoboGym reports this as "no distance reading".
    ECHO HIGH   echo was already high before the trigger, so nothing holds it low: the echo
                wire or the sensor's 5 V / GND wire is open, or the sensor locked up.
                RoboGym reports this as "no distance reading" too.
    NO ECHO     a pulse of 23.5-250 ms: the sensor works but heard nothing within 4 m
                (open space, or a soft or angled surface). Not a wiring fault.
    STUCK HIGH  echo went high after the trigger and stayed high over 250 ms: the sensor
                locked up (power-cycle the robot), or echo is shorted to 5 V.
    GLITCH      a pulse under 116 us (< 2 cm): noise, typically a contact that bounces.

  The motors are held off the whole time (TB6612 standby low, both PWM pins low).
*/

#include <Arduino.h>
#include <Servo.h>

// Pins, as in RoboGym.ino.
const uint8_t PIN_TRIG = 13;
const uint8_t PIN_ECHO = 12;
const uint8_t PIN_SERVO_PAN = 10;
const uint8_t PIN_STBY = 3;  // TB6612 standby; unused on DRV8835 shields
const uint8_t PIN_PWM_RIGHT = 5;
const uint8_t PIN_PWM_LEFT = 6;

const unsigned long BAUD = 115200;
const uint32_t SLOW_PERIOD_MS = 100;  // the start
const uint32_t FAST_PERIOD_MS = 60;   // RoboGym.ino's; the HC-SR04 wants >= 60 ms between pings
const uint32_t RISE_TIMEOUT_US = 30000;
const uint32_t MIN_VALID_US = 116;    // 2 cm, the HC-SR04's minimum range
const uint32_t MAX_RANGE_US = 23500;  // 4 m
const uint32_t STUCK_US = 250000;     // longer than any clone holds a no-echo pulse
const float MM_PER_US = 0.1715f;      // speed of sound / 2 at ~20 C
const uint8_t SUMMARY_EVERY = 20;     // pings: 2 s
const uint32_t BAR_MAX_MM = 2000;     // the bar graph's full width
const uint8_t BAR_WIDTH = 40;
const uint8_t PAN_CENTER = 90, PAN_MIN = 60, PAN_MAX = 120, PAN_STEP = 3;

enum Result : uint8_t { OK, NO_ANSWER, ECHO_HIGH, NO_ECHO, STUCK_HIGH, GLITCH, RESULT_COUNT };
const char *const RESULT_NAMES[RESULT_COUNT] = {"OK", "NO ANSWER", "ECHO HIGH", "NO ECHO",
                                                "STUCK HIGH", "GLITCH"};

Servo pan;
bool echoPullup = true;
uint32_t pingPeriodMs = SLOW_PERIOD_MS;
bool sweeping = false;
uint8_t panAngle = PAN_CENTER;
int8_t panDir = 1;

uint32_t pingNumber = 0;
uint32_t lastPingMs = 0;
// Since the last summary.
uint8_t pings = 0;
uint8_t counts[RESULT_COUNT];
uint16_t okMinMm, okMaxMm;
uint32_t okSumMm;

static void resetStats() {
  pings = 0;
  for (uint8_t i = 0; i < RESULT_COUNT; i++) counts[i] = 0;
  okMinMm = 0xFFFF;
  okMaxMm = 0;
  okSumMm = 0;
}

static uint16_t widthToMm(uint32_t width) { return (uint16_t)(width * MM_PER_US + 0.5f); }

// One ping. Returns the result; `width` gets the echo pulse length in us (0 if none).
static Result ping(uint32_t &width) {
  width = 0;
  if (digitalRead(PIN_ECHO) == HIGH) return ECHO_HIGH;  // nothing holds it low: can't trigger
  digitalWrite(PIN_TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(PIN_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(PIN_TRIG, LOW);

  uint32_t start = micros();
  while (digitalRead(PIN_ECHO) == LOW) {
    if (micros() - start > RISE_TIMEOUT_US) return NO_ANSWER;
  }
  uint32_t rise = micros();
  while (digitalRead(PIN_ECHO) == HIGH) {
    if (micros() - rise > STUCK_US) {
      width = micros() - rise;
      return STUCK_HIGH;
    }
  }
  width = micros() - rise;
  if (width < MIN_VALID_US) return GLITCH;
  if (width > MAX_RANGE_US) return NO_ECHO;
  return OK;
}

static void printLine(Result result, uint32_t width) {
  char buf[40];
  snprintf(buf, sizeof(buf), "#%6lu  ", (unsigned long)pingNumber);
  Serial.print(buf);
  if (sweeping) {
    snprintf(buf, sizeof(buf), "pan %3u  ", panAngle);
    Serial.print(buf);
  }
  if (result != OK && width == 0) {  // nothing more on this line: no padding
    Serial.println(RESULT_NAMES[result]);
    return;
  }
  snprintf(buf, sizeof(buf), "%-10s", RESULT_NAMES[result]);
  Serial.print(buf);
  if (result == OK) {
    uint16_t mm = widthToMm(width);
    snprintf(buf, sizeof(buf), "  %4u mm  %6lu us  |", mm, (unsigned long)width);
    Serial.print(buf);
    uint32_t bar = (mm < BAR_MAX_MM ? mm : BAR_MAX_MM) * BAR_WIDTH / BAR_MAX_MM;
    for (uint8_t i = 0; i < bar; i++) Serial.print('#');
  } else if (width > 0) {  // NO ECHO, STUCK HIGH, GLITCH: the pulse it did see
    snprintf(buf, sizeof(buf), "           %6lu us", (unsigned long)width);
    Serial.print(buf);
  }
  Serial.println();
}

static void printVerdict() {
  uint8_t ok = counts[OK];
  uint8_t wiring = counts[NO_ANSWER] + counts[ECHO_HIGH] + counts[GLITCH];
  Serial.print(F("    -> "));
  if (ok == pings) {
    Serial.println(F("all OK: sensor, cable and plugs look fine"));
  } else if (counts[STUCK_HIGH] > 0) {
    Serial.println(F("echo stuck high: the sensor locked up - power-cycle the robot. If it "
                     "keeps happening, look for echo shorted to 5 V"));
  } else if (counts[ECHO_HIGH] == pings) {
    Serial.println(F("nothing holds echo low: the echo wire or the sensor's 5 V / GND wire is "
                     "open, or the sensor locked up. Power-cycle the robot; if that doesn't "
                     "help, check the cable, both plugs and the header solder joints"));
  } else if (counts[NO_ANSWER] == pings) {
    Serial.println(F("sensor powered and echo wire fine, but it ignores the trigger: the "
                     "trigger wire is open (cable, plug, solder joint) or the sensor is dead"));
  } else if (wiring > 0) {
    Serial.print(F("INTERMITTENT: a contact keeps making and breaking ("));
    if (counts[NO_ANSWER] > 0) Serial.print(F("trigger line"));
    if (counts[NO_ANSWER] > 0 && counts[ECHO_HIGH] + counts[GLITCH] > 0) Serial.print(F(", "));
    if (counts[ECHO_HIGH] > 0) Serial.print(F("echo or power line"));
    if (counts[ECHO_HIGH] > 0 && counts[GLITCH] > 0) Serial.print(F(", "));
    if (counts[GLITCH] > 0) Serial.print(F("echo line bouncing"));
    Serial.println(F("). Wiggle the cable and plugs to find the spot"));
  } else {
    Serial.println(F("NO ECHO: nothing heard within 4 m (open space, or a soft or angled "
                     "surface). Point it at a flat wall. Not a wiring fault"));
  }
}

static void printSummary() {
  Serial.print(F("--- last "));
  Serial.print(pings);
  Serial.print(F(" pings:"));
  bool first = true;
  for (uint8_t i = 0; i < RESULT_COUNT; i++) {
    if (counts[i] == 0) continue;
    Serial.print(first ? F(" ") : F(", "));
    Serial.print(counts[i]);
    Serial.print(' ');
    Serial.print(RESULT_NAMES[i]);
    first = false;
  }
  if (counts[OK] > 0) {
    Serial.print(F(" | OK range "));
    Serial.print(okMinMm);
    Serial.print(F("-"));
    Serial.print(okMaxMm);
    Serial.print(F(" mm, mean "));
    Serial.print(okSumMm / counts[OK]);
    Serial.print(F(" mm"));
  }
  Serial.print(F(" [pull-up "));
  Serial.print(echoPullup ? F("on") : F("off"));
  Serial.print(F(", every "));
  Serial.print(pingPeriodMs);
  Serial.println(F(" ms]"));
  printVerdict();
  if (!echoPullup && counts[OK] < pings) {
    Serial.println(F("    (pull-up off: if pings succeed with it on - press p - the echo high "
                     "level is weak: a worn contact in the echo or 5 V line)"));
  }
  resetStats();
}

static void setEchoPullup(bool on) {
  echoPullup = on;
  pinMode(PIN_ECHO, on ? INPUT_PULLUP : INPUT);
  Serial.println(on ? F("echo pull-up ON")
                    : F("echo pull-up OFF, as older RoboGym.ino: a weak echo high level now reads low"));
}

static void setSweep(bool on) {
  if (on == sweeping) return;
  sweeping = on;
  if (on) {
    pan.attach(PIN_SERVO_PAN);
    Serial.println(F("pan sweep ON: the head turns 60-120 degrees while pinging"));
  } else {
    panAngle = PAN_CENTER;
    pan.write(PAN_CENTER);
    delay(400);  // let it get there before the servo goes limp
    pan.detach();
    Serial.println(F("pan sweep OFF: head centred"));
  }
}

static void stepSweep() {
  if (!sweeping) return;
  if (panAngle + PAN_STEP > PAN_MAX) panDir = -1;
  if (panAngle < PAN_MIN + PAN_STEP) panDir = 1;
  panAngle += panDir * PAN_STEP;
  pan.write(panAngle);
}

static void handleSerial() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == 's' || c == 'S') {
      setSweep(!sweeping);
    } else if (c == 'p' || c == 'P') {
      setEchoPullup(!echoPullup);
      resetStats();
    } else if (c == 'f' || c == 'F') {
      pingPeriodMs = pingPeriodMs == SLOW_PERIOD_MS ? FAST_PERIOD_MS : SLOW_PERIOD_MS;
      Serial.print(F("pinging every "));
      Serial.print(pingPeriodMs);
      Serial.println(F(" ms"));
      resetStats();
    } else if (c == 'r' || c == 'R') {
      resetStats();
      Serial.println(F("counters reset"));
    }
  }
}

void setup() {
  // Motors off for good: TB6612 in standby, both PWM low.
  pinMode(PIN_STBY, OUTPUT);
  digitalWrite(PIN_STBY, LOW);
  pinMode(PIN_PWM_LEFT, OUTPUT);
  digitalWrite(PIN_PWM_LEFT, LOW);
  pinMode(PIN_PWM_RIGHT, OUTPUT);
  digitalWrite(PIN_PWM_RIGHT, LOW);

  pinMode(PIN_TRIG, OUTPUT);
  digitalWrite(PIN_TRIG, LOW);
  pinMode(PIN_ECHO, INPUT_PULLUP);  // a broken echo wire reads high, not random noise
  resetStats();

  Serial.begin(BAUD);
  Serial.println();
  Serial.println(F("UltrasonicTest: HC-SR04 trigger D13, echo D12, 10 pings/s, summary every 2 s."));
  Serial.println(F("Point the sensor at a wall 0.3-2 m away, then wiggle the cable and plugs one "
                   "spot at a time."));
  Serial.println(F("Send 's' = pan sweep on/off, 'p' = echo pull-up on/off, 'f' = 60 / 100 ms "
                   "between pings, 'r' = reset counters."));
  delay(50);
  Serial.print(F("echo line at rest: "));
  Serial.println(digitalRead(PIN_ECHO) == LOW
                     ? F("LOW - the sensor is holding it low (powered, echo wire connected)")
                     : F("HIGH - nothing holds it low: echo or power wire open, or sensor locked up"));
  Serial.println();
}

void loop() {
  handleSerial();
  if (millis() - lastPingMs < pingPeriodMs) return;
  lastPingMs = millis();

  stepSweep();
  uint32_t width;
  Result result = ping(width);
  pingNumber++;
  pings++;
  counts[result]++;
  if (result == OK) {
    uint16_t mm = widthToMm(width);
    if (mm < okMinMm) okMinMm = mm;
    if (mm > okMaxMm) okMaxMm = mm;
    okSumMm += mm;
  }
  printLine(result, width);
  if (pings >= SUMMARY_EVERY) printSummary();
}
