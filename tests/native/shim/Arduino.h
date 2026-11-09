// Minimal host-side stand-in for the Arduino core, just enough to compile and run
// arduino/RoboGym/RoboGym.ino on a PC for tests (see tests/native/sim_main.cpp).
#pragma once
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef bool boolean;
typedef uint8_t byte;

#define HIGH 0x1
#define LOW 0x0
#define INPUT 0x0
#define OUTPUT 0x1
#define INPUT_PULLUP 0x2
#define PI 3.1415926535897932384626433832795
#define TWO_PI 6.283185307179586476925286766559
#define A0 14
#define A1 15
#define A2 16
#define A3 17
#define A4 18
#define A5 19
#define _BV(bit) (1 << (bit))

void pinMode(uint8_t pin, uint8_t mode);
void digitalWrite(uint8_t pin, uint8_t val);
int digitalRead(uint8_t pin);
int analogRead(uint8_t pin);
void analogWrite(uint8_t pin, int val);
unsigned long millis();
unsigned long micros();
void delay(unsigned long ms);
void delayMicroseconds(unsigned int us);

// Pin-change interrupt registers the sketch touches.
extern uint8_t PCICR;
extern uint8_t PCMSK0;
#define PCIE0 0
#define PCINT4 4
#define ISR(vector) void vector##_handler()

class HardwareSerial {
 public:
  void begin(unsigned long baud);
  int available();
  int read();
  size_t write(const uint8_t *buf, size_t n);
};
extern HardwareSerial Serial;
