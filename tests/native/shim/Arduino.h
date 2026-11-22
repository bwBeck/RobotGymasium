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

// F("text") keeps strings in flash on an AVR; here it is just the string.
class __FlashStringHelper;
#define F(string_literal) (reinterpret_cast<const __FlashStringHelper *>(string_literal))

class HardwareSerial {
 public:
  void begin(unsigned long baud);
  int available();
  int read();
  size_t write(const uint8_t *buf, size_t n);

  // print / println like Arduino's Print class: numbers in decimal, char as a character.
  size_t print(const char *s) { return write(reinterpret_cast<const uint8_t *>(s), strlen(s)); }
  size_t print(const __FlashStringHelper *s) { return print(reinterpret_cast<const char *>(s)); }
  size_t print(char c) { return write(reinterpret_cast<const uint8_t *>(&c), 1); }
  size_t print(unsigned long n) { return printf_("%lu", n); }
  size_t print(long n) { return printf_("%ld", n); }
  size_t print(unsigned int n) { return print((unsigned long)n); }
  size_t print(int n) { return print((long)n); }
  size_t print(unsigned char n) { return print((unsigned long)n); }
  size_t println() { return print("\r\n"); }
  template <typename T>
  size_t println(T value) {
    size_t n = print(value);
    return n + println();
  }

 private:
  template <typename T>
  size_t printf_(const char *format, T value) {
    char text[24];
    snprintf(text, sizeof(text), format, value);
    return print(text);
  }
};
extern HardwareSerial Serial;
