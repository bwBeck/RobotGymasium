// Host-side stand-in for the Servo library (the sketch only centres the servos at boot).
#pragma once
#include <stdint.h>

class Servo {
 public:
  uint8_t attach(int) { return 0; }
  void write(int) {}
  void detach() {}
};
