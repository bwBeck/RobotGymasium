// Host-side stand-in for the Wire (I2C) library; the device model is in sim_main.cpp.
#pragma once
#include <stddef.h>
#include <stdint.h>

#define WIRE_HAS_TIMEOUT 1

class TwoWire {
 public:
  void begin();
  void setClock(uint32_t hz);
  void setWireTimeout(uint32_t timeout_us = 25000, bool reset_with_timeout = false);
  void beginTransmission(uint8_t address);
  size_t write(uint8_t value);
  uint8_t endTransmission(bool stop = true);
  uint8_t requestFrom(uint8_t address, uint8_t quantity);
  int read();
};
extern TwoWire Wire;
