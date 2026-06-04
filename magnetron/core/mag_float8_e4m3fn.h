/*
** +---------------------------------------------------------------------+
** | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
** | Licensed under the Apache License, Version 2.0                      |
** |                                                                     |
** | Website : https://mariosieg.com                                     |
** | GitHub  : https://github.com/MarioSieg                              |
** | License : https://www.apache.org/licenses/LICENSE-2.0               |
** +---------------------------------------------------------------------+
*/

#ifndef MAG_FLOAT8_E4M3FN_H
#define MAG_FLOAT8_E4M3FN_H

#include "mag_def.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 8 bit floating point type - 4 exponent bits + 3 mantissa bits + sign, bias=7 */
typedef struct mag_float8_e4m3fn_t { uint8_t bits; } mag_float8_e4m3fn_t;
mag_static_assert(sizeof(mag_float8_e4m3fn_t) == 1);

#define mag_float8_e4m3fnc(x) (mag_float8_e4m3fn_t){(x)&255}

#define MAG_FLOAT8_E4M3FN_EPS mag_float8_e4m3fnc(0x20)
#define MAG_FLOAT8_E4M3FN_MAX mag_float8_e4m3fnc(0x7e)
#define MAG_FLOAT8_E4M3FN_MAX_SUBNORMAL mag_float8_e4m3fnc(0x07)
#define MAG_FLOAT8_E4M3FN_MIN mag_float8_e4m3fnc(0xfe)
#define MAG_FLOAT8_E4M3FN_MIN_POS mag_float8_e4m3fnc(0x08)
#define MAG_FLOAT8_E4M3FN_MIN_POS_SUBNORMAL mag_float8_e4m3fnc(0x01)
#define MAG_FLOAT8_E4M3FN_NAN mag_float8_e4m3fnc(0x7f)
#define MAG_FLOAT8_E4M3FN_NEG_ONE mag_float8_e4m3fnc(0xb8)
#define MAG_FLOAT8_E4M3FN_NEG_ZERO mag_float8_e4m3fnc(0x80)
#define MAG_FLOAT8_E4M3FN_ONE mag_float8_e4m3fnc(0x38)
#define MAG_FLOAT8_E4M3FN_ZERO mag_float8_e4m3fnc(0x00)

static MAG_AINLINE MAG_CUDA_DEVICE mag_float8_e4m3fn_t mag_float8_e4m3fn_from_float32_soft_fp(float x) {
  uint32_t b;
  memcpy(&b, &x, sizeof(b));
  uint32_t sgn = b&0x80000000u;
  b^=sgn;
  uint8_t r=0;
  if (b >= 0x43f00000u) {
      r = b > 0x7f800000u ? 0x7f : 0x7e;
  } else {
    if (b < 0x3c800000u) {
      uint32_t denorm_mask = 0x46800000;
      float t, dm;
      memcpy(&t, &b, sizeof(t));
      memcpy(&dm, &denorm_mask, sizeof(dm));
      t += dm;
      memcpy(&b, &t, sizeof(t));
      r = (uint8_t)(b-denorm_mask);
    } else {
      uint8_t mant = 1&(b>>20);
      b += ((uint32_t)(7-127)<<23)+0x7ffff;
      b += mant;
      r = (uint8_t)(b>>20);
      r = r == 0x7f ? 0x7e : r;
    }
  }
  r|=(uint8_t)(sgn>>24);
  return (mag_float8_e4m3fn_t){.bits=r};
}

static MAG_AINLINE MAG_CUDA_DEVICE float mag_float8_e4m3fn_to_float32_soft_fp(mag_float8_e4m3fn_t x) {
  uint32_t w = (uint32_t)x.bits<<24;
  uint32_t sgn = w & 0x80000000u;
  uint32_t dat = w & 0x7fffffffu;
  uint32_t renorm;
  #ifdef __CUDA_ARCH__
    renorm = __clz(dat);
  #elif defined(_MSC_VER)
    unsigned long bsr;
    _BitScanReverse(&bsr, (unsigned long)dat);
    renorm = 31^(uint32_t)bsr;
  #else
    renorm = dat ? __builtin_clz(dat) : sizeof(uint32_t)<<3;
  #endif
  renorm = renorm > 4 ? renorm-4 : 0;
  uint32_t r = sgn|((((dat<<renorm>>4)+((0x78-renorm)<<23))|(((int32_t)(dat+0x01000000)>>8) & 0x7f800000))&~((int32_t)(dat-1)>>31));
  float rf;
  memcpy(&rf, &r, sizeof(rf));
  return rf;
}

#ifdef __cplusplus
}
#endif

#endif
