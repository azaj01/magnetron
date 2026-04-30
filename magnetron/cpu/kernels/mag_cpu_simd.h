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

#ifndef MAG_SIMD_H
#define MAG_SIMD_H

#include <core/mag_def.h>

#include <core/mag_bfloat16.h>

#include <float.h>
#include <math.h>

#ifdef _MSC_VER
  #include <intrin.h>
#else
  #if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
  #elif (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    #include <arm_neon.h>
    #include <arm_acle.h>
  #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)

  /* mask type */
  typedef uint32x4_t mag_vmask32_t;
  static MAG_AINLINE mag_vmask32_t mag_vmask32_zero(void) { return vdupq_n_u32(0); }
  static MAG_AINLINE mag_vmask32_t mag_vmask32_full(void) { return vdupq_n_u32(~0u); }
  static MAG_AINLINE mag_vmask32_t mag_vmask32_not(mag_vmask32_t x) { return vmvnq_u32(x); }
  static MAG_AINLINE mag_vmask32_t mag_vmask32_and(mag_vmask32_t x, mag_vmask32_t y) { return vandq_u32(x, y); }
  static MAG_AINLINE mag_vmask32_t mag_vmask32_or(mag_vmask32_t x, mag_vmask32_t y) { return vorrq_u32(x, y); }
  static MAG_AINLINE mag_vmask32_t mag_vmask32_xor(mag_vmask32_t x, mag_vmask32_t y) { return veorq_u32(x, y); }
  static MAG_AINLINE mag_vmask32_t mag_vmask32_andnot(mag_vmask32_t x, mag_vmask32_t y) { return vbicq_u32(y, x); }
  static MAG_AINLINE int mag_vmask32_any(mag_vmask32_t x) { return vmaxvq_u32(x) != 0; }
  static MAG_AINLINE int mag_vmask32_all(mag_vmask32_t x) { return vminvq_u32(x) == ~0u; }

  /* f32 vector */
  typedef float32x4_t mag_vf32_t;
  static MAG_AINLINE mag_vf32_t mag_vf32_zero(void) { return vdupq_n_f32(0.f); }
  static MAG_AINLINE mag_vf32_t mag_vf32_splat(float x) { return vdupq_n_f32(x); }
  static MAG_AINLINE mag_vf32_t mag_vf32_broadcast(const float *p) { return vdupq_n_f32(*p); }
  static MAG_AINLINE mag_vf32_t mag_vf32_loada(const float *p) { return vld1q_f32(p); }
  static MAG_AINLINE mag_vf32_t mag_vf32_loadu(const float *p) { return vld1q_f32(p); }
  static MAG_AINLINE mag_vf32_t mag_vf32_loadu_masked(const float *p, int n) {
    mag_alignas(16) float tmp[4] = {0};
    for (int i=0; i < n; ++i) tmp[i] = p[i];
    return vld1q_f32(tmp);
  }
  static MAG_AINLINE void mag_vf32_storea(float *p, mag_vf32_t v) { vst1q_f32(p, v); }
  static MAG_AINLINE void mag_vf32_storeu(float *p, mag_vf32_t v) { vst1q_f32(p, v); }
  static MAG_AINLINE void mag_vf32_storeu_masked(float *p, mag_vf32_t v, int n) {
      mag_alignas(16) float tmp[4];
      vst1q_f32(tmp, v);
      for (int i=0; i < n; ++i) p[i] = tmp[i];
  }
  static MAG_AINLINE mag_vmask32_t mag_vf32_cmpeq(mag_vf32_t x, mag_vf32_t y) { return vceqq_f32(x, y); }
  static MAG_AINLINE mag_vmask32_t mag_vf32_cmpne(mag_vf32_t x, mag_vf32_t y) { return vmvnq_u32(vceqq_f32(x, y)); }
  static MAG_AINLINE mag_vmask32_t mag_vf32_cmplt(mag_vf32_t x, mag_vf32_t y) { return vcltq_f32(x, y); }
  static MAG_AINLINE mag_vmask32_t mag_vf32_cmple(mag_vf32_t x, mag_vf32_t y) { return vcleq_f32(x, y); }
  static MAG_AINLINE mag_vmask32_t mag_vf32_cmpgt(mag_vf32_t x, mag_vf32_t y) { return vcgtq_f32(x, y); }
  static MAG_AINLINE mag_vmask32_t mag_vf32_cmpge(mag_vf32_t x, mag_vf32_t y) { return vcgeq_f32(x, y); }
  static MAG_AINLINE mag_vf32_t mag_vf32_blend(mag_vmask32_t m, mag_vf32_t t, mag_vf32_t f) { return vbslq_f32(m, t, f); }
  static MAG_AINLINE mag_vmask32_t mag_vmask32_reinterpret_from_vf32(mag_vf32_t x) { return vreinterpretq_u32_f32(x); }
  static MAG_AINLINE mag_vf32_t mag_vf32_reinterpret_from_vmask32(mag_vmask32_t x) { return vreinterpretq_f32_u32(x); }
  static MAG_AINLINE mag_vf32_t mag_vf32_add(mag_vf32_t x, mag_vf32_t y) { return vaddq_f32(x, y); }
  static MAG_AINLINE mag_vf32_t mag_vf32_sub(mag_vf32_t x, mag_vf32_t y) { return vsubq_f32(x, y); }
  static MAG_AINLINE mag_vf32_t mag_vf32_mul(mag_vf32_t x, mag_vf32_t y) { return vmulq_f32(x, y); }
  static MAG_AINLINE mag_vf32_t mag_vf32_div(mag_vf32_t x, mag_vf32_t y) { return vdivq_f32(x, y); }
  static MAG_AINLINE mag_vf32_t mag_vf32_min(mag_vf32_t x, mag_vf32_t y) { return vminq_f32(x, y); }
  static MAG_AINLINE mag_vf32_t mag_vf32_max(mag_vf32_t x, mag_vf32_t y) { return vmaxq_f32(x, y); }
  static MAG_AINLINE mag_vf32_t mag_vf32_fmadd(mag_vf32_t x, mag_vf32_t y, mag_vf32_t z) { return vfmaq_f32(z, x, y); }
  static MAG_AINLINE mag_vf32_t mag_vf32_fnmadd(mag_vf32_t x, mag_vf32_t y, mag_vf32_t z) { return vfmsq_f32(z, x, y); }
  static MAG_AINLINE float mag_vf32_reduce_add(mag_vf32_t x) { return vaddvq_f32(x); }
  static MAG_AINLINE mag_vf32_t mag_vf32_abs(mag_vf32_t x) { return vabsq_f32(x); }
  static MAG_AINLINE mag_vf32_t mag_vf32_rcp_approx(mag_vf32_t x) { return vrecpeq_f32(x); }
  static MAG_AINLINE mag_vf32_t mag_vf32_rcp_refine_step(mag_vf32_t x, mag_vf32_t r) { return vmulq_f32(vrecpsq_f32(x, r), r); }
  static MAG_AINLINE mag_vf32_t mag_vf32_and_bits(mag_vf32_t x, mag_vf32_t y) { return vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x), vreinterpretq_u32_f32(y))); }
  static MAG_AINLINE mag_vf32_t mag_vf32_or_bits(mag_vf32_t x, mag_vf32_t y) { return vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(x), vreinterpretq_u32_f32(y))); }
  static MAG_AINLINE mag_vf32_t mag_vf32_not_bits(mag_vf32_t x) { return vreinterpretq_f32_u32(vmvnq_u32(vreinterpretq_u32_f32(x))); }
  static MAG_AINLINE mag_vf32_t mag_vf32_andnot_bits(mag_vf32_t x, mag_vf32_t y) { return vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(y), vreinterpretq_u32_f32(x))); }
  static MAG_AINLINE mag_vf32_t mag_vf32_xor_bits(mag_vf32_t x, mag_vf32_t y) { return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(x), vreinterpretq_u32_f32(y))); }

  /* i32 vector */
  typedef int32x4_t mag_vi32_t;
  static MAG_AINLINE mag_vi32_t mag_vi32_zero(void) { return vdupq_n_s32(0); }
  static MAG_AINLINE mag_vi32_t mag_vi32_splat(int32_t x) { return vdupq_n_s32(x); }
  static MAG_AINLINE mag_vi32_t mag_vi32_broadcast(const int32_t *p) { return vdupq_n_s32(*p); }
  static MAG_AINLINE mag_vi32_t mag_vi32_loada(const int32_t *p) { return vld1q_s32(p); }
  static MAG_AINLINE mag_vi32_t mag_vi32_loadu(const int32_t *p) { return vld1q_s32(p); }
  static MAG_AINLINE mag_vi32_t mag_vi32_loadu_masked(const int32_t *p, int n) {
    mag_alignas(16) int32_t tmp[4] = {0};
    for (int i=0; i < n; ++i) tmp[i] = p[i];
    return vld1q_s32(tmp);
  }
  static MAG_AINLINE void mag_vi32_storea(int32_t *p, mag_vi32_t v) { vst1q_s32(p, v); }
  static MAG_AINLINE void mag_vi32_storeu(int32_t *p, mag_vi32_t v) { vst1q_s32(p, v); }
  static MAG_AINLINE void mag_vi32_storeu_masked(int32_t *p, mag_vi32_t v, int n) {
    mag_alignas(16) int32_t tmp[4];
    vst1q_s32(tmp, v);
    for (int i=0; i < n; ++i) p[i] = tmp[i];
  }
  static MAG_AINLINE mag_vmask32_t mag_vi32_cmpeq(mag_vi32_t x, mag_vi32_t y) { return vceqq_s32(x, y); }
  static MAG_AINLINE mag_vmask32_t mag_vi32_cmpne(mag_vi32_t x, mag_vi32_t y) { return vmvnq_u32(vceqq_s32(x, y)); }
  static MAG_AINLINE mag_vmask32_t mag_vi32_cmplt(mag_vi32_t x, mag_vi32_t y) { return vcltq_s32(x, y); }
  static MAG_AINLINE mag_vmask32_t mag_vi32_cmple(mag_vi32_t x, mag_vi32_t y) { return vcleq_s32(x, y); }
  static MAG_AINLINE mag_vmask32_t mag_vi32_cmpgt(mag_vi32_t x, mag_vi32_t y) { return vcgtq_s32(x, y); }
  static MAG_AINLINE mag_vmask32_t mag_vi32_cmpge(mag_vi32_t x, mag_vi32_t y) { return vcgeq_s32(x, y); }
  static MAG_AINLINE mag_vi32_t mag_vi32_blend(mag_vmask32_t m, mag_vi32_t t, mag_vi32_t f) { return vbslq_s32(m, t, f); }
  static MAG_AINLINE mag_vmask32_t mag_vmask32_reinterpret_from_vi32(mag_vi32_t x) { return vreinterpretq_u32_s32(x); }
  static MAG_AINLINE mag_vi32_t mag_vi32_reinterpret_from_vmask32(mag_vmask32_t x) { return vreinterpretq_s32_u32(x); }
  static MAG_AINLINE mag_vi32_t mag_vi32_add(mag_vi32_t x, mag_vi32_t y) { return vaddq_s32(x, y); }
  static MAG_AINLINE mag_vi32_t mag_vi32_sub(mag_vi32_t x, mag_vi32_t y) { return vsubq_s32(x, y); }
  static MAG_AINLINE mag_vi32_t mag_vi32_mul(mag_vi32_t x, mag_vi32_t y) { return vmulq_s32(x, y); }
  static MAG_AINLINE mag_vi32_t mag_vi32_and(mag_vi32_t x, mag_vi32_t y) { return vreinterpretq_s32_u32(vandq_u32(vreinterpretq_u32_s32(x), vreinterpretq_u32_s32(y))); }
  static MAG_AINLINE mag_vi32_t mag_vi32_or(mag_vi32_t x, mag_vi32_t y) {return vreinterpretq_s32_u32(vorrq_u32(vreinterpretq_u32_s32(x), vreinterpretq_u32_s32(y)));}
  static MAG_AINLINE mag_vi32_t mag_vi32_xor(mag_vi32_t x, mag_vi32_t y) {return vreinterpretq_s32_u32(veorq_u32(vreinterpretq_u32_s32(x), vreinterpretq_u32_s32(y)));}
  static MAG_AINLINE mag_vi32_t mag_vi32_andnot(mag_vi32_t x, mag_vi32_t y) { return vreinterpretq_s32_u32(vbicq_u32(vreinterpretq_u32_s32(y), vreinterpretq_u32_s32(x))); }
  static MAG_AINLINE mag_vi32_t mag_vi32_not(mag_vi32_t x) { return vreinterpretq_s32_u32(vmvnq_u32(vreinterpretq_u32_s32(x))); }
  static MAG_AINLINE mag_vi32_t mag_vi32_slli(mag_vi32_t x, int n) { return vreinterpretq_s32_u32(vshlq_u32(vreinterpretq_u32_s32(x), vdupq_n_s32(n))); }
  static MAG_AINLINE mag_vi32_t mag_vi32_srli(mag_vi32_t x, int n) { return vreinterpretq_s32_u32(vshlq_u32(vreinterpretq_u32_s32(x), vdupq_n_s32(-n)));}
  static MAG_AINLINE mag_vi32_t mag_vi32_srai(mag_vi32_t x, int n) { return vshlq_s32(x, vdupq_n_s32(-n)); }
  static MAG_AINLINE mag_vi32_t mag_vi32_min(mag_vi32_t x, mag_vi32_t y) { return vminq_s32(x, y); }
  static MAG_AINLINE mag_vi32_t mag_vi32_max(mag_vi32_t x, mag_vi32_t y) { return vmaxq_s32(x, y); }
  static MAG_AINLINE int32_t mag_vi32_reduce_add(mag_vi32_t x) { return vaddvq_s32(x); }
  static MAG_AINLINE mag_vf32_t mag_vf32_reinterpret_from_vi32(mag_vi32_t x) { return vreinterpretq_f32_s32(x); }
  static MAG_AINLINE mag_vi32_t mag_vi32_reinterpret_from_vf32(mag_vf32_t x) { return vreinterpretq_s32_f32(x); }
  static MAG_AINLINE mag_vf32_t mag_vi32_to_f32(mag_vi32_t x) { return vcvtq_f32_s32(x); }
  static MAG_AINLINE mag_vi32_t mag_vf32_trunc_to_vi32(mag_vf32_t x) { return vcvtq_s32_f32(x); }

#endif

/* Prefetching */

#if defined(__x86_64__) || defined(_M_X64)
    static MAG_AINLINE void mag_simd_prefetch_t0(const void *p) { _mm_prefetch((const char *)p, _MM_HINT_T0); }
    static MAG_AINLINE void mag_simd_prefetch_t1(const void *p) { _mm_prefetch((const char *)p, _MM_HINT_T1); }
    static MAG_AINLINE void mag_simd_prefetch_t2(const void *p) { _mm_prefetch((const char *)p, _MM_HINT_T2); }
    static MAG_AINLINE void mag_simd_prefetch_nta(const void *p) { _mm_prefetch((const char *)p, _MM_HINT_NTA); }
#elif defined(__GNUC__) || defined(__clang__)
    static MAG_AINLINE void mag_simd_prefetch_t0(const void *p) { __builtin_prefetch(p, 0, 3); }
    static MAG_AINLINE void mag_simd_prefetch_t1(const void *p) { __builtin_prefetch(p, 0, 2); }
    static MAG_AINLINE void mag_simd_prefetch_t2(const void *p) { __builtin_prefetch(p, 0, 1); }
    static MAG_AINLINE void mag_simd_prefetch_nta(const void *p) { __builtin_prefetch(p, 0, 0); }
#else
    static MAG_AINLINE void mag_simd_prefetch_t0(const void *p) { (void)p; }
    static MAG_AINLINE void mag_simd_prefetch_t1(const void *p) { (void)p; }
    static MAG_AINLINE void mag_simd_prefetch_t2(const void *p) { (void)p; }
    static MAG_AINLINE void mag_simd_prefetch_nta(const void *p) { (void)p; }
#endif

#if defined(__GNUC__) || defined(__clang__)
    static MAG_AINLINE void mag_simd_prefetchw(const void *p) { __builtin_prefetch(p, 1, 3); }
#else
    static MAG_AINLINE void mag_simd_prefetchw(const void *p) { (void)p; }
#endif

#ifdef __cplusplus
}
#endif

#endif