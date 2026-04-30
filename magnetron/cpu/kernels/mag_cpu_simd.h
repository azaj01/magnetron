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
  typedef uint32x4_t mag_vmask32_t;
  typedef float32x4_t mag_vf32_t;
  typedef int32x4_t mag_vi32_t;
#elif defined(__AVX512F__)
  typedef __mmask16 mag_vmask32_t;
  typedef __m512 mag_vf32_t;
  typedef __m512i mag_vi32_t;
#else /* Scalar fallback path */
  typedef uint32_t mag_vmask32_t;
  typedef float mag_vf32_t;
  typedef int32_t mag_vi32_t;
#endif

#define MAG_VF32_LANES ((int64_t)(sizeof(mag_vf32_t)/sizeof(float)))

/* mask type */
static MAG_AINLINE mag_vmask32_t mag_vmask32_zero(void) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
      return vdupq_n_u32(0);
  #elif defined(__AVX512F__)
      return 0;
  #else
      return 0;
  #endif
}
static MAG_AINLINE mag_vmask32_t mag_vmask32_full(void) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vdupq_n_u32(~0u);
  #elif defined(__AVX512F__)
    return 0xffff;
  #else
    return ~0u;
  #endif
}
static MAG_AINLINE mag_vmask32_t mag_vmask32_not(mag_vmask32_t x) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vmvnq_u32(x);
  #elif defined(__AVX512F__)
    return ~x&0xffff;
  #else
    return ~x;
  #endif
}
static MAG_AINLINE mag_vmask32_t mag_vmask32_and(mag_vmask32_t x, mag_vmask32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vandq_u32(x, y);
  #else
    return x&y;
  #endif
}
static MAG_AINLINE mag_vmask32_t mag_vmask32_or(mag_vmask32_t x, mag_vmask32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
      return vorrq_u32(x, y);
  #else
      return x|y;
  #endif
}
static MAG_AINLINE mag_vmask32_t mag_vmask32_xor(mag_vmask32_t x, mag_vmask32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return veorq_u32(x, y);
  #else
    return x^y;
  #endif
}
static MAG_AINLINE mag_vmask32_t mag_vmask32_andnot(mag_vmask32_t x, mag_vmask32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vbicq_u32(y, x);
  #else
    return ~x&y;
  #endif
}
static MAG_AINLINE int mag_vmask32_any(mag_vmask32_t x) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vmaxvq_u32(x) != 0;
  #else
    return x != 0;
  #endif
}
static MAG_AINLINE int mag_vmask32_all(mag_vmask32_t x) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vminvq_u32(x) == ~0u;
  #elif defined(__AVX512F__)
    return x == 0xffff;
  #else
    return x == ~0u;
  #endif
}

/* f32 vector */
static MAG_AINLINE mag_vf32_t mag_vf32_zero(void) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vdupq_n_f32(0.f);
  #elif defined(__AVX512F__)
    return _mm512_setzero_ps();
  #else
    return 0.f;
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_splat(float x) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vdupq_n_f32(x);
  #elif defined(__AVX512F__)
    return _mm512_set1_ps(x);
  #else
    return x;
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_broadcast(const float *p) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vdupq_n_f32(*p);
  #elif defined(__AVX512F__)
    return _mm512_set1_ps(*p);
  #else
    return *p;
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_loada(const float *p) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vld1q_f32(p);
  #elif defined(__AVX512F__)
    return _mm512_load_ps(p);
  #else
    return *p;
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_loadu(const float *p) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vld1q_f32(p);
  #elif defined(__AVX512F__)
    return _mm512_loadu_ps(p);
  #else
    return *p;
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_loadu_masked(const float *p, int n) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    mag_alignas(16) float tmp[4] = {0};
    for (int i=0; i < n; ++i) tmp[i] = p[i];
    return vld1q_f32(tmp);
  #elif defined(__AVX512F__)
    return _mm512_maskz_loadu_ps((__mmask16)((1u <n)-1u), p);
  #else
    return n ? *p : 0;
  #endif
}
static MAG_AINLINE void mag_vf32_storea(float *p, mag_vf32_t v) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    vst1q_f32(p, v);
  #elif defined(__AVX512F__)
    _mm512_store_ps(p, v);
  #else
    *p = v;
  #endif
}
static MAG_AINLINE void mag_vf32_storeu(float *p, mag_vf32_t v) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    vst1q_f32(p, v);
  #elif defined(__AVX512F__)
    _mm512_storeu_ps(p, v);
  #else
    *p = v;
  #endif
}
static MAG_AINLINE void mag_vf32_storeu_masked(float *p, mag_vf32_t v, int n) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    mag_alignas(16) float tmp[4];
    vst1q_f32(tmp, v);
    for (int i=0; i < n; ++i) p[i] = tmp[i];
  #elif defined(__AVX512F__)
    _mm512_mask_store_ps(p, (__mmask16)((1u<<n)-1u), v);
  #else
    if (n) *p = v;
  #endif
}
static MAG_AINLINE mag_vmask32_t mag_vf32_cmpeq(mag_vf32_t x, mag_vf32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vceqq_f32(x, y);
  #elif defined(__AVX512F__)
    return _mm512_cmp_ps_mask(x, y, _CMP_EQ_OQ);
  #else
    return x==y ? ~0u : 0u;
  #endif
}
static MAG_AINLINE mag_vmask32_t mag_vf32_cmpne(mag_vf32_t x, mag_vf32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vmvnq_u32(vceqq_f32(x, y));
  #elif defined(__AVX512F__)
    return _mm512_cmp_ps_mask(x, y, _CMP_NEQ_OQ);
  #else
    return x!=y ? ~0u : 0u;
  #endif
}
static MAG_AINLINE mag_vmask32_t mag_vf32_cmplt(mag_vf32_t x, mag_vf32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vcltq_f32(x, y);
  #elif defined(__AVX512F__)
    return _mm512_cmp_ps_mask(x, y, _CMP_LT_OQ);
  #else
    return x<y ? ~0u : 0u;
  #endif
}
static MAG_AINLINE mag_vmask32_t mag_vf32_cmple(mag_vf32_t x, mag_vf32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vcleq_f32(x, y);
  #elif defined(__AVX512F__)
    return _mm512_cmp_ps_mask(x, y, _CMP_LE_OQ);
  #else
    return x<=y ? ~0u : 0u;
  #endif
}
static MAG_AINLINE mag_vmask32_t mag_vf32_cmpgt(mag_vf32_t x, mag_vf32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vcgtq_f32(x, y);
  #elif defined(__AVX512F__)
    return _mm512_cmp_ps_mask(x, y, _CMP_GT_OQ);
  #else
    return x>y ? ~0u : 0u;
  #endif
}
static MAG_AINLINE mag_vmask32_t mag_vf32_cmpge(mag_vf32_t x, mag_vf32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vcgeq_f32(x, y);
  #elif defined(__AVX512F__)
    return _mm512_cmp_ps_mask(x, y, _CMP_GE_OQ);
  #else
    return x>=y ? ~0u : 0u;
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_blend(mag_vmask32_t m, mag_vf32_t t, mag_vf32_t f) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vbslq_f32(m, t, f);
  #elif defined(__AVX512F__)
    return _mm512_mask_blend_ps(m, f, t);
  #else
    uint32_t tb, fb;
    memcpy(&tb, &t, sizeof(tb));
    memcpy(&fb, &f, sizeof(fb));
    uint32_t rb = ((m&tb)|(~m&fb));
    mag_vf32_t r;
    memcpy(&r, &rb, sizeof(r));
    return r;
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_add(mag_vf32_t x, mag_vf32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vaddq_f32(x, y);
  #elif defined(__AVX512F__)
    return _mm512_add_ps(x, y);
  #else
    return x+y;
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_sub(mag_vf32_t x, mag_vf32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vsubq_f32(x, y);
  #elif defined(__AVX512F__)
    return _mm512_sub_ps(x, y);
  #else
    return x-y;
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_mul(mag_vf32_t x, mag_vf32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vmulq_f32(x, y);
  #elif defined(__AVX512F__)
    return _mm512_mul_ps(x, y);
  #else
    return x*y;
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_div(mag_vf32_t x, mag_vf32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vdivq_f32(x, y);
  #elif defined(__AVX512F__)
    return _mm512_div_ps(x, y);
  #else
    return x/y;
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_min(mag_vf32_t x, mag_vf32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vminq_f32(x, y);
  #elif defined(__AVX512F__)
    return _mm512_min_ps(x, y);
  #else
    return x<y ? x : y;
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_max(mag_vf32_t x, mag_vf32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vmaxq_f32(x, y);
  #elif defined(__AVX512F__)
    return _mm512_max_ps(x, y);
  #else
    return x>y ? x : y;
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_fmadd(mag_vf32_t x, mag_vf32_t y, mag_vf32_t z) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vfmaq_f32(z, x, y);
  #elif defined(__AVX512F__)
    return _mm512_fmadd_ps(x, y, z);
  #else
    return x*y + z;
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_fnmadd(mag_vf32_t x, mag_vf32_t y, mag_vf32_t z) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vfmsq_f32(z, x, y);
  #elif defined(__AVX512F__)
    return _mm512_fmsub_ps(x, y, z);
  #else
    return z - x*y;
  #endif
}
static MAG_AINLINE float mag_vf32_reduce_add(mag_vf32_t x) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vaddvq_f32(x);
  #elif defined(__AVX512F__)
    return _mm512_reduce_add_ps(x);
  #else
    return x;
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_abs(mag_vf32_t x) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vabsq_f32(x);
  #elif defined(__AVX512F__)
    return _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(x), _mm512_set1_epi32(0x7fffffff)));
  #else
    return fabsf(x);
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_rcp_approx(mag_vf32_t x) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vrecpeq_f32(x);
  #elif defined(__AVX512F__)
    return _mm512_rcp14_ps(x);
  #else
    return 1.f/x;
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_rcp_refine_step(mag_vf32_t x, mag_vf32_t r) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vmulq_f32(vrecpsq_f32(x, r), r);
  #elif defined(__AVX512F__)
    return _mm512_mul_ps(r, _mm512_sub_ps(_mm512_set1_ps(2.f), _mm512_mul_ps(x, r)));
  #else
    return r*(2.f - x*r); /* Let's do a Newton step here */
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_and_bits(mag_vf32_t x, mag_vf32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(x), vreinterpretq_u32_f32(y)));
  #elif defined(__AVX512F__)
    return _mm512_and_ps(x, y);
  #else
    uint32_t xb, yb;
    memcpy(&xb, &x, sizeof(xb));
    memcpy(&yb, &y, sizeof(yb));
    uint32_t rb = xb&yb;
    mag_vf32_t r;
    memcpy(&r, &rb, sizeof(rb));
    return r;
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_or_bits(mag_vf32_t x, mag_vf32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(x), vreinterpretq_u32_f32(y)));
  #elif defined(__AVX512F__)
    return _mm512_or_ps(x, y);
  #else
    uint32_t xb, yb;
    memcpy(&xb, &x, sizeof(xb));
    memcpy(&yb, &y, sizeof(yb));
    uint32_t rb = xb|yb;
    mag_vf32_t r;
    memcpy(&r, &rb, sizeof(rb));
    return r;
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_xor_bits(mag_vf32_t x, mag_vf32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(x), vreinterpretq_u32_f32(y)));
  #elif defined(__AVX512F__)
    return _mm512_xor_ps(x, y);
  #else
    uint32_t xb, yb;
    memcpy(&xb, &x, sizeof(xb));
    memcpy(&yb, &y, sizeof(yb));
    uint32_t rb = xb^yb;
    mag_vf32_t r;
    memcpy(&r, &rb, sizeof(rb));
    return r;
  #endif
}

static MAG_AINLINE mag_vf32_t mag_vf32_not_bits(mag_vf32_t x) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vreinterpretq_f32_u32(vmvnq_u32(vreinterpretq_u32_f32(x)));
  #elif defined(__AVX512F__)
    return _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(x), _mm512_set1_epi32(-1)));
  #else
    uint32_t xb;
    memcpy(&xb, &x, sizeof(xb));
    xb = ~xb;
    mag_vf32_t r;
    memcpy(&r, &xb, sizeof(xb));
    return r;
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_andnot_bits(mag_vf32_t x, mag_vf32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(y), vreinterpretq_u32_f32(x)));
  #elif defined(__AVX512F__)
    return _mm512_andnot_ps(x, y);
  #else
    uint32_t xb, yb;
    memcpy(&xb, &x, sizeof(xb));
    memcpy(&yb, &y, sizeof(yb));
    uint32_t rb = ~xb&yb;
    mag_vf32_t r;
    memcpy(&r, &rb, sizeof(rb));
    return r;
  #endif
}

static MAG_AINLINE mag_vf32_t mag_vf32_loadu_f16(const mag_float16_t *p) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vcvt_f32_f16(vld1_f16((const __fp16 *)p));
  #elif defined(__AVX512F__) && defined(__F16C__)
    return _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)p));
  #elif defined(__AVX512F__)
    mag_alignas(64) float tmp[16];
    for (int i=0; i < 16; ++i)
      tmp[i] = mag_float16_to_float32(p[i]);
    return _mm512_load_ps(tmp);
  #else
    return mag_float16_to_float32(*p);
  #endif
}

static MAG_AINLINE void mag_vf32_storeu_f16(mag_float16_t *p, mag_vf32_t v) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    vst1_f16((__fp16 *)p, vcvt_f16_f32(v));
  #elif defined(__AVX512F__) && defined(__F16C__)
    __m256i h = _mm512_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    _mm256_storeu_si256((__m256i *)p, h);
  #elif defined(__AVX512F__)
    mag_alignas(64) float tmp[16];
    _mm512_store_ps(tmp, v);
    for (int i=0; i < 16; ++i)
      p[i] = mag_float32_to_float16(tmp[i]);
  #else
    *p = mag_float32_to_float16(v);
  #endif
}

static MAG_AINLINE mag_vf32_t mag_vf32_loadu_bf16(const mag_bfloat16_t *p) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    uint16x4_t h = vld1_u16((const uint16_t *)p);
    uint32x4_t u = vshll_n_u16(h, 16);
    return vreinterpretq_f32_u32(u);
  #elif defined(__AVX512F__)
    __m256i h = _mm256_loadu_si256((const __m256i *)p);
    __m512i u = _mm512_cvtepu16_epi32(h);
    u = _mm512_slli_epi32(u, 16);
    return _mm512_castsi512_ps(u);
  #else
    return mag_bfloat16_to_float32(*p);
  #endif
}

static MAG_AINLINE void mag_vf32_storeu_bf16(mag_bfloat16_t *p, mag_vf32_t v) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    uint32x4_t u = vreinterpretq_u32_f32(v);
    uint16x4_t h = vmovn_u32(vshrq_n_u32(u, 16));
    vst1_u16((uint16_t *)p, h);
  #elif defined(__AVX512F__) && defined(__AVX512BF16__)
    __m256bh h = _mm512_cvtneps_pbh(v);
    _mm256_storeu_si256((__m256i *)p, (__m256i)h);
  #elif defined(__AVX512F__)
    __m512i u = _mm512_castps_si512(v);
    u = _mm512_srli_epi32(u, 16);
    __m256i h = _mm512_cvtepi32_epi16(u);
    _mm256_storeu_si256((__m256i *)p, h);
  #else
    *p = mag_float32_to_bfloat16(v);
  #endif
}

/* i32 vector */
static MAG_AINLINE mag_vi32_t mag_vi32_zero(void) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vdupq_n_s32(0);
  #elif defined(__AVX512F__)
    return _mm512_setzero_si512();
  #else
    return 0;
  #endif
}
static MAG_AINLINE mag_vi32_t mag_vi32_splat(int32_t x) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vdupq_n_s32(x);
  #elif defined(__AVX512F__)
    return _mm512_set1_epi32(x);
  #else
    return x;
  #endif
}
static MAG_AINLINE mag_vi32_t mag_vi32_broadcast(const int32_t *p) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vdupq_n_s32(*p);
  #elif defined(__AVX512F__)
    return _mm512_set1_epi32(*p);
  #else
    return *p;
  #endif
}
static MAG_AINLINE mag_vi32_t mag_vi32_loada(const int32_t *p) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vld1q_s32(p);
  #elif defined(__AVX512F__)
    return _mm512_load_epi32(p);
  #else
    return *p;
  #endif
}
static MAG_AINLINE mag_vi32_t mag_vi32_loadu(const int32_t *p) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vld1q_s32(p);
  #elif defined(__AVX512F__)
    return _mm512_loadu_epi32(p);
  #else
    return *p;
  #endif
}
static MAG_AINLINE mag_vi32_t mag_vi32_loadu_masked(const int32_t *p, int n) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    mag_alignas(16) int32_t tmp[4] = {0};
    for (int i=0; i < n; ++i) tmp[i] = p[i];
    return vld1q_s32(tmp);
  #elif defined(__AVX512F__)
    return _mm512_maskz_loadu_epi32((__mmask16)((1u <n)-1u), p);
  #else
    return n ? *p : 0;
  #endif
}
static MAG_AINLINE void mag_vi32_storea(int32_t *p, mag_vi32_t v) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    vst1q_s32(p, v);
  #elif defined(__AVX512F__)
    _mm512_store_epi32(p, v);
  #else
    *p = v;
  #endif
}
static MAG_AINLINE void mag_vi32_storeu(int32_t *p, mag_vi32_t v) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    vst1q_s32(p, v);
  #elif defined(__AVX512F__)
    _mm512_storeu_epi32(p, v);
  #else
    *p = v;
  #endif
}
static MAG_AINLINE void mag_vi32_storeu_masked(int32_t *p, mag_vi32_t v, int n) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    mag_alignas(16) int32_t tmp[4];
    vst1q_s32(tmp, v);
    for (int i=0; i < n; ++i) p[i] = tmp[i];
  #elif defined(__AVX512F__)
    _mm512_mask_store_epi32(p, (__mmask16)((1u<<n)-1u), v);
  #else
    if (n) *p = v;
  #endif
}
static MAG_AINLINE mag_vmask32_t mag_vi32_cmpeq(mag_vi32_t x, mag_vi32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vceqq_s32(x, y);
  #elif defined(__AVX512F__)
      return _mm512_cmp_epi32_mask(x, y, _MM_CMPINT_EQ);
  #else
    return x==y ? ~0u : 0u;
  #endif
}
static MAG_AINLINE mag_vmask32_t mag_vi32_cmpne(mag_vi32_t x, mag_vi32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vmvnq_u32(vceqq_s32(x, y));
  #elif defined(__AVX512F__)
    return _mm512_cmp_epi32_mask(x, y, _MM_CMPINT_NE);
  #else
    return x!=y ? ~0u : 0u;
  #endif
}
static MAG_AINLINE mag_vmask32_t mag_vi32_cmplt(mag_vi32_t x, mag_vi32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vcltq_s32(x, y);
  #elif defined(__AVX512F__)
    return _mm512_cmp_epi32_mask(x, y, _MM_CMPINT_LT);
  #else
    return x<y ? ~0u : 0u;
  #endif
}
static MAG_AINLINE mag_vmask32_t mag_vi32_cmple(mag_vi32_t x, mag_vi32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vcleq_s32(x, y);
  #elif defined(__AVX512F__)
    return _mm512_cmp_epi32_mask(x, y, _MM_CMPINT_LE);
  #else
    return x<=y ? ~0u : 0u;
  #endif
}
static MAG_AINLINE mag_vmask32_t mag_vi32_cmpgt(mag_vi32_t x, mag_vi32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vcgtq_s32(x, y);
  #elif defined(__AVX512F__)
    return _mm512_cmp_epi32_mask(x, y, _MM_CMPINT_GT);
  #else
    return x>y ? ~0u : 0u;
  #endif
}
static MAG_AINLINE mag_vmask32_t mag_vi32_cmpge(mag_vi32_t x, mag_vi32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vcgeq_s32(x, y);
  #elif defined(__AVX512F__)
    return _mm512_cmp_epi32_mask(x, y, _MM_CMPINT_GE);
  #else
    return x>=y ? ~0u : 0u;
  #endif
}
static MAG_AINLINE mag_vi32_t mag_vi32_blend(mag_vmask32_t m, mag_vi32_t t, mag_vi32_t f) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vbslq_s32(m, t, f);
  #elif defined(__AVX512F__)
    return _mm512_mask_blend_epi32(m, f, t);
  #else
    return (m&t)|(~m&f);
  #endif
}
static MAG_AINLINE mag_vi32_t mag_vi32_add(mag_vi32_t x, mag_vi32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vaddq_s32(x, y);
  #elif defined(__AVX512F__)
    return _mm512_add_epi32(x, y);
  #else
    return x+y;
  #endif
}
static MAG_AINLINE mag_vi32_t mag_vi32_sub(mag_vi32_t x, mag_vi32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vsubq_s32(x, y);
  #elif defined(__AVX512F__)
    return _mm512_sub_epi32(x, y);
  #else
    return x-y;
  #endif
}
static MAG_AINLINE mag_vi32_t mag_vi32_mul(mag_vi32_t x, mag_vi32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vmulq_s32(x, y);
  #elif defined(__AVX512F__)
    return _mm512_mul_epi32(x, y);
  #else
    return x*y;
  #endif
}
static MAG_AINLINE mag_vi32_t mag_vi32_and(mag_vi32_t x, mag_vi32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vreinterpretq_s32_u32(vandq_u32(vreinterpretq_u32_s32(x), vreinterpretq_u32_s32(y)));
  #elif defined(__AVX512F__)
    return _mm512_and_epi32(x, y);
  #else
    return x&y;
  #endif
}
static MAG_AINLINE mag_vi32_t mag_vi32_or(mag_vi32_t x, mag_vi32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vreinterpretq_s32_u32(vorrq_u32(vreinterpretq_u32_s32(x), vreinterpretq_u32_s32(y)));
  #elif defined(__AVX512F__)
    return _mm512_or_epi32(x, y);
  #else
    return x|y;
  #endif
}
static MAG_AINLINE mag_vi32_t mag_vi32_xor(mag_vi32_t x, mag_vi32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vreinterpretq_s32_u32(veorq_u32(vreinterpretq_u32_s32(x), vreinterpretq_u32_s32(y)));
  #elif defined(__AVX512F__)
    return _mm512_xor_epi32(x, y);
  #else
    return x^y;
  #endif
}
static MAG_AINLINE mag_vi32_t mag_vi32_andnot(mag_vi32_t x, mag_vi32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vreinterpretq_s32_u32(vbicq_u32(vreinterpretq_u32_s32(y), vreinterpretq_u32_s32(x)));
  #elif defined(__AVX512F__)
    return _mm512_andnot_epi32(x, y);
  #else
    return ~x&y;
  #endif
}
static MAG_AINLINE mag_vi32_t mag_vi32_not(mag_vi32_t x) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vreinterpretq_s32_u32(vmvnq_u32(vreinterpretq_u32_s32(x)));
  #elif defined(__AVX512F__)
    return _mm512_xor_si512(x, _mm512_set1_epi32(-1));
  #else
    return ~x;
  #endif
}
static MAG_AINLINE mag_vi32_t mag_vi32_slli(mag_vi32_t x, int n) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vreinterpretq_s32_u32(vshlq_u32(vreinterpretq_u32_s32(x), vdupq_n_s32(n)));
  #elif defined(__AVX512F__)
    return _mm512_slli_epi32(x, n);
  #else
    return x<<n;
  #endif
}
static MAG_AINLINE mag_vi32_t mag_vi32_srli(mag_vi32_t x, int n) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vreinterpretq_s32_u32(vshlq_u32(vreinterpretq_u32_s32(x), vdupq_n_s32(-n)));
  #elif defined(__AVX512F__)
    return _mm512_srli_epi32(x, n);
  #else
    return (mag_vi32_t)((uint32_t)x >> n);
  #endif
}
static MAG_AINLINE mag_vi32_t mag_vi32_srai(mag_vi32_t x, int n) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vshlq_s32(x, vdupq_n_s32(-n));
  #elif defined(__AVX512F__)
    return _mm512_srai_epi32(x, n);
  #else
    return x>>n;
  #endif
}
static MAG_AINLINE mag_vi32_t mag_vi32_min(mag_vi32_t x, mag_vi32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vminq_s32(x, y);
  #elif defined(__AVX512F__)
    return _mm512_min_epi32(x, y);
  #else
    return x<y ? x : y;
  #endif
}
static MAG_AINLINE mag_vi32_t mag_vi32_max(mag_vi32_t x, mag_vi32_t y) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vmaxq_s32(x, y);
  #elif defined(__AVX512F__)
    return _mm512_max_epi32(x, y);
  #else
    return x>y ? x : y;
  #endif
}
static MAG_AINLINE int32_t mag_vi32_reduce_add(mag_vi32_t x) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vaddvq_s32(x);
  #elif defined(__AVX512F__)
    return _mm512_reduce_add_epi32(x);
  #else
    return x;
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vf32_reinterpret_from_vi32(mag_vi32_t x) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vreinterpretq_f32_s32(x);
  #elif defined(__AVX512F__)
    return _mm512_castsi512_ps(x);
  #else
    mag_vf32_t r;
    memcpy(&r, &x, sizeof(r));
    return r;
  #endif
}
static MAG_AINLINE mag_vi32_t mag_vi32_reinterpret_from_vf32(mag_vf32_t x) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vreinterpretq_s32_f32(x);
  #elif defined(__AVX512F__)
    return _mm512_castps_si512(x);
  #else
    mag_vi32_t r;
    memcpy(&r, &x, sizeof(r));
    return r;
  #endif
}
static MAG_AINLINE mag_vf32_t mag_vi32_to_f32(mag_vi32_t x) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vcvtq_f32_s32(x);
  #elif defined(__AVX512F__)
    return _mm512_cvtepi32_ps(x);
  #else
    return (mag_vf32_t)x;
  #endif
}
static MAG_AINLINE mag_vi32_t mag_vf32_trunc_to_vi32(mag_vf32_t x) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vcvtq_s32_f32(x);
  #elif defined(__AVX512F__)
    return _mm512_cvtps_epi32(x);
  #else
    return (mag_vi32_t)x;
  #endif
}

static MAG_AINLINE mag_vi32_t mag_vi32_from_vmask32_bits(mag_vmask32_t m) {
  #if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
    return vreinterpretq_s32_u32(m);
  #elif defined(__AVX512F__)
    return _mm512_maskz_set1_epi32(m, -1);
  #else
    return (mag_vi32_t)m;
  #endif
}

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