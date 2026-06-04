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

static MAG_AINLINE void mag_vphilox4x32_round(
  mag_vi32_t *c0,
  mag_vi32_t *c1,
  mag_vi32_t *c2,
  mag_vi32_t *c3,
  mag_vi32_t *k0,
  mag_vi32_t *k1
) {
  mag_vi32_t m0 = mag_vi32_splat(MAG_PHILOX_M0);
  mag_vi32_t m1 = mag_vi32_splat(MAG_PHILOX_M1);
  mag_vi32_t lo0 = mag_vi32_mul(*c0, m0);
  mag_vi32_t hi0 = mag_vi32_mulhi_u32(*c0, m0);
  mag_vi32_t lo1 = mag_vi32_mul(*c2, m1);
  mag_vi32_t hi1 = mag_vi32_mulhi_u32(*c2, m1);
  mag_vi32_t nc0 = mag_vi32_xor(mag_vi32_xor(hi1, *c1), *k0);
  mag_vi32_t nc1 = lo1;
  mag_vi32_t nc2 = mag_vi32_xor(mag_vi32_xor(hi0, *c3), *k1);
  mag_vi32_t nc3 = lo0;
  *c0 = nc0;
  *c1 = nc1;
  *c2 = nc2;
  *c3 = nc3;
  *k0 = mag_vi32_add(*k0, mag_vi32_splat(MAG_PHILOX_W0));
  *k1 = mag_vi32_add(*k1, mag_vi32_splat(MAG_PHILOX_W1));
}

static MAG_AINLINE void mag_vphilox_4x32_sample(
  uint64_t counter,
  uint64_t subseq,
  uint64_t seed,
  mag_vi32_t *o0,
  mag_vi32_t *o1,
  mag_vi32_t *o2,
  mag_vi32_t *o3
) {
  mag_vi32_t c0 = mag_vi32_add(mag_vi32_splat((uint32_t)counter),  mag_vi32_iota());
  mag_vi32_t c1 = mag_vi32_splat((uint32_t)(counter>>32));
  mag_vi32_t c2 = mag_vi32_splat((uint32_t)subseq);
  mag_vi32_t c3 = mag_vi32_splat((uint32_t)(subseq>>32));
  mag_vi32_t k0 = mag_vi32_splat((uint32_t)seed);
  mag_vi32_t k1 = mag_vi32_splat((uint32_t)(seed>>32));
  for (int i=0; i < MAG_PHILOX_ROUNDS; ++i)
    mag_vphilox4x32_round(&c0, &c1, &c2, &c3, &k0, &k1);
  *o0 = c0;
  *o1 = c1;
  *o2 = c2;
  *o3 = c3;
}

static MAG_AINLINE mag_vf32_t mag_vphilox_u32_to_f32(mag_vi32_t x) {
  x = mag_vi32_srli(x, 9);
  return mag_vf32_mul(
    mag_vf32_add(mag_vi32_to_f32(x), mag_vf32_splat(0.5f)),
    mag_vf32_splat(1.f/0x1.0p23f)
  );
}

static MAG_AINLINE void mag_vphilox_4x32_uniform_f32(
  uint64_t counter,
  uint64_t subseq,
  uint64_t seed,
  mag_vf32_t *u0,
  mag_vf32_t *u1,
  mag_vf32_t *u2,
  mag_vf32_t *u3
) {
  mag_vi32_t r0, r1, r2, r3;
  mag_vphilox_4x32_sample(counter, subseq, seed, &r0, &r1, &r2, &r3);
  *u0 = mag_vphilox_u32_to_f32(r0);
  *u1 = mag_vphilox_u32_to_f32(r1);
  *u2 = mag_vphilox_u32_to_f32(r2);
  *u3 = mag_vphilox_u32_to_f32(r3);
}

static MAG_AINLINE void mag_vphilox_4x32_normal_f32(
  uint64_t counter,
  uint64_t subseq,
  uint64_t seed,
  float mean,
  float std,
  mag_vf32_t *z0,
  mag_vf32_t *z1,
  mag_vf32_t *z2,
  mag_vf32_t *z3
) {
  mag_vf32_t u0, u1, u2, u3;
  mag_vphilox_4x32_uniform_f32(counter, subseq, seed, &u0, &u1, &u2, &u3);
  u0 = mag_vf32_max(u0, mag_vf32_splat(1e-37f));
  u2 = mag_vf32_max(u2, mag_vf32_splat(1e-37f));
  mag_vf32_t rho0 = mag_vf32_sqrt(mag_vf32_mul(mag_vf32_splat(-2.0f), mag_vf32_log(u0)));
  mag_vf32_t rho1 = mag_vf32_sqrt(mag_vf32_mul(mag_vf32_splat(-2.0f), mag_vf32_log(u2)));
  mag_vf32_t tau = mag_vf32_splat(MAG_TAU);
  mag_vf32_t theta0 = mag_vf32_mul(tau, u1);
  mag_vf32_t theta1 = mag_vf32_mul(tau, u3);
  mag_vf32_t s0, c0, s1, c1;
  mag_vf32_sincos(theta0, &s0, &c0);
  mag_vf32_sincos(theta1, &s1, &c1);
  mag_vf32_t vmean = mag_vf32_splat(mean);
  mag_vf32_t vstd  = mag_vf32_splat(std);
  *z0 = mag_vf32_fmadd(mag_vf32_mul(rho0, c0), vstd, vmean);
  *z1 = mag_vf32_fmadd(mag_vf32_mul(rho0, s0), vstd, vmean);
  *z2 = mag_vf32_fmadd(mag_vf32_mul(rho1, c1), vstd, vmean);
  *z3 = mag_vf32_fmadd(mag_vf32_mul(rho1, s1), vstd, vmean);
}

#define mag_gen_vrand_uniform_fp_simd(T, STOREU, STOREM)                         \
  static void MAG_AINLINE mag_vrand_uniform_##T##_simd(                          \
    uint64_t seed, uint64_t subseq, uint64_t counter,                            \
    int64_t numel, T *restrict o, float min, float max                           \
  ) {                                                                            \
    int64_t i = 0;                                                               \
    mag_vf32_t vmin = mag_vf32_splat(min);                                       \
    mag_vf32_t vscale = mag_vf32_splat(max - min);                               \
                                                                                 \
    for (; i + 4 * MAG_VF32_LANES <= numel; i += 4 * MAG_VF32_LANES) {           \
      mag_vf32_t u0, u1, u2, u3;                                                 \
      mag_vphilox_4x32_uniform_f32(counter, subseq, seed, &u0, &u1, &u2, &u3);    \
      STOREU(o + i + 0 * MAG_VF32_LANES, mag_vf32_fmadd(u0, vscale, vmin));      \
      STOREU(o + i + 1 * MAG_VF32_LANES, mag_vf32_fmadd(u1, vscale, vmin));      \
      STOREU(o + i + 2 * MAG_VF32_LANES, mag_vf32_fmadd(u2, vscale, vmin));      \
      STOREU(o + i + 3 * MAG_VF32_LANES, mag_vf32_fmadd(u3, vscale, vmin));      \
      counter += MAG_VF32_LANES;                                                 \
    }                                                                            \
                                                                                 \
    int64_t rem = numel - i;                                                     \
    if (rem <= 0) return;                                                        \
                                                                                 \
    mag_vf32_t u0, u1, u2, u3;                                                   \
    mag_vphilox_4x32_uniform_f32(counter, subseq, seed, &u0, &u1, &u2, &u3);      \
    u0 = mag_vf32_fmadd(u0, vscale, vmin);                                       \
    u1 = mag_vf32_fmadd(u1, vscale, vmin);                                       \
    u2 = mag_vf32_fmadd(u2, vscale, vmin);                                       \
    u3 = mag_vf32_fmadd(u3, vscale, vmin);                                       \
                                                                                 \
    if (rem > 0) { int k = rem < MAG_VF32_LANES ? (int)rem : MAG_VF32_LANES; STOREM(o+i, u0, k); i += k; rem -= k; } \
    if (rem > 0) { int k = rem < MAG_VF32_LANES ? (int)rem : MAG_VF32_LANES; STOREM(o+i, u1, k); i += k; rem -= k; } \
    if (rem > 0) { int k = rem < MAG_VF32_LANES ? (int)rem : MAG_VF32_LANES; STOREM(o+i, u2, k); i += k; rem -= k; } \
    if (rem > 0) { int k = rem < MAG_VF32_LANES ? (int)rem : MAG_VF32_LANES; STOREM(o+i, u3, k); }                  \
  }

#define mag_gen_vrand_normal_fp_simd(T, STOREU, STOREM)                          \
  static void MAG_AINLINE mag_vrand_normal_##T##_simd(                           \
    uint64_t seed, uint64_t subseq, uint64_t counter,                            \
    int64_t numel, T *restrict o, float mean, float std                          \
  ) {                                                                            \
    int64_t i = 0;                                                               \
                                                                                 \
    for (; i + 4 * MAG_VF32_LANES <= numel; i += 4 * MAG_VF32_LANES) {           \
      mag_vf32_t z0, z1, z2, z3;                                                 \
      mag_vphilox_4x32_normal_f32(counter, subseq, seed, mean, std, &z0, &z1, &z2, &z3); \
      STOREU(o + i + 0 * MAG_VF32_LANES, z0);                                    \
      STOREU(o + i + 1 * MAG_VF32_LANES, z1);                                    \
      STOREU(o + i + 2 * MAG_VF32_LANES, z2);                                    \
      STOREU(o + i + 3 * MAG_VF32_LANES, z3);                                    \
      counter += MAG_VF32_LANES;                                                 \
    }                                                                            \
                                                                                 \
    int64_t rem = numel - i;                                                     \
    if (rem <= 0) return;                                                        \
                                                                                 \
    mag_vf32_t z0, z1, z2, z3;                                                   \
    mag_vphilox_4x32_normal_f32(counter, subseq, seed, mean, std, &z0, &z1, &z2, &z3); \
                                                                                 \
    if (rem > 0) { int k = rem < MAG_VF32_LANES ? (int)rem : MAG_VF32_LANES; STOREM(o+i, z0, k); i += k; rem -= k; } \
    if (rem > 0) { int k = rem < MAG_VF32_LANES ? (int)rem : MAG_VF32_LANES; STOREM(o+i, z1, k); i += k; rem -= k; } \
    if (rem > 0) { int k = rem < MAG_VF32_LANES ? (int)rem : MAG_VF32_LANES; STOREM(o+i, z2, k); i += k; rem -= k; } \
    if (rem > 0) { int k = rem < MAG_VF32_LANES ? (int)rem : MAG_VF32_LANES; STOREM(o+i, z3, k); }                  \
  }

mag_gen_vrand_uniform_fp_simd(float, mag_vf32_storeu, mag_vf32_storeu_masked)
mag_gen_vrand_uniform_fp_simd(mag_float16_t, mag_vf32_storeu_f16, mag_vf32_storeu_masked_f16)
mag_gen_vrand_uniform_fp_simd(mag_bfloat16_t, mag_vf32_storeu_bf16, mag_vf32_storeu_masked_bf16)
mag_gen_vrand_uniform_fp_simd(mag_float8_e4m3fn_t, mag_vf32_storeu_float8_e4m3fn, mag_vf32_storeu_masked_float8_e4m3fn)

mag_gen_vrand_normal_fp_simd(float, mag_vf32_storeu, mag_vf32_storeu_masked)
mag_gen_vrand_normal_fp_simd(mag_float16_t, mag_vf32_storeu_f16, mag_vf32_storeu_masked_f16)
mag_gen_vrand_normal_fp_simd(mag_bfloat16_t, mag_vf32_storeu_bf16, mag_vf32_storeu_masked_bf16)
mag_gen_vrand_normal_fp_simd(mag_float8_e4m3fn_t, mag_vf32_storeu_float8_e4m3fn, mag_vf32_storeu_masked_float8_e4m3fn)

#undef mag_gen_vrand_normal_fp_simd
#undef mag_gen_vrand_uniform_fp_simd
