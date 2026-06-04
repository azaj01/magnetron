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

#define MAG_PF_GROUP        8
#define MAG_PFDIST_B_L1     16
#define MAG_PFDIST_B_L2     96
#define MAG_PFDIST_A_L1     16
#define MAG_PFDIST_A_L2     80

#ifndef MAG_VF32_LANES
#define MAG_VF32_LANES ((int64_t)(sizeof(mag_vf32_t)/sizeof(float)))
#endif
#define MAG_L ((int)MAG_VF32_LANES)

static MAG_AINLINE mag_vf32_t mag_load_bf16_partial(const mag_bfloat16_t *p, int n) {
  if (n == MAG_L) return mag_vf32_loadu_bf16(p);
  mag_alignas(64) float tmp[MAG_L];
  for (int i=0; i < MAG_L; ++i) tmp[i] = 0.f;
  for (int i=0; i < n; ++i) tmp[i] = mag_bfloat16_to_float32(p[i]);
  return mag_vf32_loadu(tmp);
}
static MAG_AINLINE void mag_store_bf16_partial(mag_bfloat16_t *p, mag_vf32_t v, int n) {
  if (n == MAG_L) { mag_vf32_storeu_bf16(p, v); return; }
  mag_alignas(64) float tmp[MAG_L];
  mag_vf32_storeu(tmp, v);
  for (int i=0; i < n; ++i) p[i] = mag_float32_to_bfloat16(tmp[i]);
}

static MAG_AINLINE void mag_mm_tile_8x8_float32(int64_t kc, const float *restrict a, ptrdiff_t lda, const float *restrict b, ptrdiff_t ldb, float *restrict c, ptrdiff_t ldc, bool acc) {
  (void)lda;
  enum { ROWS = 8, COLS = 8, L = MAG_L, NV = (COLS + L - 1) / L, TAIL = COLS - (NV - 1) * L };
  mag_vf32_t C[ROWS][NV];
  for (int r=0; r < ROWS; ++r) {
    for (int v=0; v < NV; ++v) {
      if (!acc) C[r][v] = mag_vf32_zero();
      else if (v == NV - 1 && TAIL != L) C[r][v] = mag_vf32_loadu_masked(c + r*ldc + v*L, TAIL);
      else C[r][v] = mag_vf32_loadu(c + r*ldc + v*L);
    }
  }
  for (int64_t k=0; k < kc; ++k) {
    if ((k & (MAG_PF_GROUP - 1)) == 0) {
      mag_simd_prefetch_t0(b + (k + MAG_PFDIST_B_L1)*ldb);
      mag_simd_prefetch_t1(b + (k + MAG_PFDIST_B_L2)*ldb);
      mag_simd_prefetch_t0(a + (k + MAG_PFDIST_A_L1)*8);
      mag_simd_prefetch_t1(a + (k + MAG_PFDIST_A_L2)*8);
    }
    mag_vf32_t Bv[NV];
    for (int v=0; v < NV; ++v) {
      if (v == NV - 1 && TAIL != L) Bv[v] = mag_vf32_loadu_masked(b + k*ldb + v*L, TAIL);
      else Bv[v] = mag_vf32_loadu(b + k*ldb + v*L);
    }
    const float *ak = a + k*8;
    for (int r=0; r < ROWS; ++r) {
      mag_vf32_t Av = mag_vf32_broadcast(ak + r);
      for (int v=0; v < NV; ++v)
        C[r][v] = mag_vf32_fmadd(Av, Bv[v], C[r][v]);
    }
  }
  for (int r=0; r < ROWS; ++r) {
    for (int v=0; v < NV; ++v) {
      if (v == NV - 1 && TAIL != L) mag_vf32_storeu_masked(c + r*ldc + v*L, C[r][v], TAIL);
      else mag_vf32_storeu(c + r*ldc + v*L, C[r][v]);
    }
  }
}

static MAG_AINLINE void mag_mm_tile_8x16_float32(int64_t kc, const float *restrict a, ptrdiff_t lda, const float *restrict b, ptrdiff_t ldb, float *restrict c, ptrdiff_t ldc, bool acc) {
  mag_mm_tile_8x8_float32(kc, a, lda, b,   ldb, c,   ldc, acc);
  mag_mm_tile_8x8_float32(kc, a, lda, b+8, ldb, c+8, ldc, acc);
}

static MAG_AINLINE void mag_mm_tile_8x32_float32(int64_t kc, const float *restrict a, ptrdiff_t lda, const float *restrict b, ptrdiff_t ldb, float *restrict c, ptrdiff_t ldc, bool acc) {
  mag_mm_tile_8x16_float32(kc, a, lda, b,    ldb, c,    ldc, acc);
  mag_mm_tile_8x16_float32(kc, a, lda, b+16, ldb, c+16, ldc, acc);
}

static MAG_AINLINE void mag_mm_tile_1x8_float32(int64_t kc, const float *restrict a, const float *restrict b, ptrdiff_t ldb, float *restrict c, bool acc) {
  enum { COLS = 8, L = MAG_L, NV = (COLS + L - 1) / L, TAIL = COLS - (NV - 1) * L };
  mag_vf32_t C[NV];
  for (int v=0; v < NV; ++v) {
    if (!acc) C[v] = mag_vf32_zero();
    else if (v == NV - 1 && TAIL != L) C[v] = mag_vf32_loadu_masked(c + v*L, TAIL);
    else C[v] = mag_vf32_loadu(c + v*L);
  }
  for (int64_t k=0; k < kc; ++k) {
    if ((k & (MAG_PF_GROUP - 1)) == 0) {
      mag_simd_prefetch_t0(b + (k + MAG_PFDIST_B_L1)*ldb);
      mag_simd_prefetch_t1(b + (k + MAG_PFDIST_B_L2)*ldb);
      mag_simd_prefetch_t0(a + (k + MAG_PFDIST_A_L1));
      mag_simd_prefetch_t1(a + (k + MAG_PFDIST_A_L2));
    }
    mag_vf32_t Av = mag_vf32_broadcast(a + k);
    for (int v=0; v < NV; ++v) {
      mag_vf32_t Bv = (v == NV - 1 && TAIL != L)
        ? mag_vf32_loadu_masked(b + k*ldb + v*L, TAIL)
        : mag_vf32_loadu(b + k*ldb + v*L);
      C[v] = mag_vf32_fmadd(Av, Bv, C[v]);
    }
  }
  for (int v=0; v < NV; ++v) {
    if (v == NV - 1 && TAIL != L) mag_vf32_storeu_masked(c + v*L, C[v], TAIL);
    else mag_vf32_storeu(c + v*L, C[v]);
  }
}

static MAG_AINLINE void mag_mm_tile_1x16_float32(int64_t kc, const float *restrict a, const float *restrict b, ptrdiff_t ldb, float *restrict c, bool acc) {
  enum { COLS = 16, L = MAG_L, NV = (COLS + L - 1) / L, TAIL = COLS - (NV - 1) * L };
  mag_vf32_t C[NV];
  for (int v=0; v < NV; ++v) {
    if (!acc) C[v] = mag_vf32_zero();
    else if (v == NV - 1 && TAIL != L) C[v] = mag_vf32_loadu_masked(c + v*L, TAIL);
    else C[v] = mag_vf32_loadu(c + v*L);
  }
  for (int64_t k=0; k < kc; ++k) {
    if ((k & (MAG_PF_GROUP - 1)) == 0) {
      mag_simd_prefetch_t0(b + (k + MAG_PFDIST_B_L1)*ldb);
      mag_simd_prefetch_t1(b + (k + MAG_PFDIST_B_L2)*ldb);
      mag_simd_prefetch_t0(a + (k + MAG_PFDIST_A_L1));
      mag_simd_prefetch_t1(a + (k + MAG_PFDIST_A_L2));
    }
    mag_vf32_t Av = mag_vf32_broadcast(a + k);
    for (int v=0; v < NV; ++v) {
      mag_vf32_t Bv = (v == NV - 1 && TAIL != L)
        ? mag_vf32_loadu_masked(b + k*ldb + v*L, TAIL)
        : mag_vf32_loadu(b + k*ldb + v*L);
      C[v] = mag_vf32_fmadd(Av, Bv, C[v]);
    }
  }
  for (int v=0; v < NV; ++v) {
    if (v == NV - 1 && TAIL != L) mag_vf32_storeu_masked(c + v*L, C[v], TAIL);
    else mag_vf32_storeu(c + v*L, C[v]);
  }
}

static MAG_AINLINE void mag_mm_tile_1x32_float32(int64_t kc, const float *restrict a, const float *restrict b, ptrdiff_t ldb, float *restrict c, bool acc) {
  mag_mm_tile_1x16_float32(kc, a, b,    ldb, c,    acc);
  mag_mm_tile_1x16_float32(kc, a, b+16, ldb, c+16, acc);
}

static MAG_AINLINE void mag_mm_tile_16x16_float32(int64_t kc, const float *restrict a, ptrdiff_t lda, const float *restrict b, ptrdiff_t ldb, float *restrict c, ptrdiff_t ldc, bool acc) {
  mag_mm_tile_8x16_float32(kc, a,         lda, b, ldb, c,         ldc, acc);
  mag_mm_tile_8x16_float32(kc, a + 8*lda, lda, b, ldb, c + 8*ldc, ldc, acc);
}

static MAG_AINLINE void mag_mm_tile_16x32_float32(int64_t kc, const float *restrict a, ptrdiff_t lda, const float *restrict b, ptrdiff_t ldb, float *restrict c, ptrdiff_t ldc, bool acc) {
  mag_mm_tile_16x16_float32(kc, a, lda, b,    ldb, c,    ldc, acc);
  mag_mm_tile_16x16_float32(kc, a, lda, b+16, ldb, c+16, ldc, acc);
}

static MAG_AINLINE void mag_mm_tile_8x8_bfloat16(
  int64_t kc,
  const mag_bfloat16_t *restrict a, ptrdiff_t lda,
  const mag_bfloat16_t *restrict b, ptrdiff_t ldb,
  mag_bfloat16_t *restrict c, ptrdiff_t ldc,
  bool acc
) {
  (void)lda;
  enum { ROWS = 8, COLS = 8, L = MAG_L, NV = (COLS + L - 1) / L, TAIL = COLS - (NV - 1) * L };
  mag_vf32_t C[ROWS][NV];
  for (int r=0; r < ROWS; ++r) {
    for (int v=0; v < NV; ++v) {
      if (!acc) C[r][v] = mag_vf32_zero();
      else if (v == NV - 1 && TAIL != L) C[r][v] = mag_load_bf16_partial(c + r*ldc + v*L, TAIL);
      else C[r][v] = mag_vf32_loadu_bf16(c + r*ldc + v*L);
    }
  }
  for (int64_t k=0; k < kc; ++k) {
    if ((k & (MAG_PF_GROUP - 1)) == 0) {
      mag_simd_prefetch_t0(b + (k + MAG_PFDIST_B_L1)*ldb);
      mag_simd_prefetch_t1(b + (k + MAG_PFDIST_B_L2)*ldb);
      mag_simd_prefetch_t0(a + (k + MAG_PFDIST_A_L1)*8);
      mag_simd_prefetch_t1(a + (k + MAG_PFDIST_A_L2)*8);
    }
    mag_vf32_t Bv[NV];
    for (int v=0; v < NV; ++v) {
      if (v == NV - 1 && TAIL != L) Bv[v] = mag_load_bf16_partial(b + k*ldb + v*L, TAIL);
      else Bv[v] = mag_vf32_loadu_bf16(b + k*ldb + v*L);
    }
    const mag_bfloat16_t *ak = a + k*8;
    for (int r=0; r < ROWS; ++r) {
      mag_vf32_t Av = mag_vf32_splat(mag_bfloat16_to_float32(ak[r]));
      for (int v=0; v < NV; ++v)
        C[r][v] = mag_vf32_fmadd(Av, Bv[v], C[r][v]);
    }
  }
  for (int r=0; r < ROWS; ++r) {
    for (int v=0; v < NV; ++v) {
      if (v == NV - 1 && TAIL != L) mag_store_bf16_partial(c + r*ldc + v*L, C[r][v], TAIL);
      else mag_vf32_storeu_bf16(c + r*ldc + v*L, C[r][v]);
    }
  }
}

static MAG_AINLINE void mag_mm_tile_8x16_bfloat16(
  int64_t kc,
  const mag_bfloat16_t *restrict a, ptrdiff_t lda,
  const mag_bfloat16_t *restrict b, ptrdiff_t ldb,
  mag_bfloat16_t *restrict c, ptrdiff_t ldc,
  bool acc
) {
  mag_mm_tile_8x8_bfloat16(kc, a, lda, b,   ldb, c,   ldc, acc);
  mag_mm_tile_8x8_bfloat16(kc, a, lda, b+8, ldb, c+8, ldc, acc);
}

static MAG_AINLINE void mag_mm_tile_8x32_bfloat16(
  int64_t kc,
  const mag_bfloat16_t *restrict a, ptrdiff_t lda,
  const mag_bfloat16_t *restrict b, ptrdiff_t ldb,
  mag_bfloat16_t *restrict c, ptrdiff_t ldc,
  bool acc
) {
  mag_mm_tile_8x16_bfloat16(kc, a, lda, b,    ldb, c,    ldc, acc);
  mag_mm_tile_8x16_bfloat16(kc, a, lda, b+16, ldb, c+16, ldc, acc);
}

static MAG_AINLINE void mag_mm_tile_1x8_bfloat16(
  int64_t kc,
  const mag_bfloat16_t *restrict a,
  const mag_bfloat16_t *restrict b, ptrdiff_t ldb,
  mag_bfloat16_t *restrict c,
  bool acc
) {
  enum { COLS = 8, L = MAG_L, NV = (COLS + L - 1) / L, TAIL = COLS - (NV - 1) * L };
  mag_vf32_t C[NV];
  for (int v=0; v < NV; ++v) {
    if (!acc) C[v] = mag_vf32_zero();
    else if (v == NV - 1 && TAIL != L) C[v] = mag_load_bf16_partial(c + v*L, TAIL);
    else C[v] = mag_vf32_loadu_bf16(c + v*L);
  }
  for (int64_t k=0; k < kc; ++k) {
    if ((k & (MAG_PF_GROUP - 1)) == 0) {
      mag_simd_prefetch_t0(b + (k + MAG_PFDIST_B_L1)*ldb);
      mag_simd_prefetch_t1(b + (k + MAG_PFDIST_B_L2)*ldb);
      mag_simd_prefetch_t0(a + (k + MAG_PFDIST_A_L1));
      mag_simd_prefetch_t1(a + (k + MAG_PFDIST_A_L2));
    }
    mag_vf32_t Av = mag_vf32_splat(mag_bfloat16_to_float32(a[k]));
    for (int v=0; v < NV; ++v) {
      mag_vf32_t Bv = (v == NV - 1 && TAIL != L)
        ? mag_load_bf16_partial(b + k*ldb + v*L, TAIL)
        : mag_vf32_loadu_bf16(b + k*ldb + v*L);
      C[v] = mag_vf32_fmadd(Av, Bv, C[v]);
    }
  }
  for (int v=0; v < NV; ++v) {
    if (v == NV - 1 && TAIL != L) mag_store_bf16_partial(c + v*L, C[v], TAIL);
    else mag_vf32_storeu_bf16(c + v*L, C[v]);
  }
}

static MAG_AINLINE void mag_mm_tile_1x16_bfloat16(
  int64_t kc,
  const mag_bfloat16_t *restrict a,
  const mag_bfloat16_t *restrict b, ptrdiff_t ldb,
  mag_bfloat16_t *restrict c,
  bool acc
) {
  enum { COLS = 16, L = MAG_L, NV = (COLS + L - 1) / L, TAIL = COLS - (NV - 1) * L };
  mag_vf32_t C[NV];
  for (int v=0; v < NV; ++v) {
    if (!acc) C[v] = mag_vf32_zero();
    else if (v == NV - 1 && TAIL != L) C[v] = mag_load_bf16_partial(c + v*L, TAIL);
    else C[v] = mag_vf32_loadu_bf16(c + v*L);
  }
  for (int64_t k=0; k < kc; ++k) {
    if ((k & (MAG_PF_GROUP - 1)) == 0) {
      mag_simd_prefetch_t0(b + (k + MAG_PFDIST_B_L1)*ldb);
      mag_simd_prefetch_t1(b + (k + MAG_PFDIST_B_L2)*ldb);
      mag_simd_prefetch_t0(a + (k + MAG_PFDIST_A_L1));
      mag_simd_prefetch_t1(a + (k + MAG_PFDIST_A_L2));
    }
    mag_vf32_t Av = mag_vf32_splat(mag_bfloat16_to_float32(a[k]));
    for (int v=0; v < NV; ++v) {
      mag_vf32_t Bv = (v == NV - 1 && TAIL != L)
        ? mag_load_bf16_partial(b + k*ldb + v*L, TAIL)
        : mag_vf32_loadu_bf16(b + k*ldb + v*L);
      C[v] = mag_vf32_fmadd(Av, Bv, C[v]);
    }
  }
  for (int v=0; v < NV; ++v) {
    if (v == NV - 1 && TAIL != L) mag_store_bf16_partial(c + v*L, C[v], TAIL);
    else mag_vf32_storeu_bf16(c + v*L, C[v]);
  }
}

static MAG_AINLINE void mag_mm_tile_1x32_bfloat16(
  int64_t kc,
  const mag_bfloat16_t *restrict a,
  const mag_bfloat16_t *restrict b, ptrdiff_t ldb,
  mag_bfloat16_t *restrict c,
  bool acc
) {
  mag_mm_tile_1x16_bfloat16(kc, a, b,    ldb, c,    acc);
  mag_mm_tile_1x16_bfloat16(kc, a, b+16, ldb, c+16, acc);
}

static MAG_AINLINE void mag_mm_tile_16x16_bfloat16(
  int64_t kc,
  const mag_bfloat16_t *restrict a, ptrdiff_t lda,
  const mag_bfloat16_t *restrict b, ptrdiff_t ldb,
  mag_bfloat16_t *restrict c, ptrdiff_t ldc,
  bool acc
) {
  mag_mm_tile_8x16_bfloat16(kc, a,         lda, b, ldb, c,         ldc, acc);
  mag_mm_tile_8x16_bfloat16(kc, a + 8*lda, lda, b, ldb, c + 8*ldc, ldc, acc);
}

static MAG_AINLINE void mag_mm_tile_16x32_bfloat16(
  int64_t kc,
  const mag_bfloat16_t *restrict a, ptrdiff_t lda,
  const mag_bfloat16_t *restrict b, ptrdiff_t ldb,
  mag_bfloat16_t *restrict c, ptrdiff_t ldc,
  bool acc
) {
  mag_mm_tile_16x16_bfloat16(kc, a, lda, b,    ldb, c,    ldc, acc);
  mag_mm_tile_16x16_bfloat16(kc, a, lda, b+16, ldb, c+16, ldc, acc);
}

static MAG_AINLINE void mag_mm_pack_B_kc_nc_float32(int64_t kc, int64_t nc, const float *restrict Bsrc, ptrdiff_t strideK, ptrdiff_t strideN, float *restrict Bp) {
  if (strideN == 1) {
    for (int64_t k=0; k < kc; ++k) {
      const float *src = Bsrc + k*strideK;
      float *dst = Bp + k*nc;
      int64_t j = 0;
      for (; j + 4*MAG_L <= nc; j += 4*MAG_L) {
        mag_simd_prefetch_t0(src + j + 16*MAG_L);
        mag_simd_prefetch_t1(src + j + 64*MAG_L);
        mag_vf32_storeu(dst + j + 0*MAG_L, mag_vf32_loadu(src + j + 0*MAG_L));
        mag_vf32_storeu(dst + j + 1*MAG_L, mag_vf32_loadu(src + j + 1*MAG_L));
        mag_vf32_storeu(dst + j + 2*MAG_L, mag_vf32_loadu(src + j + 2*MAG_L));
        mag_vf32_storeu(dst + j + 3*MAG_L, mag_vf32_loadu(src + j + 3*MAG_L));
      }
      for (; j + MAG_L <= nc; j += MAG_L)
        mag_vf32_storeu(dst + j, mag_vf32_loadu(src + j));
      if (j < nc) {
        int rem = (int)(nc - j);
        mag_vf32_storeu_masked(dst + j, mag_vf32_loadu_masked(src + j, rem), rem);
      }
    }
  } else {
    for (int64_t k=0; k < kc; ++k) {
      const float *src = Bsrc + k*strideK;
      for (int64_t j=0; j < nc; ++j)
        Bp[k*nc + j] = src[j*strideN];
    }
  }
}

static MAG_AINLINE void mag_mm_pack_B_vec_float32(int64_t kc, int64_t nc, const float *restrict yvec, float *restrict Bp) {
  for (int64_t k=0; k < kc; ++k) {
    mag_vf32_t val = mag_vf32_broadcast(yvec + k);
    float *dst = Bp + k*nc;
    int64_t j = 0;
    for (; j + 4*MAG_L <= nc; j += 4*MAG_L) {
      mag_vf32_storeu(dst + j + 0*MAG_L, val);
      mag_vf32_storeu(dst + j + 1*MAG_L, val);
      mag_vf32_storeu(dst + j + 2*MAG_L, val);
      mag_vf32_storeu(dst + j + 3*MAG_L, val);
    }
    for (; j + MAG_L <= nc; j += MAG_L)
      mag_vf32_storeu(dst + j, val);
    if (j < nc) {
      int rem = (int)(nc - j);
      mag_vf32_storeu_masked(dst + j, val, rem);
    }
  }
}

static MAG_AINLINE void mag_mm_pack_B_kc_nc_bfloat16(
  int64_t kc, int64_t nc,
  const mag_bfloat16_t *restrict Bsrc, ptrdiff_t strideK, ptrdiff_t strideN,
  mag_bfloat16_t *restrict Bp
) {
  if (strideN == 1) {
    for (int64_t k=0; k < kc; ++k) {
      if (k + 1 < kc) mag_simd_prefetch_t0((const char *)(Bsrc + (k + 1) * strideK));
      memcpy(Bp + k*nc, Bsrc + k*strideK, (size_t)nc * sizeof(*Bsrc));
    }
  } else {
    for (int64_t k=0; k < kc; ++k) {
      const mag_bfloat16_t *src = Bsrc + k*strideK;
      for (int64_t j=0; j < nc; ++j)
        Bp[k*nc + j] = src[j*strideN];
    }
  }
}

static MAG_AINLINE void mag_mm_pack_B_vec_bfloat16(
  int64_t kc, int64_t nc,
  const mag_bfloat16_t *restrict yvec,
  mag_bfloat16_t *restrict Bp
) {
  for (int64_t k=0; k < kc; ++k) {
    mag_bfloat16_t v = yvec[k];
    mag_bfloat16_t *dst = Bp + k*nc;
    for (int64_t j=0; j < nc; ++j) dst[j] = v;
  }
}

static MAG_AINLINE void mag_mm_pack_A_mc_kc_panel8_float32(int64_t kc, int64_t mr, const float *restrict ra, ptrdiff_t sMx, ptrdiff_t sKx, float *restrict pa) {
  int64_t m8 = mr & ~7;
  for (int64_t i=0; i < m8; i += 8) {
    const float *p0 = ra + (i+0)*sMx;
    const float *p1 = ra + (i+1)*sMx;
    const float *p2 = ra + (i+2)*sMx;
    const float *p3 = ra + (i+3)*sMx;
    const float *p4 = ra + (i+4)*sMx;
    const float *p5 = ra + (i+5)*sMx;
    const float *p6 = ra + (i+6)*sMx;
    const float *p7 = ra + (i+7)*sMx;
    float *dst = pa + i*kc;
    for (int64_t k=0; k < kc; ++k) {
      if ((k & (MAG_PF_GROUP - 1)) == 0) {
        mag_simd_prefetch_t0(p0 + (int64_t)MAG_PFDIST_A_L1*sKx);
        mag_simd_prefetch_t0(p4 + (int64_t)MAG_PFDIST_A_L1*sKx);
        mag_simd_prefetch_t1(p0 + (int64_t)MAG_PFDIST_A_L2*sKx);
        mag_simd_prefetch_t1(p4 + (int64_t)MAG_PFDIST_A_L2*sKx);
      }
      dst[k*8 + 0] = p0[k*sKx];
      dst[k*8 + 1] = p1[k*sKx];
      dst[k*8 + 2] = p2[k*sKx];
      dst[k*8 + 3] = p3[k*sKx];
      dst[k*8 + 4] = p4[k*sKx];
      dst[k*8 + 5] = p5[k*sKx];
      dst[k*8 + 6] = p6[k*sKx];
      dst[k*8 + 7] = p7[k*sKx];
    }
  }
  for (int64_t i=m8; i < mr; ++i) {
    const float *src = ra + i*sMx;
    float *dst = pa + i*kc;
    for (int64_t k=0; k < kc; ++k)
      dst[k] = src[k*sKx];
  }
}

static MAG_AINLINE void mag_mm_pack_A_mc_kc_panel8_bfloat16(
  int64_t kc, int64_t mr,
  const mag_bfloat16_t *restrict ra, ptrdiff_t sMx, ptrdiff_t sKx,
  mag_bfloat16_t *restrict pa
) {
  int64_t m8 = mr & ~7;
  for (int64_t i=0; i < m8; i += 8) {
    const mag_bfloat16_t *p0 = ra + (i+0)*sMx;
    const mag_bfloat16_t *p1 = ra + (i+1)*sMx;
    const mag_bfloat16_t *p2 = ra + (i+2)*sMx;
    const mag_bfloat16_t *p3 = ra + (i+3)*sMx;
    const mag_bfloat16_t *p4 = ra + (i+4)*sMx;
    const mag_bfloat16_t *p5 = ra + (i+5)*sMx;
    const mag_bfloat16_t *p6 = ra + (i+6)*sMx;
    const mag_bfloat16_t *p7 = ra + (i+7)*sMx;
    mag_bfloat16_t *dst = pa + i*kc;
    for (int64_t k=0; k < kc; ++k) {
      if ((k & (MAG_PF_GROUP - 1)) == 0) {
        mag_simd_prefetch_t0(p0 + (int64_t)MAG_PFDIST_A_L1*sKx);
        mag_simd_prefetch_t0(p4 + (int64_t)MAG_PFDIST_A_L1*sKx);
        mag_simd_prefetch_t1(p0 + (int64_t)MAG_PFDIST_A_L2*sKx);
        mag_simd_prefetch_t1(p4 + (int64_t)MAG_PFDIST_A_L2*sKx);
      }
      dst[k*8 + 0] = p0[k*sKx];
      dst[k*8 + 1] = p1[k*sKx];
      dst[k*8 + 2] = p2[k*sKx];
      dst[k*8 + 3] = p3[k*sKx];
      dst[k*8 + 4] = p4[k*sKx];
      dst[k*8 + 5] = p5[k*sKx];
      dst[k*8 + 6] = p6[k*sKx];
      dst[k*8 + 7] = p7[k*sKx];
    }
  }
  for (int64_t i=m8; i < mr; ++i) {
    const mag_bfloat16_t *src = ra + i*sMx;
    mag_bfloat16_t *dst = pa + i*kc;
    for (int64_t k=0; k < kc; ++k)
      dst[k] = src[k*sKx];
  }
}

/* ===========================================================================
 *  GEMV: C[N] = A[K] * B[K,N], B row-major with stride ldb in K
 * =========================================================================== */

static MAG_AINLINE void mag_gemv_float32(int64_t K, int64_t N, const float *restrict A, const float *restrict B, int64_t ldb, float *restrict C) {
  int64_t j = 0;
  for (; j + 8*MAG_L <= N; j += 8*MAG_L) {
    mag_vf32_t s0 = mag_vf32_zero(), s1 = mag_vf32_zero();
    mag_vf32_t s2 = mag_vf32_zero(), s3 = mag_vf32_zero();
    mag_vf32_t s4 = mag_vf32_zero(), s5 = mag_vf32_zero();
    mag_vf32_t s6 = mag_vf32_zero(), s7 = mag_vf32_zero();
    const float *brow = B + j;
    for (int64_t k=0; k < K; ++k, brow += ldb) {
      mag_vf32_t a = mag_vf32_broadcast(A + k);
      s0 = mag_vf32_fmadd(a, mag_vf32_loadu(brow + 0*MAG_L), s0);
      s1 = mag_vf32_fmadd(a, mag_vf32_loadu(brow + 1*MAG_L), s1);
      s2 = mag_vf32_fmadd(a, mag_vf32_loadu(brow + 2*MAG_L), s2);
      s3 = mag_vf32_fmadd(a, mag_vf32_loadu(brow + 3*MAG_L), s3);
      s4 = mag_vf32_fmadd(a, mag_vf32_loadu(brow + 4*MAG_L), s4);
      s5 = mag_vf32_fmadd(a, mag_vf32_loadu(brow + 5*MAG_L), s5);
      s6 = mag_vf32_fmadd(a, mag_vf32_loadu(brow + 6*MAG_L), s6);
      s7 = mag_vf32_fmadd(a, mag_vf32_loadu(brow + 7*MAG_L), s7);
    }
    mag_vf32_storeu(C + j + 0*MAG_L, s0);
    mag_vf32_storeu(C + j + 1*MAG_L, s1);
    mag_vf32_storeu(C + j + 2*MAG_L, s2);
    mag_vf32_storeu(C + j + 3*MAG_L, s3);
    mag_vf32_storeu(C + j + 4*MAG_L, s4);
    mag_vf32_storeu(C + j + 5*MAG_L, s5);
    mag_vf32_storeu(C + j + 6*MAG_L, s6);
    mag_vf32_storeu(C + j + 7*MAG_L, s7);
  }
  for (; j + 4*MAG_L <= N; j += 4*MAG_L) {
    mag_vf32_t s0 = mag_vf32_zero(), s1 = mag_vf32_zero();
    mag_vf32_t s2 = mag_vf32_zero(), s3 = mag_vf32_zero();
    const float *brow = B + j;
    for (int64_t k=0; k < K; ++k, brow += ldb) {
      mag_vf32_t a = mag_vf32_broadcast(A + k);
      s0 = mag_vf32_fmadd(a, mag_vf32_loadu(brow + 0*MAG_L), s0);
      s1 = mag_vf32_fmadd(a, mag_vf32_loadu(brow + 1*MAG_L), s1);
      s2 = mag_vf32_fmadd(a, mag_vf32_loadu(brow + 2*MAG_L), s2);
      s3 = mag_vf32_fmadd(a, mag_vf32_loadu(brow + 3*MAG_L), s3);
    }
    mag_vf32_storeu(C + j + 0*MAG_L, s0);
    mag_vf32_storeu(C + j + 1*MAG_L, s1);
    mag_vf32_storeu(C + j + 2*MAG_L, s2);
    mag_vf32_storeu(C + j + 3*MAG_L, s3);
  }
  for (; j + MAG_L <= N; j += MAG_L) {
    mag_vf32_t s = mag_vf32_zero();
    const float *brow = B + j;
    for (int64_t k=0; k < K; ++k, brow += ldb)
      s = mag_vf32_fmadd(mag_vf32_broadcast(A + k), mag_vf32_loadu(brow), s);
    mag_vf32_storeu(C + j, s);
  }
  if (j < N) {
    int rem = (int)(N - j);
    mag_vf32_t s = mag_vf32_zero();
    const float *brow = B + j;
    for (int64_t k=0; k < K; ++k, brow += ldb)
      s = mag_vf32_fmadd(mag_vf32_broadcast(A + k), mag_vf32_loadu_masked(brow, rem), s);
    mag_vf32_storeu_masked(C + j, s, rem);
  }
}

static MAG_AINLINE void mag_gemv_bfloat16(
  int64_t K, int64_t N,
  const mag_bfloat16_t *restrict A,
  const mag_bfloat16_t *restrict B,
  int64_t ldb,
  mag_bfloat16_t *restrict C
) {
  int64_t j = 0;
  for (; j + 4*MAG_L <= N; j += 4*MAG_L) {
    mag_vf32_t s0 = mag_vf32_zero(), s1 = mag_vf32_zero();
    mag_vf32_t s2 = mag_vf32_zero(), s3 = mag_vf32_zero();
    const mag_bfloat16_t *brow = B + j;
    for (int64_t k=0; k < K; ++k, brow += ldb) {
      mag_vf32_t a = mag_vf32_splat(mag_bfloat16_to_float32(A[k]));
      s0 = mag_vf32_fmadd(a, mag_vf32_loadu_bf16(brow + 0*MAG_L), s0);
      s1 = mag_vf32_fmadd(a, mag_vf32_loadu_bf16(brow + 1*MAG_L), s1);
      s2 = mag_vf32_fmadd(a, mag_vf32_loadu_bf16(brow + 2*MAG_L), s2);
      s3 = mag_vf32_fmadd(a, mag_vf32_loadu_bf16(brow + 3*MAG_L), s3);
    }
    mag_vf32_storeu_bf16(C + j + 0*MAG_L, s0);
    mag_vf32_storeu_bf16(C + j + 1*MAG_L, s1);
    mag_vf32_storeu_bf16(C + j + 2*MAG_L, s2);
    mag_vf32_storeu_bf16(C + j + 3*MAG_L, s3);
  }
  for (; j + MAG_L <= N; j += MAG_L) {
    mag_vf32_t s = mag_vf32_zero();
    const mag_bfloat16_t *brow = B + j;
    for (int64_t k=0; k < K; ++k, brow += ldb) {
      mag_vf32_t a = mag_vf32_splat(mag_bfloat16_to_float32(A[k]));
      s = mag_vf32_fmadd(a, mag_vf32_loadu_bf16(brow), s);
    }
    mag_vf32_storeu_bf16(C + j, s);
  }
  for (; j < N; ++j) {
    float sum = 0.0f;
    for (int64_t k=0; k < K; ++k)
      sum += mag_bfloat16_to_float32(A[k]) * mag_bfloat16_to_float32(B[k*ldb + j]);
    C[j] = mag_float32_to_bfloat16(sum);
  }
}

static MAG_HOTPROC void mag_mm_block_float32(int64_t kc, int64_t mr, int64_t nr, const float *A, int64_t lda, const float *B, int64_t ldb, float *C, int64_t ldc, bool acc) {
  int64_t j = 0;
  for (; nr-j >= 32; j += 32) {
    int64_t i = 0;
    for (; mr-i >= 16; i += 16) mag_mm_tile_16x32_float32(kc, A + i*lda, lda, B + j, ldb, C + i*ldc + j, ldc, acc);
    for (; mr-i >= 8;  i +=  8) mag_mm_tile_8x32_float32 (kc, A + i*lda, lda, B + j, ldb, C + i*ldc + j, ldc, acc);
    for (; i < mr; ++i)         mag_mm_tile_1x32_float32 (kc, A + i*lda,      B + j, ldb, C + i*ldc + j, acc);
  }
  for (; nr-j >= 16; j += 16) {
    int64_t i = 0;
    for (; mr-i >= 8; i += 8) mag_mm_tile_8x16_float32(kc, A + i*lda, lda, B + j, ldb, C + i*ldc + j, ldc, acc);
    for (; i < mr; ++i)       mag_mm_tile_1x16_float32(kc, A + i*lda,      B + j, ldb, C + i*ldc + j, acc);
  }
  for (; nr-j >= 8; j += 8) {
    int64_t i = 0;
    for (; mr-i >= 8; i += 8) mag_mm_tile_8x8_float32(kc, A + i*lda, lda, B + j, ldb, C + i*ldc + j, ldc, acc);
    for (; i < mr; ++i)       mag_mm_tile_1x8_float32(kc, A + i*lda,      B + j, ldb, C + i*ldc + j, acc);
  }
  int64_t rem = nr - j;
  if (!rem) return;
  for (int64_t i2=0; i2 < mr; ++i2) {
    const float *ap = A + i2*lda;
    float *cp = C + i2*ldc + j;
    for (int64_t jj=0; jj < rem; ++jj) {
      float sum = acc ? cp[jj] : 0.f;
      for (int64_t k=0; k < kc; ++k)
        sum += ap[k]*B[k*ldb + (j + jj)];
      cp[jj] = sum;
    }
  }
}

static MAG_HOTPROC void mag_mm_block_bfloat16(
  int64_t kc, int64_t mr, int64_t nr,
  const mag_bfloat16_t *A, int64_t lda,
  const mag_bfloat16_t *B, int64_t ldb,
  mag_bfloat16_t *C, int64_t ldc,
  bool acc
) {
  int64_t j = 0;
  for (; nr-j >= 32; j += 32) {
    int64_t i = 0;
    for (; mr-i >= 16; i += 16) mag_mm_tile_16x32_bfloat16(kc, A + i*lda, lda, B + j, ldb, C + i*ldc + j, ldc, acc);
    for (; mr-i >=  8; i +=  8) mag_mm_tile_8x32_bfloat16 (kc, A + i*lda, lda, B + j, ldb, C + i*ldc + j, ldc, acc);
    for (; i < mr; ++i)         mag_mm_tile_1x32_bfloat16 (kc, A + i*lda,      B + j, ldb, C + i*ldc + j, acc);
  }
  for (; nr-j >= 16; j += 16) {
    int64_t i = 0;
    for (; mr-i >= 8; i += 8) mag_mm_tile_8x16_bfloat16(kc, A + i*lda, lda, B + j, ldb, C + i*ldc + j, ldc, acc);
    for (; i < mr; ++i)       mag_mm_tile_1x16_bfloat16(kc, A + i*lda,      B + j, ldb, C + i*ldc + j, acc);
  }
  for (; nr-j >= 8; j += 8) {
    int64_t i = 0;
    for (; mr-i >= 8; i += 8) mag_mm_tile_8x8_bfloat16(kc, A + i*lda, lda, B + j, ldb, C + i*ldc + j, ldc, acc);
    for (; i < mr; ++i)       mag_mm_tile_1x8_bfloat16(kc, A + i*lda,      B + j, ldb, C + i*ldc + j, acc);
  }
  int64_t rem = nr - j;
  if (!rem) return;
  for (int64_t i2=0; i2 < mr; ++i2) {
    const mag_bfloat16_t *ap = A + i2*lda;
    mag_bfloat16_t *cp = C + i2*ldc + j;
    for (int64_t jj=0; jj < rem; ++jj) {
      float sum = acc ? mag_bfloat16_to_float32(cp[jj]) : 0.0f;
      for (int64_t k=0; k < kc; ++k)
        sum += mag_bfloat16_to_float32(ap[k]) * mag_bfloat16_to_float32(B[k*ldb + (j + jj)]);
      cp[jj] = mag_float32_to_bfloat16(sum);
    }
  }
}

static int64_t mag_offset_rmn(const mag_tensor_t *t, int64_t flat, int64_t i, int64_t j) {
  int64_t ra = t->coords.rank;
  const int64_t *restrict td = t->coords.shape;
  const int64_t *restrict ts = t->coords.strides;
  if (mag_likely(ra <= 3)) { /* Fast path */
    switch (ra) {
      case 0: return 0;
      case 1: return flat*ts[0];
      case 2: return i*ts[0] + j*ts[1];
      case 3: return flat*ts[0] + i*ts[1] + j*ts[2];
    }
  }
  int64_t off = 0, rem = flat;
  for (int64_t d = ra-3; d >= 0; --d) {
    int64_t idx = rem % td[d];
    rem /= td[d];
    off += idx*ts[d];
  }
  off += i*ts[ra-2];
  off += j*ts[ra-1];
  return off;
}

MAG_HOTPROC static mag_status_t mag_matmul_float32(mag_error_t *err, const mag_kernel_payload_t *payload) {
  (void)err;
  mag_tensor_t *r = mag_cmd_out(0);
  const mag_tensor_t *x = mag_cmd_in(0);
  const mag_tensor_t *y = mag_cmd_in(1);
  const float *bx = (const float *)mag_tensor_data_ptr(x);
  const float *by = (const float *)mag_tensor_data_ptr(y);
  float *br = (float *)mag_tensor_data_ptr_mut(r);
  int64_t MR = payload->mm_params.MR;
  int64_t MC = payload->mm_params.MC;
  int64_t KC = payload->mm_params.KC;
  int64_t NC = payload->mm_params.NC;
  int64_t NR = payload->mm_params.NR;
  int64_t M = x->coords.rank == 1 ? 1 : x->coords.shape[x->coords.rank-2];
  int64_t N = y->coords.rank == 1 ? 1 : y->coords.shape[y->coords.rank-1];
  int64_t K = x->coords.shape[x->coords.rank-1];
  int64_t bdr = r->coords.rank > 2 ? r->coords.rank - 2 : 0;
  int64_t batch_total = 1;
  for (int64_t d=0; d < bdr; ++d)
    batch_total *= r->coords.shape[d];
  if (M == 1 && K >= 128 && N >= 4096 && y->coords.rank == 2 && y->coords.strides[y->coords.rank-1] == 1) { /* Detect GEMV */
    int64_t nth = payload->thread_num;
    int64_t tid = payload->thread_idx;
    int64_t j_per_thread = (N + nth - 1) / nth;
    int64_t j0 = tid*j_per_thread;
    int64_t j1 = mag_xmin(N, j0 + j_per_thread);
    for (int64_t batch = 0; batch < batch_total; ++batch) {
      const float *A = bx + mag_offset_rmn(x, batch, 0, 0);
      const float *B = by + mag_offset_rmn(y, batch, 0, 0) + j0;
      float *C = br + mag_offset_rmn(r, batch, 0, 0) + j0;
      mag_gemv_float32(K, j1 - j0, A, B, N, C);
    }
    return MAG_STATUS_OK;
  }
  int64_t bdx = x->coords.rank > 2 ? x->coords.rank-2 : 0;
  int64_t bdy = y->coords.rank > 2 ? y->coords.rank-2 : 0;
  int64_t tic = (M+MC-1)/MC;
  int64_t tjc = (N+NC-1)/NC;
  int64_t tpb = tic*tjc;
  int64_t tt = batch_total*tpb;
  mag_scratch_arena_clear(&mag_tls_arena);
  float *scratch = mag_scratch_arena_alloc(&mag_tls_arena, sizeof(*scratch)*(KC*NC + MC*KC));
  float *Bp = scratch;
  float *Ap = Bp + KC*NC;
  for (;;) {
    int64_t tile = mag_atomic64_fetch_add(payload->mm_next_tile, 1, MAG_MO_RELAXED);
    if (tile >= tt) break;
    int64_t batch_idx = tile / tpb;
    int64_t rem = tile % tpb;
    int64_t jc = rem % tjc;
    int64_t ic = rem / tjc;
    int64_t idx_r[MAG_MAX_DIMS] = {0};
    for (int64_t d=bdr-1, t=batch_idx; d >= 0; --d) {
      idx_r[d] = t % r->coords.shape[d];
      t /= r->coords.shape[d];
    }
    int64_t xb_flat = 0;
    for (int64_t d=0; d < bdx; ++d) {
      int64_t rd = bdr - bdx + d;
      xb_flat = xb_flat*x->coords.shape[d] + (x->coords.shape[d] == 1 ? 0 : idx_r[rd]);
    }
    int64_t yb_flat = 0;
    for (int64_t d=0; d < bdy; ++d) {
      int64_t rd = bdr - bdy + d;
      yb_flat = yb_flat*y->coords.shape[d] + (y->coords.shape[d] == 1 ? 0 : idx_r[rd]);
    }
    bool yv = y->coords.rank == 1;
    const float *px_base = bx + mag_offset_rmn(x, xb_flat, 0, 0);
    const float *py_base = by + mag_offset_rmn(y, yb_flat, 0, 0);
    float *pr_base = br + mag_offset_rmn(r, batch_idx, 0, 0);
    int64_t i0 = ic*MC;
    int64_t mc = i0+MC <= M ? MC : M-i0;
    int64_t j0 = jc*NC;
    int64_t nc = j0+NC <= N ? NC : N-j0;
    int64_t sMx = x->coords.strides[x->coords.rank-2];
    int64_t sKx = x->coords.strides[x->coords.rank-1];
    int64_t sKy = yv ? 0 : y->coords.strides[y->coords.rank-2];
    int64_t sNy = yv ? 0 : y->coords.strides[y->coords.rank-1];
    for (int64_t pc = 0; pc < K; pc += KC) {
      int64_t kc = mag_xmin(KC, K - pc);
      if (y->coords.rank == 1) mag_mm_pack_B_vec_float32(kc, nc, py_base + pc, Bp);
      else mag_mm_pack_B_kc_nc_float32(kc, nc, py_base + pc*sKy +  j0*sNy, sKy, sNy, Bp);
      mag_mm_pack_A_mc_kc_panel8_float32(kc, mc,  px_base + i0*sMx + pc*sKx, sMx, sKx, Ap);
      for (int64_t ir=0; ir < mc; ir += MR)
        for (int64_t jr=0; jr < nc; jr += NR)
          mag_mm_block_float32(
            kc,
            mag_xmin(MR, mc - ir),
            mag_xmin(NR, nc - jr),
            Ap + ir*kc,
            kc,
            Bp + jr,
            nc,
            pr_base + (i0 + ir)*N + (j0 + jr),
            N,
            pc);
    }
  }
  mag_scratch_arena_clear(&mag_tls_arena);
  return MAG_STATUS_OK;
}

MAG_HOTPROC static mag_status_t mag_matmul_bfloat16(mag_error_t *err,const mag_kernel_payload_t *payload) {
  (void)err;
  mag_tensor_t *r = mag_cmd_out(0);
  const mag_tensor_t *x = mag_cmd_in(0);
  const mag_tensor_t *y = mag_cmd_in(1);
  const mag_bfloat16_t *bx = (const mag_bfloat16_t *)mag_tensor_data_ptr(x);
  const mag_bfloat16_t *by = (const mag_bfloat16_t *)mag_tensor_data_ptr(y);
  mag_bfloat16_t *br = (mag_bfloat16_t *)mag_tensor_data_ptr_mut(r);
  int64_t MR = payload->mm_params.MR;
  int64_t MC = payload->mm_params.MC;
  int64_t KC = payload->mm_params.KC;
  int64_t NC = payload->mm_params.NC;
  int64_t NR = payload->mm_params.NR;
  int64_t M = x->coords.rank == 1 ? 1 : x->coords.shape[x->coords.rank-2];
  int64_t N = y->coords.rank == 1 ? 1 : y->coords.shape[y->coords.rank-1];
  int64_t K = x->coords.shape[x->coords.rank-1];
  int64_t bdr = r->coords.rank > 2 ? r->coords.rank - 2 : 0;
  int64_t batch_total = 1;
  for (int64_t d=0; d < bdr; ++d)
    batch_total *= r->coords.shape[d];
  if (M == 1 && K >= 128 && N >= 4096 && y->coords.rank == 2 && y->coords.strides[y->coords.rank-1] == 1) { /* Detect GEMV */
    int64_t nth = payload->thread_num;
    int64_t tid = payload->thread_idx;
    int64_t j_per_thread = (N + nth - 1) / nth;
    int64_t j0 = tid*j_per_thread;
    int64_t j1 = mag_xmin(N, j0 + j_per_thread);
    for (int64_t batch = 0; batch < batch_total; ++batch) {
      const mag_bfloat16_t *A = bx + mag_offset_rmn(x, batch, 0, 0);
      const mag_bfloat16_t *B = by + mag_offset_rmn(y, batch, 0, 0) + j0;
      mag_bfloat16_t *C = br + mag_offset_rmn(r, batch, 0, 0) + j0;
      mag_gemv_bfloat16(K, j1 - j0, A, B, N, C);
    }
    return MAG_STATUS_OK;
  }
  int64_t bdx = x->coords.rank > 2 ? x->coords.rank-2 : 0;
  int64_t bdy = y->coords.rank > 2 ? y->coords.rank-2 : 0;
  int64_t tic = (M+MC-1)/MC;
  int64_t tjc = (N+NC-1)/NC;
  int64_t tpb = tic*tjc;
  int64_t tt = batch_total*tpb;
  mag_scratch_arena_clear(&mag_tls_arena);
  mag_bfloat16_t *scratch = mag_scratch_arena_alloc(&mag_tls_arena, sizeof(*scratch)*(KC*NC + MC*KC));
  mag_bfloat16_t *Bp = scratch;
  mag_bfloat16_t *Ap = Bp + KC*NC;
  for (;;) {
    int64_t tile = mag_atomic64_fetch_add(payload->mm_next_tile, 1, MAG_MO_RELAXED);
    if (tile >= tt) break;
    int64_t batch_idx = tile / tpb;
    int64_t rem = tile % tpb;
    int64_t jc = rem % tjc;
    int64_t ic = rem / tjc;
    int64_t idx_r[MAG_MAX_DIMS] = {0};
    for (int64_t d=bdr-1, t=batch_idx; d >= 0; --d) {
      idx_r[d] = t % r->coords.shape[d];
      t /= r->coords.shape[d];
    }
    int64_t xb_flat = 0;
    for (int64_t d=0; d < bdx; ++d) {
      int64_t rd = bdr - bdx + d;
      xb_flat = xb_flat*x->coords.shape[d] + (x->coords.shape[d] == 1 ? 0 : idx_r[rd]);
    }
    int64_t yb_flat = 0;
    for (int64_t d=0; d < bdy; ++d) {
      int64_t rd = bdr - bdy + d;
      yb_flat = yb_flat*y->coords.shape[d] + (y->coords.shape[d] == 1 ? 0 : idx_r[rd]);
    }
    bool yv = y->coords.rank == 1;
    const mag_bfloat16_t *px_base = bx + mag_offset_rmn(x, xb_flat, 0, 0);
    const mag_bfloat16_t *py_base = by + mag_offset_rmn(y, yb_flat, 0, 0);
    mag_bfloat16_t *pr_base = br + mag_offset_rmn(r, batch_idx, 0, 0);
    int64_t i0 = ic*MC;
    int64_t mc = i0+MC <= M ? MC : M-i0;
    int64_t j0 = jc*NC;
    int64_t nc = j0+NC <= N ? NC : N-j0;
    int64_t sMx = x->coords.strides[x->coords.rank-2];
    int64_t sKx = x->coords.strides[x->coords.rank-1];
    int64_t sKy = yv ? 0 : y->coords.strides[y->coords.rank-2];
    int64_t sNy = yv ? 0 : y->coords.strides[y->coords.rank-1];
    for (int64_t pc = 0; pc < K; pc += KC) {
      int64_t kc = mag_xmin(KC, K - pc);
      if (y->coords.rank == 1) mag_mm_pack_B_vec_bfloat16(kc, nc, py_base + pc, Bp);
      else mag_mm_pack_B_kc_nc_bfloat16(kc, nc, py_base + pc*sKy +  j0*sNy, sKy, sNy, Bp);
      mag_mm_pack_A_mc_kc_panel8_bfloat16(kc, mc,  px_base + i0*sMx + pc*sKx, sMx, sKx, Ap);
      for (int64_t ir=0; ir < mc; ir += MR)
        for (int64_t jr=0; jr < nc; jr += NR)
          mag_mm_block_bfloat16(
            kc,
            mag_xmin(MR, mc - ir),
            mag_xmin(NR, nc - jr),
            Ap + ir*kc,
            kc,
            Bp + jr,
            nc,
            pr_base + (i0 + ir)*N + (j0 + jr),
            N,
            pc);
    }
  }
  mag_scratch_arena_clear(&mag_tls_arena);
  return MAG_STATUS_OK;
}

static MAG_AINLINE mag_vf32_t mag_load_f8e4m3fn_partial(const mag_float8_e4m3fn_t *p, int n) {
  if (n == MAG_L) return mag_vf32_loadu_float8_e4m3fn(p);
  mag_alignas(64) float tmp[MAG_L];
  for (int i=0; i < MAG_L; ++i) tmp[i] = 0.f;
  for (int i=0; i < n; ++i) tmp[i] = mag_float8_e4m3fn_to_float32(p[i]);
  return mag_vf32_loadu(tmp);
}

static MAG_AINLINE void mag_store_f8e4m3fn_partial(mag_float8_e4m3fn_t *p, mag_vf32_t v, int n) {
  if (n == MAG_L) { mag_vf32_storeu_float8_e4m3fn(p, v); return; }
  mag_alignas(64) float tmp[MAG_L];
  mag_vf32_storeu(tmp, v);
  for (int i=0; i < n; ++i) p[i] = mag_float32_to_float8_e4m3fn(tmp[i]);
}

static MAG_AINLINE void mag_mm_tile_8x8_float8_e4m3fn(
  int64_t kc,
  const mag_float8_e4m3fn_t *restrict a, ptrdiff_t lda,
  const mag_float8_e4m3fn_t *restrict b, ptrdiff_t ldb,
  mag_float8_e4m3fn_t *restrict c, ptrdiff_t ldc,
  bool acc
) {
  (void)lda;
  enum { ROWS = 8, COLS = 8, L = MAG_L, NV = (COLS + L - 1) / L, TAIL = COLS - (NV - 1) * L };

  mag_vf32_t C[ROWS][NV];

  for (int r=0; r < ROWS; ++r) {
    for (int v=0; v < NV; ++v) {
      if (!acc) C[r][v] = mag_vf32_zero();
      else if (v == NV - 1 && TAIL != L) C[r][v] = mag_load_f8e4m3fn_partial(c + r*ldc + v*L, TAIL);
      else C[r][v] = mag_vf32_loadu_float8_e4m3fn(c + r*ldc + v*L);
    }
  }

  for (int64_t k=0; k < kc; ++k) {
    if ((k & (MAG_PF_GROUP - 1)) == 0) {
      mag_simd_prefetch_t0(b + (k + MAG_PFDIST_B_L1)*ldb);
      mag_simd_prefetch_t1(b + (k + MAG_PFDIST_B_L2)*ldb);
      mag_simd_prefetch_t0(a + (k + MAG_PFDIST_A_L1)*8);
      mag_simd_prefetch_t1(a + (k + MAG_PFDIST_A_L2)*8);
    }

    mag_vf32_t Bv[NV];
    for (int v=0; v < NV; ++v) {
      if (v == NV - 1 && TAIL != L) Bv[v] = mag_load_f8e4m3fn_partial(b + k*ldb + v*L, TAIL);
      else Bv[v] = mag_vf32_loadu_float8_e4m3fn(b + k*ldb + v*L);
    }

    const mag_float8_e4m3fn_t *ak = a + k*8;
    for (int r=0; r < ROWS; ++r) {
      mag_vf32_t Av = mag_vf32_splat(mag_float8_e4m3fn_to_float32(ak[r]));
      for (int v=0; v < NV; ++v)
        C[r][v] = mag_vf32_fmadd(Av, Bv[v], C[r][v]);
    }
  }

  for (int r=0; r < ROWS; ++r) {
    for (int v=0; v < NV; ++v) {
      if (v == NV - 1 && TAIL != L) mag_store_f8e4m3fn_partial(c + r*ldc + v*L, C[r][v], TAIL);
      else mag_vf32_storeu_float8_e4m3fn(c + r*ldc + v*L, C[r][v]);
    }
  }
}

static MAG_AINLINE void mag_mm_tile_8x16_float8_e4m3fn(
  int64_t kc,
  const mag_float8_e4m3fn_t *restrict a, ptrdiff_t lda,
  const mag_float8_e4m3fn_t *restrict b, ptrdiff_t ldb,
  mag_float8_e4m3fn_t *restrict c, ptrdiff_t ldc,
  bool acc
) {
  mag_mm_tile_8x8_float8_e4m3fn(kc, a, lda, b,   ldb, c,   ldc, acc);
  mag_mm_tile_8x8_float8_e4m3fn(kc, a, lda, b+8, ldb, c+8, ldc, acc);
}

static MAG_AINLINE void mag_mm_tile_8x32_float8_e4m3fn(
  int64_t kc,
  const mag_float8_e4m3fn_t *restrict a, ptrdiff_t lda,
  const mag_float8_e4m3fn_t *restrict b, ptrdiff_t ldb,
  mag_float8_e4m3fn_t *restrict c, ptrdiff_t ldc,
  bool acc
) {
  mag_mm_tile_8x16_float8_e4m3fn(kc, a, lda, b,    ldb, c,    ldc, acc);
  mag_mm_tile_8x16_float8_e4m3fn(kc, a, lda, b+16, ldb, c+16, ldc, acc);
}

static MAG_AINLINE void mag_mm_tile_1x8_float8_e4m3fn(
  int64_t kc,
  const mag_float8_e4m3fn_t *restrict a,
  const mag_float8_e4m3fn_t *restrict b, ptrdiff_t ldb,
  mag_float8_e4m3fn_t *restrict c,
  bool acc
) {
  enum { COLS = 8, L = MAG_L, NV = (COLS + L - 1) / L, TAIL = COLS - (NV - 1) * L };
  mag_vf32_t C[NV];

  for (int v=0; v < NV; ++v) {
    if (!acc) C[v] = mag_vf32_zero();
    else if (v == NV - 1 && TAIL != L) C[v] = mag_load_f8e4m3fn_partial(c + v*L, TAIL);
    else C[v] = mag_vf32_loadu_float8_e4m3fn(c + v*L);
  }

  for (int64_t k=0; k < kc; ++k) {
    if ((k & (MAG_PF_GROUP - 1)) == 0) {
      mag_simd_prefetch_t0(b + (k + MAG_PFDIST_B_L1)*ldb);
      mag_simd_prefetch_t1(b + (k + MAG_PFDIST_B_L2)*ldb);
      mag_simd_prefetch_t0(a + (k + MAG_PFDIST_A_L1));
      mag_simd_prefetch_t1(a + (k + MAG_PFDIST_A_L2));
    }

    mag_vf32_t Av = mag_vf32_splat(mag_float8_e4m3fn_to_float32(a[k]));
    for (int v=0; v < NV; ++v) {
      mag_vf32_t Bv = (v == NV - 1 && TAIL != L)
        ? mag_load_f8e4m3fn_partial(b + k*ldb + v*L, TAIL)
        : mag_vf32_loadu_float8_e4m3fn(b + k*ldb + v*L);
      C[v] = mag_vf32_fmadd(Av, Bv, C[v]);
    }
  }

  for (int v=0; v < NV; ++v) {
    if (v == NV - 1 && TAIL != L) mag_store_f8e4m3fn_partial(c + v*L, C[v], TAIL);
    else mag_vf32_storeu_float8_e4m3fn(c + v*L, C[v]);
  }
}

static MAG_AINLINE void mag_mm_tile_1x16_float8_e4m3fn(
  int64_t kc,
  const mag_float8_e4m3fn_t *restrict a,
  const mag_float8_e4m3fn_t *restrict b, ptrdiff_t ldb,
  mag_float8_e4m3fn_t *restrict c,
  bool acc
) {
  mag_mm_tile_1x8_float8_e4m3fn(kc, a, b,   ldb, c,   acc);
  mag_mm_tile_1x8_float8_e4m3fn(kc, a, b+8, ldb, c+8, acc);
}

static MAG_AINLINE void mag_mm_tile_1x32_float8_e4m3fn(
  int64_t kc,
  const mag_float8_e4m3fn_t *restrict a,
  const mag_float8_e4m3fn_t *restrict b, ptrdiff_t ldb,
  mag_float8_e4m3fn_t *restrict c,
  bool acc
) {
  mag_mm_tile_1x16_float8_e4m3fn(kc, a, b,    ldb, c,    acc);
  mag_mm_tile_1x16_float8_e4m3fn(kc, a, b+16, ldb, c+16, acc);
}

static MAG_AINLINE void mag_mm_tile_16x16_float8_e4m3fn(
  int64_t kc,
  const mag_float8_e4m3fn_t *restrict a, ptrdiff_t lda,
  const mag_float8_e4m3fn_t *restrict b, ptrdiff_t ldb,
  mag_float8_e4m3fn_t *restrict c, ptrdiff_t ldc,
  bool acc
) {
  mag_mm_tile_8x16_float8_e4m3fn(kc, a,         lda, b, ldb, c,         ldc, acc);
  mag_mm_tile_8x16_float8_e4m3fn(kc, a + 8*lda, lda, b, ldb, c + 8*ldc, ldc, acc);
}

static MAG_AINLINE void mag_mm_tile_16x32_float8_e4m3fn(
  int64_t kc,
  const mag_float8_e4m3fn_t *restrict a, ptrdiff_t lda,
  const mag_float8_e4m3fn_t *restrict b, ptrdiff_t ldb,
  mag_float8_e4m3fn_t *restrict c, ptrdiff_t ldc,
  bool acc
) {
  mag_mm_tile_16x16_float8_e4m3fn(kc, a, lda, b,    ldb, c,    ldc, acc);
  mag_mm_tile_16x16_float8_e4m3fn(kc, a, lda, b+16, ldb, c+16, ldc, acc);
}

static MAG_AINLINE void mag_mm_pack_B_kc_nc_float8_e4m3fn(
  int64_t kc, int64_t nc,
  const mag_float8_e4m3fn_t *restrict Bsrc, ptrdiff_t strideK, ptrdiff_t strideN,
  mag_float8_e4m3fn_t *restrict Bp
) {
  if (strideN == 1) {
    for (int64_t k=0; k < kc; ++k) {
      if (k + 1 < kc) mag_simd_prefetch_t0((const char *)(Bsrc + (k + 1) * strideK));
      memcpy(Bp + k*nc, Bsrc + k*strideK, (size_t)nc * sizeof(*Bsrc));
    }
  } else {
    for (int64_t k=0; k < kc; ++k) {
      const mag_float8_e4m3fn_t *src = Bsrc + k*strideK;
      for (int64_t j=0; j < nc; ++j)
        Bp[k*nc + j] = src[j*strideN];
    }
  }
}

static MAG_AINLINE void mag_mm_pack_B_vec_float8_e4m3fn(
  int64_t kc, int64_t nc,
  const mag_float8_e4m3fn_t *restrict yvec,
  mag_float8_e4m3fn_t *restrict Bp
) {
  for (int64_t k=0; k < kc; ++k) {
    mag_float8_e4m3fn_t v = yvec[k];
    mag_float8_e4m3fn_t *dst = Bp + k*nc;
    for (int64_t j=0; j < nc; ++j) dst[j] = v;
  }
}

static MAG_AINLINE void mag_mm_pack_A_mc_kc_panel8_float8_e4m3fn(
  int64_t kc, int64_t mr,
  const mag_float8_e4m3fn_t *restrict ra, ptrdiff_t sMx, ptrdiff_t sKx,
  mag_float8_e4m3fn_t *restrict pa
) {
  int64_t m8 = mr & ~7;

  for (int64_t i=0; i < m8; i += 8) {
    const mag_float8_e4m3fn_t *p0 = ra + (i+0)*sMx;
    const mag_float8_e4m3fn_t *p1 = ra + (i+1)*sMx;
    const mag_float8_e4m3fn_t *p2 = ra + (i+2)*sMx;
    const mag_float8_e4m3fn_t *p3 = ra + (i+3)*sMx;
    const mag_float8_e4m3fn_t *p4 = ra + (i+4)*sMx;
    const mag_float8_e4m3fn_t *p5 = ra + (i+5)*sMx;
    const mag_float8_e4m3fn_t *p6 = ra + (i+6)*sMx;
    const mag_float8_e4m3fn_t *p7 = ra + (i+7)*sMx;
    mag_float8_e4m3fn_t *dst = pa + i*kc;

    for (int64_t k=0; k < kc; ++k) {
      if ((k & (MAG_PF_GROUP - 1)) == 0) {
        mag_simd_prefetch_t0(p0 + (int64_t)MAG_PFDIST_A_L1*sKx);
        mag_simd_prefetch_t0(p4 + (int64_t)MAG_PFDIST_A_L1*sKx);
        mag_simd_prefetch_t1(p0 + (int64_t)MAG_PFDIST_A_L2*sKx);
        mag_simd_prefetch_t1(p4 + (int64_t)MAG_PFDIST_A_L2*sKx);
      }

      dst[k*8 + 0] = p0[k*sKx];
      dst[k*8 + 1] = p1[k*sKx];
      dst[k*8 + 2] = p2[k*sKx];
      dst[k*8 + 3] = p3[k*sKx];
      dst[k*8 + 4] = p4[k*sKx];
      dst[k*8 + 5] = p5[k*sKx];
      dst[k*8 + 6] = p6[k*sKx];
      dst[k*8 + 7] = p7[k*sKx];
    }
  }

  for (int64_t i=m8; i < mr; ++i) {
    const mag_float8_e4m3fn_t *src = ra + i*sMx;
    mag_float8_e4m3fn_t *dst = pa + i*kc;
    for (int64_t k=0; k < kc; ++k)
      dst[k] = src[k*sKx];
  }
}

static MAG_AINLINE void mag_gemv_float8_e4m3fn(
  int64_t K, int64_t N,
  const mag_float8_e4m3fn_t *restrict A,
  const mag_float8_e4m3fn_t *restrict B,
  int64_t ldb,
  mag_float8_e4m3fn_t *restrict C
) {
  int64_t j = 0;

  for (; j + 4*MAG_L <= N; j += 4*MAG_L) {
    mag_vf32_t s0 = mag_vf32_zero(), s1 = mag_vf32_zero();
    mag_vf32_t s2 = mag_vf32_zero(), s3 = mag_vf32_zero();
    const mag_float8_e4m3fn_t *brow = B + j;

    for (int64_t k=0; k < K; ++k, brow += ldb) {
      mag_vf32_t a = mag_vf32_splat(mag_float8_e4m3fn_to_float32(A[k]));
      s0 = mag_vf32_fmadd(a, mag_vf32_loadu_float8_e4m3fn(brow + 0*MAG_L), s0);
      s1 = mag_vf32_fmadd(a, mag_vf32_loadu_float8_e4m3fn(brow + 1*MAG_L), s1);
      s2 = mag_vf32_fmadd(a, mag_vf32_loadu_float8_e4m3fn(brow + 2*MAG_L), s2);
      s3 = mag_vf32_fmadd(a, mag_vf32_loadu_float8_e4m3fn(brow + 3*MAG_L), s3);
    }

    mag_vf32_storeu_float8_e4m3fn(C + j + 0*MAG_L, s0);
    mag_vf32_storeu_float8_e4m3fn(C + j + 1*MAG_L, s1);
    mag_vf32_storeu_float8_e4m3fn(C + j + 2*MAG_L, s2);
    mag_vf32_storeu_float8_e4m3fn(C + j + 3*MAG_L, s3);
  }

  for (; j + MAG_L <= N; j += MAG_L) {
    mag_vf32_t s = mag_vf32_zero();
    const mag_float8_e4m3fn_t *brow = B + j;

    for (int64_t k=0; k < K; ++k, brow += ldb) {
      mag_vf32_t a = mag_vf32_splat(mag_float8_e4m3fn_to_float32(A[k]));
      s = mag_vf32_fmadd(a, mag_vf32_loadu_float8_e4m3fn(brow), s);
    }

    mag_vf32_storeu_float8_e4m3fn(C + j, s);
  }

  for (; j < N; ++j) {
    float sum = 0.0f;
    for (int64_t k=0; k < K; ++k)
      sum += mag_float8_e4m3fn_to_float32(A[k]) * mag_float8_e4m3fn_to_float32(B[k*ldb + j]);
    C[j] = mag_float32_to_float8_e4m3fn(sum);
  }
}

static MAG_HOTPROC void mag_mm_block_float8_e4m3fn(
  int64_t kc, int64_t mr, int64_t nr,
  const mag_float8_e4m3fn_t *A, int64_t lda,
  const mag_float8_e4m3fn_t *B, int64_t ldb,
  mag_float8_e4m3fn_t *C, int64_t ldc,
  bool acc
) {
  int64_t j = 0;

  for (; nr-j >= 32; j += 32) {
    int64_t i = 0;
    for (; mr-i >= 16; i += 16) mag_mm_tile_16x32_float8_e4m3fn(kc, A + i*lda, lda, B + j, ldb, C + i*ldc + j, ldc, acc);
    for (; mr-i >=  8; i +=  8) mag_mm_tile_8x32_float8_e4m3fn (kc, A + i*lda, lda, B + j, ldb, C + i*ldc + j, ldc, acc);
    for (; i < mr; ++i)         mag_mm_tile_1x32_float8_e4m3fn (kc, A + i*lda,      B + j, ldb, C + i*ldc + j, acc);
  }

  for (; nr-j >= 16; j += 16) {
    int64_t i = 0;
    for (; mr-i >= 8; i += 8) mag_mm_tile_8x16_float8_e4m3fn(kc, A + i*lda, lda, B + j, ldb, C + i*ldc + j, ldc, acc);
    for (; i < mr; ++i)       mag_mm_tile_1x16_float8_e4m3fn(kc, A + i*lda,      B + j, ldb, C + i*ldc + j, acc);
  }

  for (; nr-j >= 8; j += 8) {
    int64_t i = 0;
    for (; mr-i >= 8; i += 8) mag_mm_tile_8x8_float8_e4m3fn(kc, A + i*lda, lda, B + j, ldb, C + i*ldc + j, ldc, acc);
    for (; i < mr; ++i)       mag_mm_tile_1x8_float8_e4m3fn(kc, A + i*lda,      B + j, ldb, C + i*ldc + j, acc);
  }

  int64_t rem = nr - j;
  if (!rem) return;

  for (int64_t i2=0; i2 < mr; ++i2) {
    const mag_float8_e4m3fn_t *ap = A + i2*lda;
    mag_float8_e4m3fn_t *cp = C + i2*ldc + j;

    for (int64_t jj=0; jj < rem; ++jj) {
      float sum = acc ? mag_float8_e4m3fn_to_float32(cp[jj]) : 0.0f;
      for (int64_t k=0; k < kc; ++k)
        sum += mag_float8_e4m3fn_to_float32(ap[k]) * mag_float8_e4m3fn_to_float32(B[k*ldb + (j + jj)]);
      cp[jj] = mag_float32_to_float8_e4m3fn(sum);
    }
  }
}

MAG_HOTPROC static mag_status_t mag_matmul_float8_e4m3fn(mag_error_t *err, const mag_kernel_payload_t *payload) {
  (void)err;

  mag_tensor_t *r = mag_cmd_out(0);
  const mag_tensor_t *x = mag_cmd_in(0);
  const mag_tensor_t *y = mag_cmd_in(1);

  const mag_float8_e4m3fn_t *bx = (const mag_float8_e4m3fn_t *)mag_tensor_data_ptr(x);
  const mag_float8_e4m3fn_t *by = (const mag_float8_e4m3fn_t *)mag_tensor_data_ptr(y);
  mag_float8_e4m3fn_t *br = (mag_float8_e4m3fn_t *)mag_tensor_data_ptr_mut(r);

  int64_t MR = payload->mm_params.MR;
  int64_t MC = payload->mm_params.MC;
  int64_t KC = payload->mm_params.KC;
  int64_t NC = payload->mm_params.NC;
  int64_t NR = payload->mm_params.NR;

  int64_t M = x->coords.rank == 1 ? 1 : x->coords.shape[x->coords.rank-2];
  int64_t N = y->coords.rank == 1 ? 1 : y->coords.shape[y->coords.rank-1];
  int64_t K = x->coords.shape[x->coords.rank-1];

  int64_t bdr = r->coords.rank > 2 ? r->coords.rank - 2 : 0;
  int64_t batch_total = 1;
  for (int64_t d=0; d < bdr; ++d)
    batch_total *= r->coords.shape[d];

  if (M == 1 && K >= 128 && N >= 4096 && y->coords.rank == 2 && y->coords.strides[y->coords.rank-1] == 1) {
    int64_t nth = payload->thread_num;
    int64_t tid = payload->thread_idx;
    int64_t j_per_thread = (N + nth - 1) / nth;
    int64_t j0 = tid*j_per_thread;
    int64_t j1 = mag_xmin(N, j0 + j_per_thread);

    for (int64_t batch = 0; batch < batch_total; ++batch) {
      const mag_float8_e4m3fn_t *A = bx + mag_offset_rmn(x, batch, 0, 0);
      const mag_float8_e4m3fn_t *B = by + mag_offset_rmn(y, batch, 0, 0) + j0;
      mag_float8_e4m3fn_t *C = br + mag_offset_rmn(r, batch, 0, 0) + j0;
      mag_gemv_float8_e4m3fn(K, j1 - j0, A, B, N, C);
    }

    return MAG_STATUS_OK;
  }

  int64_t bdx = x->coords.rank > 2 ? x->coords.rank-2 : 0;
  int64_t bdy = y->coords.rank > 2 ? y->coords.rank-2 : 0;

  int64_t tic = (M+MC-1)/MC;
  int64_t tjc = (N+NC-1)/NC;
  int64_t tpb = tic*tjc;
  int64_t tt = batch_total*tpb;

  mag_scratch_arena_clear(&mag_tls_arena);

  mag_float8_e4m3fn_t *scratch =
    mag_scratch_arena_alloc(&mag_tls_arena, sizeof(*scratch)*(KC*NC + MC*KC));

  mag_float8_e4m3fn_t *Bp = scratch;
  mag_float8_e4m3fn_t *Ap = Bp + KC*NC;

  for (;;) {
    int64_t tile = mag_atomic64_fetch_add(payload->mm_next_tile, 1, MAG_MO_RELAXED);
    if (tile >= tt) break;

    int64_t batch_idx = tile / tpb;
    int64_t rem = tile % tpb;
    int64_t jc = rem % tjc;
    int64_t ic = rem / tjc;

    int64_t idx_r[MAG_MAX_DIMS] = {0};
    for (int64_t d=bdr-1, t=batch_idx; d >= 0; --d) {
      idx_r[d] = t % r->coords.shape[d];
      t /= r->coords.shape[d];
    }

    int64_t xb_flat = 0;
    for (int64_t d=0; d < bdx; ++d) {
      int64_t rd = bdr - bdx + d;
      xb_flat = xb_flat*x->coords.shape[d] + (x->coords.shape[d] == 1 ? 0 : idx_r[rd]);
    }

    int64_t yb_flat = 0;
    for (int64_t d=0; d < bdy; ++d) {
      int64_t rd = bdr - bdy + d;
      yb_flat = yb_flat*y->coords.shape[d] + (y->coords.shape[d] == 1 ? 0 : idx_r[rd]);
    }

    bool yv = y->coords.rank == 1;

    const mag_float8_e4m3fn_t *px_base = bx + mag_offset_rmn(x, xb_flat, 0, 0);
    const mag_float8_e4m3fn_t *py_base = by + mag_offset_rmn(y, yb_flat, 0, 0);
    mag_float8_e4m3fn_t *pr_base = br + mag_offset_rmn(r, batch_idx, 0, 0);

    int64_t i0 = ic*MC;
    int64_t mc = i0+MC <= M ? MC : M-i0;

    int64_t j0 = jc*NC;
    int64_t nc = j0+NC <= N ? NC : N-j0;

    int64_t sMx = x->coords.strides[x->coords.rank-2];
    int64_t sKx = x->coords.strides[x->coords.rank-1];
    int64_t sKy = yv ? 0 : y->coords.strides[y->coords.rank-2];
    int64_t sNy = yv ? 0 : y->coords.strides[y->coords.rank-1];

    for (int64_t pc = 0; pc < K; pc += KC) {
      int64_t kc = mag_xmin(KC, K - pc);

      if (y->coords.rank == 1)
        mag_mm_pack_B_vec_float8_e4m3fn(kc, nc, py_base + pc, Bp);
      else
        mag_mm_pack_B_kc_nc_float8_e4m3fn(kc, nc, py_base + pc*sKy + j0*sNy, sKy, sNy, Bp);

      mag_mm_pack_A_mc_kc_panel8_float8_e4m3fn(kc, mc, px_base + i0*sMx + pc*sKx, sMx, sKx, Ap);

      for (int64_t ir=0; ir < mc; ir += MR) {
        for (int64_t jr=0; jr < nc; jr += NR) {
          mag_mm_block_float8_e4m3fn(
            kc,
            mag_xmin(MR, mc - ir),
            mag_xmin(NR, nc - jr),
            Ap + ir*kc,
            kc,
            Bp + jr,
            nc,
            pr_base + (i0 + ir)*N + (j0 + jr),
            N,
            pc
          );
        }
      }
    }
  }

  mag_scratch_arena_clear(&mag_tls_arena);
  return MAG_STATUS_OK;
}

static MAG_AINLINE float mag_load_x_f16_as_f32(
  const mag_tensor_t *x, const mag_float16_t *bx,
  int64_t xb_flat, int64_t i, int64_t k
) {
  if (x->coords.rank == 1) {
    int64_t off = mag_offset_rmn(x, k, 0, 0);
    return mag_float16_to_float32(bx[off]);
  } else {
    int64_t off = mag_offset_rmn(x, xb_flat, i, k);
    return mag_float16_to_float32(bx[off]);
  }
}

static MAG_AINLINE float mag_load_y_f16_as_f32(
  const mag_tensor_t *y, const mag_float16_t *by,
  int64_t yb_flat, int64_t k, int64_t n
) {
  if (y->coords.rank == 1) {
    int64_t off = mag_offset_rmn(y, k, 0, 0);
    return mag_float16_to_float32(by[off]);
  } else {
    int64_t off = mag_offset_rmn(y, yb_flat, k, n);
    return mag_float16_to_float32(by[off]);
  }
}

static MAG_AINLINE void mag_store_r_f16_from_f32(
  mag_tensor_t *r, mag_float16_t *br,
  int64_t rb_flat, int64_t i, int64_t n, float v
) {
  if (r->coords.rank == 0) {
    br[0] = mag_float32_to_float16(v);
  } else if (r->coords.rank == 1) {
    int64_t off = mag_offset_rmn(r, n, 0, 0);
    br[off] = mag_float32_to_float16(v);
  } else {
    int64_t off = mag_offset_rmn(r, rb_flat, i, n);
    br[off] = mag_float32_to_float16(v);
  }
}

static MAG_HOTPROC mag_status_t mag_matmul_float16(mag_error_t *err,const mag_kernel_payload_t *payload) {
  (void)err;
  mag_tensor_t *r = mag_cmd_out(0);
  const mag_tensor_t *x = mag_cmd_in(0);
  const mag_tensor_t *y = mag_cmd_in(1);
  mag_float16_t *br = (mag_float16_t *)mag_tensor_data_ptr_mut(r);
  const mag_float16_t *bx = (const mag_float16_t *)mag_tensor_data_ptr(x);
  const mag_float16_t *by = (const mag_float16_t *)mag_tensor_data_ptr(y);
  int64_t M = x->coords.rank == 1 ? 1 : x->coords.shape[x->coords.rank-2];
  int64_t N = y->coords.rank == 1 ? 1 : y->coords.shape[y->coords.rank-1];
  int64_t K = x->coords.shape[x->coords.rank - 1];
  int64_t bdr = r->coords.rank > 2 ? r->coords.rank-2 : 0;
  int64_t batch_total = 1;
  for (int64_t d = 0; d < bdr; ++d) batch_total *= r->coords.shape[d];
  int64_t bdx = x->coords.rank > 2 ? x->coords.rank-2 : 0;
  int64_t bdy = y->coords.rank > 2 ? y->coords.rank-2 : 0;
  int64_t tid = payload->thread_idx;
  int64_t nth = payload->thread_num;
  int64_t work_total = batch_total * M;
  for (int64_t work = tid; work < work_total; work += nth) {
    int64_t b = work / M;
    int64_t i = work - b * M;
    int64_t idx_r[MAG_MAX_DIMS] = {0};
    {
      int64_t rem = b;
      for (int64_t d = bdr - 1; d >= 0; --d) {
        idx_r[d] = rem % r->coords.shape[d];
        rem /= r->coords.shape[d];
      }
    }
    int64_t xb_flat = 0;
    for (int64_t d = 0; d < bdx; ++d) {
      int64_t rd  = bdr - bdx + d;
      int64_t idx = (x->coords.shape[d] == 1) ? 0 : idx_r[rd];
      xb_flat = xb_flat * x->coords.shape[d] + idx;
    }
    int64_t yb_flat = 0;
    for (int64_t d = 0; d < bdy; ++d) {
      int64_t rd  = bdr - bdy + d;
      int64_t idx = (y->coords.shape[d] == 1) ? 0 : idx_r[rd];
      yb_flat = yb_flat * y->coords.shape[d] + idx;
    }
    int64_t rb_flat = 0;
    for (int64_t d = 0; d < bdr; ++d)
      rb_flat = rb_flat * r->coords.shape[d] + idx_r[d];
    for (int64_t n = 0; n < N; ++n) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        const float ax = mag_load_x_f16_as_f32(x, bx, xb_flat, i, k);
        const float byv = mag_load_y_f16_as_f32(y, by, yb_flat, k, n);
        sum += ax * byv;
      }
      mag_store_r_f16_from_f32(r, br, rb_flat, i, n, sum);
    }
  }
  return MAG_STATUS_OK;
}

static MAG_AINLINE float mag_f32_id_in(float x) { return x; }
static MAG_AINLINE float mag_f32_id_out(float x) { return x; }

static MAG_AINLINE float mag_vf32_hsum(mag_vf32_t v) {
  mag_alignas(64) float tmp[MAG_L];
  mag_vf32_storeu(tmp, v);
  float s = 0.0f;
  for (int i=0; i < MAG_L; ++i) s += tmp[i];
  return s;
}

static MAG_AINLINE void mag_mm_pack_B_kc_nc_fp8w_to_bfloat16(
  int64_t kc, int64_t nc,
  const mag_float8_e4m3fn_t *restrict Bsrc, ptrdiff_t strideK, ptrdiff_t strideN,
  float scale,
  mag_bfloat16_t *restrict Bp
) {
  mag_vf32_t svec = mag_vf32_splat(scale);
  if (strideN == 1) { /* Each source row is nc consecutive FP8: SIMD-load, scale, store as BF16. */
    for (int64_t k=0; k < kc; ++k) {
      if (k + 1 < kc) mag_simd_prefetch_t0((const char *)(Bsrc + (k + 1)*strideK));
      const mag_float8_e4m3fn_t *src = Bsrc + k*strideK;
      mag_bfloat16_t *dst = Bp + k*nc;
      int64_t j = 0;
      for (; j + MAG_L <= nc; j += MAG_L) {
        mag_vf32_t v = mag_vf32_mul(mag_vf32_loadu_float8_e4m3fn(src + j), svec);
        mag_vf32_storeu_bf16(dst + j, v);
      }
      for (; j < nc; ++j) dst[j] = mag_float32_to_bfloat16(mag_float8_e4m3fn_to_float32(src[j])*scale);
    }
    return;
  }
  if (strideK == 1) {
    enum { NT = 8 };
    mag_alignas(64) float stage[NT][MAG_L];
    int64_t nb = 0;
    for (; nb + NT <= nc; nb += NT) {
      int64_t k = 0;
      for (; k + MAG_L <= kc; k += MAG_L) {
        for (int jj=0; jj < NT; ++jj) {
          const mag_float8_e4m3fn_t *src = Bsrc + (nb + jj)*strideN + k;
          mag_vf32_t v = mag_vf32_mul(mag_vf32_loadu_float8_e4m3fn(src), svec);
          mag_vf32_storeu(stage[jj], v);
        }
        for (int l=0; l < MAG_L; ++l) {
          mag_bfloat16_t *dst = Bp + (k+l)*nc + nb;
          for (int jj=0; jj < NT; ++jj)
            dst[jj] = mag_float32_to_bfloat16(stage[jj][l]);
        }
      }
      for (; k < kc; ++k) {
        for (int jj=0; jj < NT; ++jj) {
          float v = mag_float8_e4m3fn_to_float32(Bsrc[(nb + jj)*strideN + k])*scale;
          Bp[k*nc + nb + jj] = mag_float32_to_bfloat16(v);
        }
      }
    }
    for (; nb < nc; ++nb) {
      const mag_float8_e4m3fn_t *src = Bsrc + nb*strideN;
      int64_t k = 0;
      for (; k + MAG_L <= kc; k += MAG_L) {
        mag_vf32_t v = mag_vf32_mul(mag_vf32_loadu_float8_e4m3fn(src + k), svec);
        mag_alignas(64) float tmp[MAG_L];
        mag_vf32_storeu(tmp, v);
        for (int l=0; l < MAG_L; ++l) Bp[(k+l)*nc + nb] = mag_float32_to_bfloat16(tmp[l]);
      }
      for (; k < kc; ++k) Bp[k*nc + nb] = mag_float32_to_bfloat16(mag_float8_e4m3fn_to_float32(src[k])*scale);
    }
    return;
  }
  for (int64_t k=0; k < kc; ++k) {
    for (int64_t j=0; j < nc; ++j) {
      float v = mag_float8_e4m3fn_to_float32(Bsrc[k*strideK + j*strideN])*scale;
      Bp[k*nc + j] = mag_float32_to_bfloat16(v);
    }
  }
}

static MAG_AINLINE void mag_mm_pack_B_vec_fp8w_to_bfloat16(
  int64_t kc, int64_t nc,
  const mag_float8_e4m3fn_t *restrict yvec, float scale,
  mag_bfloat16_t *restrict Bp
) {
  for (int64_t k=0; k < kc; ++k) {
    mag_bfloat16_t v = mag_float32_to_bfloat16(mag_float8_e4m3fn_to_float32(yvec[k])*scale);
    mag_bfloat16_t *dst = Bp + k*nc;
    for (int64_t j=0; j < nc; ++j) dst[j] = v;
  }
}

#define mag_gen_gemv_fp8w(TX, name_suffix, x_to_f32, f32_to_x, vf32_loadu_x)                             \
static MAG_AINLINE void mag_gemv_fp8w_##name_suffix(                                                     \
  int64_t K,                                                                                             \
  const TX *restrict A,                                                                                  \
  const mag_float8_e4m3fn_t *restrict B,                                                                 \
  int64_t sKy, int64_t sNy,                                                                              \
  TX *restrict C,                                                                                        \
  int64_t sNc,                                                                                           \
  int64_t n0, int64_t n1,                                                                                \
  float scale                                                                                            \
) {                                                                                                      \
  if (sKy == 1) { \
    for (int64_t n=n0; n < n1; ++n) {                                                                    \
      const mag_float8_e4m3fn_t *brow = B + n*sNy;                                                       \
      int64_t k = 0;                                                                                     \
      mag_vf32_t acc = mag_vf32_zero();                                                                  \
      for (; k + MAG_L <= K; k += MAG_L) {                                                               \
        mag_vf32_t a = vf32_loadu_x(A + k);                                                              \
        mag_vf32_t b = mag_vf32_loadu_float8_e4m3fn(brow + k);                                           \
        acc = mag_vf32_fmadd(a, b, acc);                                                                 \
      }                                                                                                  \
      float sum = mag_vf32_hsum(acc);                                                                    \
      for (; k < K; ++k) sum += x_to_f32(A[k]) * mag_float8_e4m3fn_to_float32(brow[k]);                  \
      C[n*sNc] = f32_to_x(sum*scale);                                                                    \
    }                                                                                                    \
    return;                                                                                              \
  }                                                                                                      \
  for (int64_t n=n0; n < n1; ++n) {                                                                      \
    float sum = 0.0f;                                                                                    \
    for (int64_t k=0; k < K; ++k)                                                                        \
      sum += x_to_f32(A[k]) * mag_float8_e4m3fn_to_float32(B[k*sKy + n*sNy]);                            \
    C[n*sNc] = f32_to_x(sum*scale);                                                                      \
  }                                                                                                      \
}

mag_gen_gemv_fp8w(mag_bfloat16_t, bfloat16, mag_bfloat16_to_float32, mag_float32_to_bfloat16, mag_vf32_loadu_bf16)
mag_gen_gemv_fp8w(mag_float16_t,  float16,  mag_float16_to_float32,  mag_float32_to_float16,  mag_vf32_loadu_f16)
mag_gen_gemv_fp8w(float,          float32,  mag_f32_id_in,           mag_f32_id_out,          mag_vf32_loadu_f32)

#undef mag_gen_gemv_fp8w

MAG_HOTPROC static mag_status_t mag_matmul_fp8w_bfloat16(mag_error_t *err, const mag_kernel_payload_t *payload) {
  (void)err;
  mag_tensor_t *r = mag_cmd_out(0);
  const mag_tensor_t *x = mag_cmd_in(0);
  const mag_tensor_t *w = mag_cmd_in(1);
  const mag_tensor_t *s = mag_cmd_in(2);
  const mag_bfloat16_t *bx = (const mag_bfloat16_t *)mag_tensor_data_ptr(x);
  const mag_float8_e4m3fn_t *bw = (const mag_float8_e4m3fn_t *)mag_tensor_data_ptr(w);
  mag_bfloat16_t *br = (mag_bfloat16_t *)mag_tensor_data_ptr_mut(r);
  float scale = ((const float *)mag_tensor_data_ptr(s))[0];
  int64_t MR = payload->mm_params.MR;
  int64_t MC = payload->mm_params.MC;
  int64_t KC = payload->mm_params.KC;
  int64_t NC = payload->mm_params.NC;
  int64_t NR = payload->mm_params.NR;
  int64_t M = x->coords.rank == 1 ? 1 : x->coords.shape[x->coords.rank-2];
  int64_t N = w->coords.rank == 1 ? 1 : w->coords.shape[w->coords.rank-1];
  int64_t K = x->coords.shape[x->coords.rank-1];
  int64_t bdr = r->coords.rank > 2 ? r->coords.rank-2 : 0;
  int64_t batch_total = 1;
  for (int64_t d=0; d < bdr; ++d) batch_total *= r->coords.shape[d];
  int64_t sKx = x->coords.strides[x->coords.rank-1];
  bool yv = w->coords.rank == 1;
  int64_t sKy = yv ? w->coords.strides[0] : w->coords.strides[w->coords.rank-2];
  int64_t sNy = yv ? 0 : w->coords.strides[w->coords.rank-1];
  int64_t sNr = r->coords.rank == 0 ? 0 : r->coords.strides[r->coords.rank-1];
  if (M == 1 && sKx == 1 && w->coords.rank == 2 && sNr == 1 && N > 0) {
    int64_t nth = payload->thread_num;
    int64_t tid = payload->thread_idx;
    int64_t n_per_thread = (N + nth - 1) / nth;
    int64_t n0 = tid*n_per_thread;
    int64_t n1 = mag_xmin(N, n0 + n_per_thread);
    if (n0 >= n1) return MAG_STATUS_OK;
    for (int64_t batch=0; batch < batch_total; ++batch) {
      const mag_bfloat16_t *A = bx + mag_offset_rmn(x, batch, 0, 0);
      const mag_float8_e4m3fn_t *B = bw + mag_offset_rmn(w, batch, 0, 0);
      mag_bfloat16_t *C = br + mag_offset_rmn(r, batch, 0, 0);
      mag_gemv_fp8w_bfloat16(K, A, B, sKy, sNy, C, sNr, n0, n1, scale);
    }
    return MAG_STATUS_OK;
  }
  int64_t bdx = x->coords.rank > 2 ? x->coords.rank-2 : 0;
  int64_t bdy = w->coords.rank > 2 ? w->coords.rank-2 : 0;
  int64_t tic = (M+MC-1)/MC;
  int64_t tjc = (N+NC-1)/NC;
  int64_t tpb = tic*tjc;
  int64_t tt = batch_total*tpb;
  mag_scratch_arena_clear(&mag_tls_arena);
  mag_bfloat16_t *scratch = mag_scratch_arena_alloc(&mag_tls_arena, sizeof(*scratch)*(KC*NC + MC*KC));
  mag_bfloat16_t *Bp = scratch;
  mag_bfloat16_t *Ap = Bp + KC*NC;
  for (;;) {
    int64_t tile = mag_atomic64_fetch_add(payload->mm_next_tile, 1, MAG_MO_RELAXED);
    if (tile >= tt) break;
    int64_t batch_idx = tile / tpb;
    int64_t rem = tile % tpb;
    int64_t jc = rem % tjc;
    int64_t ic = rem / tjc;
    int64_t idx_r[MAG_MAX_DIMS] = {0};
    for (int64_t d=bdr-1, t=batch_idx; d >= 0; --d) {
      idx_r[d] = t % r->coords.shape[d];
      t /= r->coords.shape[d];
    }
    int64_t xb_flat = 0;
    for (int64_t d=0; d < bdx; ++d) {
      int64_t rd = bdr - bdx + d;
      xb_flat = xb_flat*x->coords.shape[d] + (x->coords.shape[d] == 1 ? 0 : idx_r[rd]);
    }
    int64_t yb_flat = 0;
    for (int64_t d=0; d < bdy; ++d) {
      int64_t rd = bdr - bdy + d;
      yb_flat = yb_flat*w->coords.shape[d] + (w->coords.shape[d] == 1 ? 0 : idx_r[rd]);
    }
    const mag_bfloat16_t *px_base = bx + mag_offset_rmn(x, xb_flat, 0, 0);
    const mag_float8_e4m3fn_t *py_base = bw + mag_offset_rmn(w, yb_flat, 0, 0);
    mag_bfloat16_t *pr_base = br + mag_offset_rmn(r, batch_idx, 0, 0);
    int64_t i0 = ic*MC;
    int64_t mc = i0+MC <= M ? MC : M-i0;
    int64_t j0 = jc*NC;
    int64_t nc = j0+NC <= N ? NC : N-j0;
    int64_t sMx = x->coords.strides[x->coords.rank-2];
    for (int64_t pc = 0; pc < K; pc += KC) {
      int64_t kc = mag_xmin(KC, K - pc);
      if (yv) mag_mm_pack_B_vec_fp8w_to_bfloat16(kc, nc, py_base + pc, scale, Bp);
      else    mag_mm_pack_B_kc_nc_fp8w_to_bfloat16(kc, nc, py_base + pc*sKy + j0*sNy, sKy, sNy, scale, Bp);
      mag_mm_pack_A_mc_kc_panel8_bfloat16(kc, mc, px_base + i0*sMx + pc*sKx, sMx, sKx, Ap);
      for (int64_t ir=0; ir < mc; ir += MR)
        for (int64_t jr=0; jr < nc; jr += NR)
          mag_mm_block_bfloat16(
            kc,
            mag_xmin(MR, mc - ir),
            mag_xmin(NR, nc - jr),
            Ap + ir*kc,
            kc,
            Bp + jr,
            nc,
            pr_base + (i0 + ir)*N + (j0 + jr),
            N,
            pc);
    }
  }
  mag_scratch_arena_clear(&mag_tls_arena);
  return MAG_STATUS_OK;
}

/*
** Simple scalar fallback for non-BF16 activations (float16, float32). These code paths
** are not exercised by the Qwen3 example (which uses BF16 activations), so they keep
** the slow but correct path until someone needs them.
*/
#define mag_gen_matmul_fp8w_kernel_scalar(TX, name_suffix, x_to_f32, f32_to_x)                           \
MAG_HOTPROC static mag_status_t mag_matmul_fp8w_##name_suffix(mag_error_t *err, const mag_kernel_payload_t *payload) { \
  (void)err;                                                                                             \
  mag_tensor_t *r = mag_cmd_out(0);                                                                      \
  const mag_tensor_t *x = mag_cmd_in(0);                                                                 \
  const mag_tensor_t *w = mag_cmd_in(1);                                                                 \
  const mag_tensor_t *s = mag_cmd_in(2);                                                                 \
  const TX *bx = (const TX *)mag_tensor_data_ptr(x);                                                     \
  const mag_float8_e4m3fn_t *bw = (const mag_float8_e4m3fn_t *)mag_tensor_data_ptr(w);                   \
  TX *br = (TX *)mag_tensor_data_ptr_mut(r);                                                             \
  float scale = ((const float *)mag_tensor_data_ptr(s))[0];                                              \
  int64_t M = x->coords.rank == 1 ? 1 : x->coords.shape[x->coords.rank-2];                               \
  int64_t N = w->coords.rank == 1 ? 1 : w->coords.shape[w->coords.rank-1];                               \
  int64_t K = x->coords.shape[x->coords.rank-1];                                                         \
  int64_t bdr = r->coords.rank > 2 ? r->coords.rank-2 : 0;                                               \
  int64_t batch_total = 1;                                                                               \
  for (int64_t d=0; d < bdr; ++d) batch_total *= r->coords.shape[d];                                     \
  int64_t bdx = x->coords.rank > 2 ? x->coords.rank-2 : 0;                                               \
  int64_t bdy = w->coords.rank > 2 ? w->coords.rank-2 : 0;                                               \
  int64_t sKx = x->coords.strides[x->coords.rank-1];                                                     \
  bool yv = w->coords.rank == 1;                                                                         \
  int64_t sKy = yv ? w->coords.strides[0] : w->coords.strides[w->coords.rank-2];                         \
  int64_t sNy = yv ? 0 : w->coords.strides[w->coords.rank-1];                                            \
  int64_t sNr = r->coords.rank == 0 ? 0 : r->coords.strides[r->coords.rank-1];                           \
  int64_t tid = payload->thread_idx;                                                                     \
  int64_t nth = payload->thread_num;                                                                     \
  int64_t work_total = batch_total*M;                                                                    \
  for (int64_t work=tid; work < work_total; work += nth) {                                               \
    int64_t b = work / M;                                                                                \
    int64_t i = work - b*M;                                                                              \
    int64_t idx_r[MAG_MAX_DIMS] = {0};                                                                   \
    {                                                                                                    \
      int64_t rem = b;                                                                                   \
      for (int64_t d=bdr-1; d >= 0; --d) {                                                               \
        idx_r[d] = rem % r->coords.shape[d];                                                             \
        rem /= r->coords.shape[d];                                                                       \
      }                                                                                                  \
    }                                                                                                    \
    int64_t xb_flat = 0;                                                                                 \
    for (int64_t d=0; d < bdx; ++d) {                                                                    \
      int64_t rd = bdr - bdx + d;                                                                        \
      int64_t idx = x->coords.shape[d] == 1 ? 0 : idx_r[rd];                                             \
      xb_flat = xb_flat*x->coords.shape[d] + idx;                                                        \
    }                                                                                                    \
    int64_t yb_flat = 0;                                                                                 \
    for (int64_t d=0; d < bdy; ++d) {                                                                    \
      int64_t rd = bdr - bdy + d;                                                                        \
      int64_t idx = w->coords.shape[d] == 1 ? 0 : idx_r[rd];                                             \
      yb_flat = yb_flat*w->coords.shape[d] + idx;                                                        \
    }                                                                                                    \
    const TX *A = bx + mag_offset_rmn(x, xb_flat, i, 0);                                                 \
    const mag_float8_e4m3fn_t *B = bw + mag_offset_rmn(w, yb_flat, 0, 0);                                \
    TX *C = br + mag_offset_rmn(r, b, i, 0);                                                             \
    for (int64_t n=0; n < N; ++n) {                                                                      \
      float sum = 0.0f;                                                                                  \
      for (int64_t k=0; k < K; ++k)                                                                      \
        sum += x_to_f32(A[k*sKx]) * mag_float8_e4m3fn_to_float32(B[k*sKy + n*sNy]);                      \
      C[n*sNr] = f32_to_x(sum*scale);                                                                    \
    }                                                                                                    \
  }                                                                                                      \
  return MAG_STATUS_OK;                                                                                  \
}

mag_gen_matmul_fp8w_kernel_scalar(mag_float16_t, float16, mag_float16_to_float32, mag_float32_to_float16)
mag_gen_matmul_fp8w_kernel_scalar(float,         float32, mag_f32_id_in,          mag_f32_id_out)

#undef mag_gen_matmul_fp8w_kernel_scalar
