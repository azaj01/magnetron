/*
** +---------------------------------------------------------------------+
** | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
** | Licensed under the Apache License, Version 2.0                      |
** |                                                                     |
** | Website : https://mariosieg.com                                     |
** | GitHub : https://github.com/MarioSieg                              |
** | License : https://www.apache.org/licenses/LICENSE-2.0               |
** +---------------------------------------------------------------------+
*/

#define mag_gen_stub_clone(T, TF) \
  static MAG_HOTPROC void mag_clone_##TF(const mag_kernel_payload_t *payload) { \
    mag_tensor_t *r = mag_cmd_out(0); \
    const mag_tensor_t *x = mag_cmd_in(0); \
    T *br = (T *)mag_tensor_data_ptr_mut(r); \
    const T *bx = (const T *)mag_tensor_data_ptr(x); \
    int64_t total = r->numel; \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t chunk = (total+tc-1)/tc; \
    int64_t ra = ti*chunk; \
    int64_t rb = mag_xmin(ra+chunk, total); \
    if (mag_unlikely(rb <= ra)) return; \
    if (mag_all_shapes_equal_and_contig((const mag_tensor_t *[2]){r, x}, 2)) { \
      memcpy(br+ra, bx+ra, (rb-ra)*sizeof(T)); \
      return; \
    } \
    mag_coords_iter_t cr, cx; \
    mag_coords_iter_init(&cr, &r->coords); \
    mag_coords_iter_init(&cx, &x->coords); \
    for (int64_t i=ra; i < rb; ++i) { \
      int64_t ri, xi; \
      mag_coords_iter_offset2(&cr, &cx, i, &ri, &xi); \
      mag_bnd_chk(bx+xi, bx, mag_tensor_numbytes(x)); \
      mag_bnd_chk(br+ri, br, mag_tensor_numbytes(r)); \
      br[ri] = bx[xi]; \
    } \
  }

mag_gen_stub_clone(float, float32)
mag_gen_stub_clone(mag_float16_t, float16)
mag_gen_stub_clone(mag_bfloat16_t, bfloat16)
mag_gen_stub_clone(uint8_t, uint8)
mag_gen_stub_clone(int8_t, int8)
mag_gen_stub_clone(uint16_t, uint16)
mag_gen_stub_clone(int16_t, int16)
mag_gen_stub_clone(uint32_t, uint32)
mag_gen_stub_clone(int32_t, int32)
mag_gen_stub_clone(uint64_t, uint64)
mag_gen_stub_clone(int64_t, int64)

#undef mag_gen_stub_clone

#define mag_gen_stub_unary(T, TF, FUNC) \
  static void MAG_HOTPROC mag_##FUNC##_##TF(const mag_kernel_payload_t *payload) { \
    mag_tensor_t *r = mag_cmd_out(0); \
    const mag_tensor_t *x = mag_cmd_in(0); \
    T *br = (T *)mag_tensor_data_ptr_mut(r); \
    const T *bx = (const T *)mag_tensor_data_ptr(x); \
    int64_t total = r->numel; \
    int64_t tc = payload->thread_num; \
    int64_t ti = payload->thread_idx; \
    int64_t chunk = (total + tc - 1)/tc; \
    int64_t ra = ti*chunk; \
    int64_t rb = mag_xmin(ra + chunk, total); \
    if (mag_all_shapes_equal_and_contig((const mag_tensor_t *[2]){r, x}, 2)) { \
      mag_v##FUNC##_##TF(rb-ra, br+ra, bx+ra); \
      return; \
    } \
    mag_coords_iter_t cr, cx; \
    mag_coords_iter_init(&cr, &r->coords); \
    mag_coords_iter_init(&cx, &x->coords); \
    for (int64_t i=ra; i < rb; ++i) { \
      int64_t ri, xi; \
      mag_coords_iter_offset2(&cr, &cx, i, &ri, &xi); \
      mag_bnd_chk(bx+xi, bx, mag_tensor_numbytes(x)); \
      mag_bnd_chk(br+ri, br, mag_tensor_numbytes(r)); \
      mag_v##FUNC##_##TF(1, br+ri, bx+xi); \
    } \
  }

mag_gen_stub_unary(float, float32, abs)
mag_gen_stub_unary(mag_float16_t, float16, abs)
mag_gen_stub_unary(mag_bfloat16_t, bfloat16, abs)
mag_gen_stub_unary(uint8_t, uint8, abs)
mag_gen_stub_unary(int8_t, int8, abs)
mag_gen_stub_unary(uint16_t, uint16, abs)
mag_gen_stub_unary(int16_t, int16, abs)
mag_gen_stub_unary(uint32_t, uint32, abs)
mag_gen_stub_unary(int32_t, int32, abs)
mag_gen_stub_unary(uint64_t, uint64, abs)
mag_gen_stub_unary(int64_t, int64, abs)

mag_gen_stub_unary(float, float32, sgn)
mag_gen_stub_unary(mag_float16_t, float16, sgn)
mag_gen_stub_unary(mag_bfloat16_t, bfloat16, sgn)
mag_gen_stub_unary(uint8_t, uint8, sgn)
mag_gen_stub_unary(int8_t, int8, sgn)
mag_gen_stub_unary(uint16_t, uint16, sgn)
mag_gen_stub_unary(int16_t, int16, sgn)
mag_gen_stub_unary(uint32_t, uint32, sgn)
mag_gen_stub_unary(int32_t, int32, sgn)
mag_gen_stub_unary(uint64_t, uint64, sgn)
mag_gen_stub_unary(int64_t, int64, sgn)

mag_gen_stub_unary(float, float32, neg)
mag_gen_stub_unary(mag_float16_t, float16, neg)
mag_gen_stub_unary(mag_bfloat16_t, bfloat16, neg)
mag_gen_stub_unary(uint8_t, uint8, neg)
mag_gen_stub_unary(int8_t, int8, neg)
mag_gen_stub_unary(uint16_t, uint16, neg)
mag_gen_stub_unary(int16_t, int16, neg)
mag_gen_stub_unary(uint32_t, uint32, neg)
mag_gen_stub_unary(int32_t, int32, neg)
mag_gen_stub_unary(uint64_t, uint64, neg)
mag_gen_stub_unary(int64_t, int64, neg)

mag_gen_stub_unary(float, float32, log)
mag_gen_stub_unary(mag_float16_t, float16, log)
mag_gen_stub_unary(mag_bfloat16_t, bfloat16, log)

mag_gen_stub_unary(float, float32, log10)
mag_gen_stub_unary(mag_float16_t, float16, log10)
mag_gen_stub_unary(mag_bfloat16_t, bfloat16, log10)

mag_gen_stub_unary(float, float32, log1p)
mag_gen_stub_unary(mag_float16_t, float16, log1p)
mag_gen_stub_unary(mag_bfloat16_t, bfloat16, log1p)

mag_gen_stub_unary(float, float32, log2)
mag_gen_stub_unary(mag_float16_t, float16, log2)
mag_gen_stub_unary(mag_bfloat16_t, bfloat16, log2)

mag_gen_stub_unary(float, float32, sqr)
mag_gen_stub_unary(mag_float16_t, float16, sqr)
mag_gen_stub_unary(mag_bfloat16_t, bfloat16, sqr)
mag_gen_stub_unary(uint8_t, uint8, sqr)
mag_gen_stub_unary(int8_t, int8, sqr)
mag_gen_stub_unary(uint16_t, uint16, sqr)
mag_gen_stub_unary(int16_t, int16, sqr)
mag_gen_stub_unary(uint32_t, uint32, sqr)
mag_gen_stub_unary(int32_t, int32, sqr)
mag_gen_stub_unary(uint64_t, uint64, sqr)
mag_gen_stub_unary(int64_t, int64, sqr)

mag_gen_stub_unary(float, float32, rcp)
mag_gen_stub_unary(mag_float16_t, float16, rcp)
mag_gen_stub_unary(mag_bfloat16_t, bfloat16, rcp)

mag_gen_stub_unary(float, float32, sqrt)
mag_gen_stub_unary(mag_float16_t, float16, sqrt)
mag_gen_stub_unary(mag_bfloat16_t, bfloat16, sqrt)

mag_gen_stub_unary(float, float32, rsqrt)
mag_gen_stub_unary(mag_float16_t, float16, rsqrt)
mag_gen_stub_unary(mag_bfloat16_t, bfloat16, rsqrt)

mag_gen_stub_unary(float, float32, sin)
mag_gen_stub_unary(mag_float16_t, float16, sin)
mag_gen_stub_unary(mag_bfloat16_t, bfloat16, sin)

mag_gen_stub_unary(float, float32, cos)
mag_gen_stub_unary(mag_float16_t, float16, cos)
mag_gen_stub_unary(mag_bfloat16_t, bfloat16, cos)

mag_gen_stub_unary(float, float32, tan)
mag_gen_stub_unary(mag_float16_t, float16, tan)
mag_gen_stub_unary(mag_bfloat16_t, bfloat16, tan)

mag_gen_stub_unary(float, float32, asin)
mag_gen_stub_unary(mag_float16_t, float16, asin)
mag_gen_stub_unary(mag_bfloat16_t, bfloat16, asin)

mag_gen_stub_unary(float, float32, acos)
mag_gen_stub_unary(mag_float16_t, float16, acos)
mag_gen_stub_unary(mag_bfloat16_t, bfloat16, acos)

mag_gen_stub_unary(float, float32, atan)
mag_gen_stub_unary(mag_float16_t, float16, atan)
mag_gen_stub_unary(mag_bfloat16_t, bfloat16, atan)

mag_gen_stub_unary(float, float32, sinh)
mag_gen_stub_unary(mag_float16_t, float16, sinh)
mag_gen_stub_unary(mag_bfloat16_t, bfloat16, sinh)

mag_gen_stub_unary(float, float32, cosh)
mag_gen_stub_unary(mag_float16_t, float16, cosh)
mag_gen_stub_unary(mag_bfloat16_t, bfloat16, cosh)

mag_gen_stub_unary(float, float32, tanh)
mag_gen_stub_unary(mag_float16_t, float16, tanh)
mag_gen_stub_unary(mag_bfloat16_t, bfloat16, tanh)

mag_gen_stub_unary(float, float32, asinh)
mag_gen_stub_unary(mag_float16_t, float16, asinh)
mag_gen_stub_unary(mag_bfloat16_t, bfloat16, asinh)

mag_gen_stub_unary(float, float32, acosh)
mag_gen_stub_unary(mag_float16_t, float16, acosh)
mag_gen_stub_unary(mag_bfloat16_t, bfloat16, acosh)

mag_gen_stub_unary(float, float32, atanh)
mag_gen_stub_unary(mag_float16_t, float16, atanh)
mag_gen_stub_unary(mag_bfloat16_t, bfloat16, atanh)

mag_gen_stub_unary(float, float32, step)
mag_gen_stub_unary(mag_float16_t, float16, step)
mag_gen_stub_unary(mag_bfloat16_t, bfloat16, step)

mag_gen_stub_unary(float, float32, erf)
mag_gen_stub_unary(mag_float16_t, float16, erf)
mag_gen_stub_unary(mag_bfloat16_t, bfloat16, erf)

mag_gen_stub_unary(float, float32, erfc)
mag_gen_stub_unary(mag_float16_t, float16, erfc)
mag_gen_stub_unary(mag_bfloat16_t, bfloat16, erfc)

mag_gen_stub_unary(float, float32, exp)
mag_gen_stub_unary(mag_float16_t, float16, exp)
mag_gen_stub_unary(mag_bfloat16_t, bfloat16, exp)

mag_gen_stub_unary(float, float32, exp2)
mag_gen_stub_unary(mag_float16_t, float16, exp2)
mag_gen_stub_unary(mag_bfloat16_t, bfloat16, exp2)

mag_gen_stub_unary(float, float32, expm1)
mag_gen_stub_unary(mag_float16_t, float16, expm1)
mag_gen_stub_unary(mag_bfloat16_t, bfloat16, expm1)

mag_gen_stub_unary(float, float32, floor)
mag_gen_stub_unary(mag_float16_t, float16, floor)
mag_gen_stub_unary(mag_bfloat16_t, bfloat16, floor)

mag_gen_stub_unary(float, float32, ceil)
mag_gen_stub_unary(mag_float16_t, float16, ceil)
mag_gen_stub_unary(mag_bfloat16_t, bfloat16, ceil)

mag_gen_stub_unary(float, float32, round)
mag_gen_stub_unary(mag_float16_t, float16, round)
mag_gen_stub_unary(mag_bfloat16_t, bfloat16, round)

mag_gen_stub_unary(float, float32, trunc)
mag_gen_stub_unary(mag_float16_t, float16, trunc)
mag_gen_stub_unary(mag_bfloat16_t, bfloat16, trunc)

mag_gen_stub_unary(float, float32, softmax_dv)
mag_gen_stub_unary(mag_float16_t, float16, softmax_dv)
mag_gen_stub_unary(mag_bfloat16_t, bfloat16, softmax_dv)

mag_gen_stub_unary(float, float32, sigmoid)
mag_gen_stub_unary(mag_float16_t, float16, sigmoid)
mag_gen_stub_unary(mag_bfloat16_t, bfloat16, sigmoid)

mag_gen_stub_unary(float, float32, sigmoid_dv)
mag_gen_stub_unary(mag_float16_t, float16, sigmoid_dv)
mag_gen_stub_unary(mag_bfloat16_t, bfloat16, sigmoid_dv)

mag_gen_stub_unary(float, float32, hard_sigmoid)
mag_gen_stub_unary(mag_float16_t, float16, hard_sigmoid)
mag_gen_stub_unary(mag_bfloat16_t, bfloat16, hard_sigmoid)

mag_gen_stub_unary(float, float32, silu)
mag_gen_stub_unary(mag_float16_t, float16, silu)
mag_gen_stub_unary(mag_bfloat16_t, bfloat16, silu)

mag_gen_stub_unary(float, float32, silu_dv)
mag_gen_stub_unary(mag_float16_t, float16, silu_dv)
mag_gen_stub_unary(mag_bfloat16_t, bfloat16, silu_dv)

mag_gen_stub_unary(float, float32, tanh_dv)
mag_gen_stub_unary(mag_float16_t, float16, tanh_dv)
mag_gen_stub_unary(mag_bfloat16_t, bfloat16, tanh_dv)

mag_gen_stub_unary(float, float32, relu)
mag_gen_stub_unary(mag_float16_t, float16, relu)
mag_gen_stub_unary(mag_bfloat16_t, bfloat16, relu)

mag_gen_stub_unary(float, float32, relu_dv)
mag_gen_stub_unary(mag_float16_t, float16, relu_dv)
mag_gen_stub_unary(mag_bfloat16_t, bfloat16, relu_dv)

mag_gen_stub_unary(float, float32, gelu)
mag_gen_stub_unary(mag_float16_t, float16, gelu)
mag_gen_stub_unary(mag_bfloat16_t, bfloat16, gelu)

mag_gen_stub_unary(float, float32, gelu_approx)
mag_gen_stub_unary(mag_float16_t, float16, gelu_approx)
mag_gen_stub_unary(mag_bfloat16_t, bfloat16, gelu_approx)

mag_gen_stub_unary(float, float32, gelu_dv)
mag_gen_stub_unary(mag_float16_t, float16, gelu_dv)
mag_gen_stub_unary(mag_bfloat16_t, bfloat16, gelu_dv)

mag_gen_stub_unary(uint8_t, uint8, not)
mag_gen_stub_unary(int8_t, int8, not)
mag_gen_stub_unary(uint16_t, uint16, not)
mag_gen_stub_unary(int16_t, int16, not)
mag_gen_stub_unary(uint32_t, uint32, not)
mag_gen_stub_unary(int32_t, int32, not)
mag_gen_stub_unary(uint64_t, uint64, not)
mag_gen_stub_unary(int64_t, int64, not)

#undef mag_gen_stub_unary

static void MAG_HOTPROC mag_softmax_float32(const mag_kernel_payload_t *payload) {
  mag_tensor_t *r = mag_cmd_out(0);
  const mag_tensor_t *x = mag_cmd_in(0);
  mag_assert(mag_tensor_is_contiguous(x), "Softmax input tensor must be contiguous");
  float *br = (float *)mag_tensor_data_ptr_mut(r);
  const float *bx = (const float *)mag_tensor_data_ptr(x);
  int64_t rank  = r->coords.rank;
  int64_t numel = r->numel;
  if (mag_unlikely(!numel)) return;
  if (rank == 0) {
    if (payload->thread_idx == 0) *br = 1.0f;
    return;
  }
  int64_t last_dim = r->coords.shape[rank-1];
  int64_t rows = r->numel / last_dim;
  int64_t tc = payload->thread_num;
  int64_t ti = payload->thread_idx;
  int64_t rpt = (rows + tc - 1)/tc;
  int64_t ra = ti*rpt;
  int64_t rb = mag_xmin(ra + rpt, rows);
  for (int64_t ri=ra; ri < rb; ++ri) {
    const float *row_in = bx + ri*last_dim;
    mag_bnd_chk(row_in, bx, mag_tensor_numbytes(x));
    float *row_out = br + ri*last_dim;
    float max_val = row_in[0]; /* Max val is computed for numerical stability */
    for (int64_t i=1; i < last_dim; ++i) {
      if (row_in[i] > max_val) {
        mag_bnd_chk(row_in+i, bx, mag_tensor_numbytes(x));
        max_val = row_in[i];
      }
    }
    double sum = 0.0f;
    for (int64_t i=0; i < last_dim; ++i) {
      mag_bnd_chk(row_in+i, bx, mag_tensor_numbytes(x));
      mag_bnd_chk(row_out+i, br, mag_tensor_numbytes(r));
      row_out[i] = expf(row_in[i] - max_val); /* -max for numerical stability */
      sum += row_out[i];
    }
    if (!isfinite(sum) || sum <= 0.0) {
      float inv = 1.0f / (float)last_dim;
      for (int64_t i=0; i < last_dim; ++i) row_out[i] = inv;
    } else {
      float inv = (float)(1.0 / sum);
      for (int64_t i=0; i < last_dim; ++i) row_out[i] *= inv;
    }
  }
}

static void MAG_HOTPROC mag_softmax_float16(const mag_kernel_payload_t *payload) {
  mag_tensor_t *r = mag_cmd_out(0);
  const mag_tensor_t *x = mag_cmd_in(0);
  mag_assert(mag_tensor_is_contiguous(x), "Softmax input tensor must be contiguous");
  mag_float16_t *br = (mag_float16_t *)mag_tensor_data_ptr_mut(r);
  const mag_float16_t *bx = (const mag_float16_t *)mag_tensor_data_ptr(x);
  int64_t rank  = r->coords.rank;
  int64_t numel = r->numel;
  if (mag_unlikely(!numel)) return;
  if (rank == 0) {
    if (payload->thread_idx == 0) *br = MAG_FLOAT16_ONE;
    return;
  }
  int64_t last_dim = r->coords.shape[r->coords.rank-1];
  int64_t rows = r->numel / last_dim;
  int64_t tc = payload->thread_num;
  int64_t ti = payload->thread_idx;
  int64_t rpt = (rows + tc - 1)/tc;
  int64_t ra = ti*rpt;
  int64_t rb = mag_xmin(ra + rpt, rows);
  for (int64_t ri=ra; ri < rb; ++ri) {
    const mag_float16_t *row_in = bx + ri*last_dim;
    mag_bnd_chk(row_in, bx, mag_tensor_numbytes(x));
    mag_float16_t *row_out = br + ri*last_dim;
    float max_val = mag_float16_to_float32(row_in[0]); /* Max val is computed for numerical stability */
    for (int64_t i=1; i < last_dim; ++i) {
      if (mag_float16_to_float32(row_in[i]) > max_val) {
        mag_bnd_chk(row_in+i, bx, mag_tensor_numbytes(x));
        max_val = mag_float16_to_float32(row_in[i]);
      }
    }
    double sum = 0.0f;
    for (int64_t i=0; i < last_dim; ++i) {
      mag_bnd_chk(row_in+i, bx, mag_tensor_numbytes(x));
      mag_bnd_chk(row_out+i, br, mag_tensor_numbytes(r));
      float ro = expf(mag_float16_to_float32(row_in[i]) - max_val);
      row_out[i] = mag_float32_to_float16(ro); /* -max for numerical stability */
      sum += ro;
    }
    if (!isfinite(sum) || sum <= 0.0) {
      mag_float16_t inv = mag_float32_to_float16(1.0f / (float)last_dim);
      for (int64_t i=0; i < last_dim; ++i) row_out[i] = inv;
    } else {
      float inv = (float)(1.0 / sum);
      for (int64_t i=0; i < last_dim; ++i) row_out[i] = mag_float32_to_float16(mag_float16_to_float32(row_out[i]) * inv);
    }
  }
}

static void MAG_HOTPROC mag_softmax_bfloat16(const mag_kernel_payload_t *payload) {
  mag_tensor_t *r = mag_cmd_out(0);
  const mag_tensor_t *x = mag_cmd_in(0);
  mag_assert(mag_tensor_is_contiguous(x), "Softmax input tensor must be contiguous");
  mag_bfloat16_t *br = (mag_bfloat16_t *)mag_tensor_data_ptr_mut(r);
  const mag_bfloat16_t *bx = (const mag_bfloat16_t *)mag_tensor_data_ptr(x);
  int64_t rank  = r->coords.rank;
  int64_t numel = r->numel;
  if (mag_unlikely(!numel)) return;
  if (rank == 0) {
    if (payload->thread_idx == 0) *br = MAG_BFLOAT16_ONE;
    return;
  }
  int64_t last_dim = r->coords.shape[r->coords.rank-1];
  int64_t rows = r->numel / last_dim;
  int64_t tc = payload->thread_num;
  int64_t ti = payload->thread_idx;
  int64_t rpt = (rows + tc - 1)/tc;
  int64_t ra = ti*rpt;
  int64_t rb = mag_xmin(ra + rpt, rows);
  for (int64_t ri=ra; ri < rb; ++ri) {
    const mag_bfloat16_t *row_in = bx + ri*last_dim;
    mag_bnd_chk(row_in, bx, mag_tensor_numbytes(x));
    mag_bfloat16_t *row_out = br + ri*last_dim;
    float max_val = mag_bfloat16_to_float32(row_in[0]); /* Max val is computed for numerical stability */
    for (int64_t i=1; i < last_dim; ++i) {
      if (mag_bfloat16_to_float32(row_in[i]) > max_val) {
        mag_bnd_chk(row_in+i, bx, mag_tensor_numbytes(x));
        max_val = mag_bfloat16_to_float32(row_in[i]);
      }
    }
    double sum = 0.0f;
    for (int64_t i=0; i < last_dim; ++i) {
      mag_bnd_chk(row_in+i, bx, mag_tensor_numbytes(x));
      mag_bnd_chk(row_out+i, br, mag_tensor_numbytes(r));
      float ro = expf(mag_bfloat16_to_float32(row_in[i]) - max_val);
      row_out[i] = mag_float32_to_bfloat16(ro); /* -max for numerical stability */
      sum += ro;
    }
    if (!isfinite(sum) || sum <= 0.0) {
      mag_bfloat16_t inv = mag_float32_to_bfloat16(1.0f / (float)last_dim);
      for (int64_t i=0; i < last_dim; ++i) row_out[i] = inv;
    } else {
      float inv = (float)(1.0 / sum);
      for (int64_t i=0; i < last_dim; ++i) row_out[i] = mag_float32_to_bfloat16(mag_bfloat16_to_float32(row_out[i]) * inv);
    }
  }
}

typedef void (mag_vcast_fn_t)(int64_t numel, void *restrict dst, const void *restrict src);

#define mag_cast_fn_builtin(TDst, x) ((TDst)x)
#define mag_cast_fn_float32_to_float16(TDst, x) (mag_float32_to_float16(x))
#define mag_cast_fn_float16_to_float32_upcast(TDst, x) ((TDst)mag_float16_to_float32(x))
#define mag_cast_fn_float32_to_bfloat16(TDst, x) (mag_float32_to_bfloat16(x))
#define mag_cast_fn_bfloat16_to_float32_upcast(TDst, x) ((TDst)mag_bfloat16_to_float32(x))
#define mag_cast_fn_float16_to_bfloat16(TDst, x) (mag_float32_to_bfloat16(mag_float16_to_float32(x)))
#define mag_cast_fn_bfloat16_to_float16(TDst, x) (mag_float32_to_float16(mag_bfloat16_to_float32(x)))

#define mag_gen_vcast(TSrc, TDst, F) \
  static void MAG_HOTPROC mag_vcast_##TSrc##_to_##TDst(int64_t numel, void *restrict dst, const void *restrict src) { \
    TDst *restrict o = (TDst *)dst; \
    const TSrc *restrict x = (const TSrc *)src; \
    for (int64_t i=0; i < numel; ++i) \
      o[i] = F(TDst, x[i]); \
  }

#define mag_gen_vcast_to_bool_arith(TSrc) \
  static void MAG_HOTPROC mag_vcast_##TSrc##_to_boolean(int64_t numel, void *restrict dst, const void *restrict src) { \
    uint8_t *restrict o = (uint8_t *)dst; \
    const TSrc *restrict x = (const TSrc *)src; \
    for (int64_t i=0; i < numel; ++i) \
      o[i] = (uint8_t)(x[i]!=(TSrc)0); \
  }

/* Generate all dtype cast perms. TSrc == TDst are unused but available, as the op is delegate to the clone operator before. */
mag_gen_vcast_to_bool_arith(float);

static void MAG_HOTPROC mag_vcast_mag_float16_t_to_boolean(int64_t numel, void *restrict dst, const void *restrict src) {
  uint8_t *restrict o = dst;
  const mag_float16_t *restrict x = src;
  for (int64_t i=0; i < numel; ++i)
    o[i] = (uint8_t)(mag_float16_to_float32(x[i]) != 0.0f);
}

static void MAG_HOTPROC mag_vcast_mag_bfloat16_t_to_boolean(int64_t numel, void *restrict dst, const void *restrict src) {
  uint8_t *restrict o = dst;
  const mag_bfloat16_t *restrict x = src;
  for (int64_t i=0; i < numel; ++i)
    o[i] = (uint8_t)(mag_bfloat16_to_float32(x[i]) != 0.0f);
}

mag_gen_vcast_to_bool_arith(uint8_t);
mag_gen_vcast_to_bool_arith(int8_t);
mag_gen_vcast_to_bool_arith(uint16_t);
mag_gen_vcast_to_bool_arith(int16_t);
mag_gen_vcast_to_bool_arith(uint32_t);
mag_gen_vcast_to_bool_arith(int32_t);
mag_gen_vcast_to_bool_arith(uint64_t);
mag_gen_vcast_to_bool_arith(int64_t);

#undef mag_gen_vcast_to_bool_arith

mag_gen_vcast(float, float, mag_cast_fn_builtin)
/*mag_gen_vcast(float, mag_float16_t, mag_cast_fn_float32_to_float16) - There is a SIMD fast path function for this cast below */
mag_gen_vcast(float, mag_bfloat16_t, mag_cast_fn_float32_to_bfloat16)
mag_gen_vcast(float, uint8_t, mag_cast_fn_builtin)
mag_gen_vcast(float, int8_t, mag_cast_fn_builtin)
mag_gen_vcast(float, uint16_t, mag_cast_fn_builtin)
mag_gen_vcast(float, int16_t, mag_cast_fn_builtin)
mag_gen_vcast(float, uint32_t, mag_cast_fn_builtin)
mag_gen_vcast(float, int32_t, mag_cast_fn_builtin)
mag_gen_vcast(float, uint64_t, mag_cast_fn_builtin)
mag_gen_vcast(float, int64_t, mag_cast_fn_builtin)

/*mag_gen_vcast(mag_float16_t, float, mag_cast_fn_float16_to_float32_upcast)  - There is a SIMD fast path function for this cast below */
mag_gen_vcast(mag_float16_t, mag_bfloat16_t, mag_cast_fn_float16_to_bfloat16)
mag_gen_vcast(mag_float16_t, mag_float16_t, mag_cast_fn_builtin)
mag_gen_vcast(mag_float16_t, uint8_t, mag_cast_fn_float16_to_float32_upcast)
mag_gen_vcast(mag_float16_t, int8_t, mag_cast_fn_float16_to_float32_upcast)
mag_gen_vcast(mag_float16_t, uint16_t, mag_cast_fn_float16_to_float32_upcast)
mag_gen_vcast(mag_float16_t, int16_t, mag_cast_fn_float16_to_float32_upcast)
mag_gen_vcast(mag_float16_t, uint32_t, mag_cast_fn_float16_to_float32_upcast)
mag_gen_vcast(mag_float16_t, int32_t, mag_cast_fn_float16_to_float32_upcast)
mag_gen_vcast(mag_float16_t, uint64_t, mag_cast_fn_float16_to_float32_upcast)
mag_gen_vcast(mag_float16_t, int64_t, mag_cast_fn_float16_to_float32_upcast)

mag_gen_vcast(mag_bfloat16_t, float, mag_cast_fn_bfloat16_to_float32_upcast)
mag_gen_vcast(mag_bfloat16_t, mag_bfloat16_t, mag_cast_fn_builtin)
mag_gen_vcast(mag_bfloat16_t, mag_float16_t, mag_cast_fn_bfloat16_to_float16)
mag_gen_vcast(mag_bfloat16_t, uint8_t, mag_cast_fn_bfloat16_to_float32_upcast)
mag_gen_vcast(mag_bfloat16_t, int8_t, mag_cast_fn_bfloat16_to_float32_upcast)
mag_gen_vcast(mag_bfloat16_t, uint16_t, mag_cast_fn_bfloat16_to_float32_upcast)
mag_gen_vcast(mag_bfloat16_t, int16_t, mag_cast_fn_bfloat16_to_float32_upcast)
mag_gen_vcast(mag_bfloat16_t, uint32_t, mag_cast_fn_bfloat16_to_float32_upcast)
mag_gen_vcast(mag_bfloat16_t, int32_t, mag_cast_fn_bfloat16_to_float32_upcast)
mag_gen_vcast(mag_bfloat16_t, uint64_t, mag_cast_fn_bfloat16_to_float32_upcast)
mag_gen_vcast(mag_bfloat16_t, int64_t, mag_cast_fn_bfloat16_to_float32_upcast)

mag_gen_vcast(uint8_t, float, mag_cast_fn_builtin)
mag_gen_vcast(uint8_t, mag_float16_t, mag_cast_fn_float32_to_float16)
mag_gen_vcast(uint8_t, mag_bfloat16_t, mag_cast_fn_float32_to_bfloat16)
mag_gen_vcast(uint8_t, uint8_t, mag_cast_fn_builtin)
mag_gen_vcast(uint8_t, int8_t, mag_cast_fn_builtin)
mag_gen_vcast(uint8_t, uint16_t, mag_cast_fn_builtin)
mag_gen_vcast(uint8_t, int16_t, mag_cast_fn_builtin)
mag_gen_vcast(uint8_t, uint32_t, mag_cast_fn_builtin)
mag_gen_vcast(uint8_t, int32_t, mag_cast_fn_builtin)
mag_gen_vcast(uint8_t, uint64_t, mag_cast_fn_builtin)
mag_gen_vcast(uint8_t, int64_t, mag_cast_fn_builtin)

mag_gen_vcast(int8_t, float, mag_cast_fn_builtin)
mag_gen_vcast(int8_t, mag_float16_t, mag_cast_fn_float32_to_float16)
mag_gen_vcast(int8_t, mag_bfloat16_t, mag_cast_fn_float32_to_bfloat16)
mag_gen_vcast(int8_t, uint8_t, mag_cast_fn_builtin)
mag_gen_vcast(int8_t, int8_t, mag_cast_fn_builtin)
mag_gen_vcast(int8_t, uint16_t, mag_cast_fn_builtin)
mag_gen_vcast(int8_t, int16_t, mag_cast_fn_builtin)
mag_gen_vcast(int8_t, uint32_t, mag_cast_fn_builtin)
mag_gen_vcast(int8_t, int32_t, mag_cast_fn_builtin)
mag_gen_vcast(int8_t, uint64_t, mag_cast_fn_builtin)
mag_gen_vcast(int8_t, int64_t, mag_cast_fn_builtin)

mag_gen_vcast(uint16_t, float, mag_cast_fn_builtin)
mag_gen_vcast(uint16_t, mag_float16_t, mag_cast_fn_float32_to_float16)
mag_gen_vcast(uint16_t, mag_bfloat16_t, mag_cast_fn_float32_to_bfloat16)
mag_gen_vcast(uint16_t, uint8_t, mag_cast_fn_builtin)
mag_gen_vcast(uint16_t, int8_t, mag_cast_fn_builtin)
mag_gen_vcast(uint16_t, uint16_t, mag_cast_fn_builtin)
mag_gen_vcast(uint16_t, int16_t, mag_cast_fn_builtin)
mag_gen_vcast(uint16_t, uint32_t, mag_cast_fn_builtin)
mag_gen_vcast(uint16_t, int32_t, mag_cast_fn_builtin)
mag_gen_vcast(uint16_t, uint64_t, mag_cast_fn_builtin)
mag_gen_vcast(uint16_t, int64_t, mag_cast_fn_builtin)

mag_gen_vcast(int16_t, float, mag_cast_fn_builtin)
mag_gen_vcast(int16_t, mag_float16_t, mag_cast_fn_float32_to_float16)
mag_gen_vcast(int16_t, mag_bfloat16_t, mag_cast_fn_float32_to_bfloat16)
mag_gen_vcast(int16_t, uint8_t, mag_cast_fn_builtin)
mag_gen_vcast(int16_t, int8_t, mag_cast_fn_builtin)
mag_gen_vcast(int16_t, uint16_t, mag_cast_fn_builtin)
mag_gen_vcast(int16_t, int16_t, mag_cast_fn_builtin)
mag_gen_vcast(int16_t, uint32_t, mag_cast_fn_builtin)
mag_gen_vcast(int16_t, int32_t, mag_cast_fn_builtin)
mag_gen_vcast(int16_t, uint64_t, mag_cast_fn_builtin)
mag_gen_vcast(int16_t, int64_t, mag_cast_fn_builtin)

mag_gen_vcast(uint32_t, float, mag_cast_fn_builtin)
mag_gen_vcast(uint32_t, mag_float16_t, mag_cast_fn_float32_to_float16)
mag_gen_vcast(uint32_t, mag_bfloat16_t, mag_cast_fn_float32_to_bfloat16)
mag_gen_vcast(uint32_t, uint8_t, mag_cast_fn_builtin)
mag_gen_vcast(uint32_t, int8_t, mag_cast_fn_builtin)
mag_gen_vcast(uint32_t, uint16_t, mag_cast_fn_builtin)
mag_gen_vcast(uint32_t, int16_t, mag_cast_fn_builtin)
mag_gen_vcast(uint32_t, uint32_t, mag_cast_fn_builtin)
mag_gen_vcast(uint32_t, int32_t, mag_cast_fn_builtin)
mag_gen_vcast(uint32_t, uint64_t, mag_cast_fn_builtin)
mag_gen_vcast(uint32_t, int64_t, mag_cast_fn_builtin)

mag_gen_vcast(int32_t, float, mag_cast_fn_builtin)
mag_gen_vcast(int32_t, mag_float16_t, mag_cast_fn_float32_to_float16)
mag_gen_vcast(int32_t, mag_bfloat16_t, mag_cast_fn_float32_to_bfloat16)
mag_gen_vcast(int32_t, uint8_t, mag_cast_fn_builtin)
mag_gen_vcast(int32_t, int8_t, mag_cast_fn_builtin)
mag_gen_vcast(int32_t, uint16_t, mag_cast_fn_builtin)
mag_gen_vcast(int32_t, int16_t, mag_cast_fn_builtin)
mag_gen_vcast(int32_t, uint32_t, mag_cast_fn_builtin)
mag_gen_vcast(int32_t, int32_t, mag_cast_fn_builtin)
mag_gen_vcast(int32_t, uint64_t, mag_cast_fn_builtin)
mag_gen_vcast(int32_t, int64_t, mag_cast_fn_builtin)

mag_gen_vcast(uint64_t, float, mag_cast_fn_builtin)
mag_gen_vcast(uint64_t, mag_float16_t, mag_cast_fn_float32_to_float16)
mag_gen_vcast(uint64_t, mag_bfloat16_t, mag_cast_fn_float32_to_bfloat16)
mag_gen_vcast(uint64_t, uint8_t, mag_cast_fn_builtin)
mag_gen_vcast(uint64_t, int8_t, mag_cast_fn_builtin)
mag_gen_vcast(uint64_t, uint16_t, mag_cast_fn_builtin)
mag_gen_vcast(uint64_t, int16_t, mag_cast_fn_builtin)
mag_gen_vcast(uint64_t, uint32_t, mag_cast_fn_builtin)
mag_gen_vcast(uint64_t, int32_t, mag_cast_fn_builtin)
mag_gen_vcast(uint64_t, uint64_t, mag_cast_fn_builtin)
mag_gen_vcast(uint64_t, int64_t, mag_cast_fn_builtin)

mag_gen_vcast(int64_t, float, mag_cast_fn_builtin)
mag_gen_vcast(int64_t, mag_float16_t, mag_cast_fn_float32_to_float16)
mag_gen_vcast(int64_t, mag_bfloat16_t, mag_cast_fn_float32_to_bfloat16)
mag_gen_vcast(int64_t, uint8_t, mag_cast_fn_builtin)
mag_gen_vcast(int64_t, int8_t, mag_cast_fn_builtin)
mag_gen_vcast(int64_t, uint16_t, mag_cast_fn_builtin)
mag_gen_vcast(int64_t, int16_t, mag_cast_fn_builtin)
mag_gen_vcast(int64_t, uint32_t, mag_cast_fn_builtin)
mag_gen_vcast(int64_t, int32_t, mag_cast_fn_builtin)
mag_gen_vcast(int64_t, uint64_t, mag_cast_fn_builtin)
mag_gen_vcast(int64_t, int64_t, mag_cast_fn_builtin)

#undef mag_gen_vcast

/* SIMD fast paths */

static void MAG_HOTPROC mag_vcast_float_to_mag_float16_t(int64_t numel, void *restrict xo, const void *restrict xx) {
  mag_float16_t *o = xo;
  const float *x = xx;
  int64_t i=0;
#ifdef __ARM_NEON
  for (; i+3 < numel; i += 4) {
    float32x4_t v = vld1q_f32(x+i);
    vst1_f16((__fp16 *)o+i, vcvt_f16_f32(v));
  }
#elif defined(__F16C__)
#ifdef __AVX512F__
  for (; i+15 < numel; i += 16) {
    __m512 xv = _mm512_loadu_ps(x+i);
    __m256i yv = _mm512_cvtps_ph(xv, _MM_FROUND_TO_NEAREST_INT);
    _mm256_storeu_si256((__m256i *)(o+i), yv);
  }
#endif
  for (; i+7 < numel; i += 8) {
    __m256 xv = _mm256_loadu_ps(x+i);
    __m128i yv = _mm256_cvtps_ph(xv, _MM_FROUND_TO_NEAREST_INT);
    _mm_storeu_si128((__m128i *)(o+i), yv);
  }
  for (; i+3 < numel; i += 4) {
    __m128 xv = _mm_loadu_ps(x+i);
    __m128i yv = _mm_cvtps_ph(xv, _MM_FROUND_TO_NEAREST_INT);
    _mm_storel_epi64((__m128i *)(o+i), yv);
  }
#endif
  for (; i < numel; ++i) /* Scalar drain loop */
    o[i] = mag_float32_to_float16(x[i]);
}

static void MAG_HOTPROC mag_vcast_mag_float16_t_to_float(int64_t numel, void *restrict xo, const void *restrict xx) {
  float *o = xo;
  const mag_float16_t *x = xx;
  int64_t i=0;
#ifdef __ARM_NEON
  for (; i+3 < numel; i += 4) {
    float16x4_t v = vld1_f16((const __fp16 *)x+i);
    vst1q_f32(o+i, vcvt_f32_f16(v));
  }
#elif defined(__F16C__)
#ifdef __AVX512F__
  for (; i+15 < numel; i += 16) {
    __m256i xv = _mm256_loadu_si256((const __m256i *)(x+i));
    __m512 yv = _mm512_cvtph_ps(xv);
    _mm512_storeu_ps(o+i, yv);
  }
#endif
  for (; i+7 < numel; i += 8) {
    __m128i xv = _mm_loadu_si128((const __m128i *)(x+i));
    __m256 yv = _mm256_cvtph_ps(xv);
    _mm256_storeu_ps(o+i, yv);
  }
  for (; i+3 < numel; i += 4) {
    __m128i xv = _mm_loadl_epi64((const __m128i *)(x+i));
    __m128 yv = _mm_cvtph_ps(xv);
    _mm_storeu_ps(o+i, yv);
  }
#endif
  for (; i < numel; ++i) /* Scalar drain loop */
    o[i] = mag_float16_to_float32(x[i]);
}

/* Src -> Dst */
static mag_vcast_fn_t *const mag_cast_table_2D[MAG_DTYPE__NUM][MAG_DTYPE__NUM] = {
  [MAG_DTYPE_FLOAT32] = {
    [MAG_DTYPE_FLOAT32] = mag_vcast_float_to_float,
    [MAG_DTYPE_FLOAT16] = mag_vcast_float_to_mag_float16_t,
    [MAG_DTYPE_BFLOAT16] = mag_vcast_float_to_mag_bfloat16_t,
    [MAG_DTYPE_BOOLEAN] = mag_vcast_float_to_boolean,
    [MAG_DTYPE_UINT8] = mag_vcast_float_to_uint8_t,
    [MAG_DTYPE_INT8] = mag_vcast_float_to_int8_t,
    [MAG_DTYPE_UINT16] = mag_vcast_float_to_uint16_t,
    [MAG_DTYPE_INT16] = mag_vcast_float_to_int16_t,
    [MAG_DTYPE_UINT32] = mag_vcast_float_to_uint32_t,
    [MAG_DTYPE_INT32] = mag_vcast_float_to_int32_t,
    [MAG_DTYPE_UINT64] = mag_vcast_float_to_uint64_t,
    [MAG_DTYPE_INT64] = mag_vcast_float_to_int64_t,
  },
  [MAG_DTYPE_FLOAT16] = {
    [MAG_DTYPE_FLOAT32] = mag_vcast_mag_float16_t_to_float,
    [MAG_DTYPE_FLOAT16] = mag_vcast_mag_float16_t_to_mag_float16_t,
    [MAG_DTYPE_BFLOAT16] = mag_vcast_mag_float16_t_to_mag_bfloat16_t,
    [MAG_DTYPE_BOOLEAN] = mag_vcast_mag_float16_t_to_boolean,
    [MAG_DTYPE_UINT8] = mag_vcast_mag_float16_t_to_uint8_t,
    [MAG_DTYPE_INT8] = mag_vcast_mag_float16_t_to_int8_t,
    [MAG_DTYPE_UINT16] = mag_vcast_mag_float16_t_to_uint16_t,
    [MAG_DTYPE_INT16] = mag_vcast_mag_float16_t_to_int16_t,
    [MAG_DTYPE_UINT32] = mag_vcast_mag_float16_t_to_uint32_t,
    [MAG_DTYPE_INT32] = mag_vcast_mag_float16_t_to_int32_t,
    [MAG_DTYPE_UINT64] = mag_vcast_mag_float16_t_to_uint64_t,
    [MAG_DTYPE_INT64] = mag_vcast_mag_float16_t_to_int64_t,
  },
  [MAG_DTYPE_BFLOAT16] = {
    [MAG_DTYPE_FLOAT32] = mag_vcast_mag_bfloat16_t_to_float,
    [MAG_DTYPE_FLOAT16] = mag_vcast_mag_bfloat16_t_to_mag_float16_t,
    [MAG_DTYPE_BFLOAT16] = mag_vcast_mag_bfloat16_t_to_mag_bfloat16_t,
    [MAG_DTYPE_BOOLEAN] = mag_vcast_mag_bfloat16_t_to_boolean,
    [MAG_DTYPE_UINT8] = mag_vcast_mag_bfloat16_t_to_uint8_t,
    [MAG_DTYPE_INT8] = mag_vcast_mag_bfloat16_t_to_int8_t,
    [MAG_DTYPE_UINT16] = mag_vcast_mag_bfloat16_t_to_uint16_t,
    [MAG_DTYPE_INT16] = mag_vcast_mag_bfloat16_t_to_int16_t,
    [MAG_DTYPE_UINT32] = mag_vcast_mag_bfloat16_t_to_uint32_t,
    [MAG_DTYPE_INT32] = mag_vcast_mag_bfloat16_t_to_int32_t,
    [MAG_DTYPE_UINT64] = mag_vcast_mag_bfloat16_t_to_uint64_t,
    [MAG_DTYPE_INT64] = mag_vcast_mag_bfloat16_t_to_int64_t,
  },
  [MAG_DTYPE_BOOLEAN] = {
    [MAG_DTYPE_FLOAT32] = mag_vcast_uint8_t_to_float,
    [MAG_DTYPE_FLOAT16] = mag_vcast_uint8_t_to_mag_float16_t,
    [MAG_DTYPE_BFLOAT16] = mag_vcast_uint8_t_to_mag_bfloat16_t,
    [MAG_DTYPE_BOOLEAN] = mag_vcast_uint8_t_to_boolean,
    [MAG_DTYPE_UINT8] = mag_vcast_uint8_t_to_uint8_t,
    [MAG_DTYPE_INT8] = mag_vcast_uint8_t_to_int8_t,
    [MAG_DTYPE_UINT16] = mag_vcast_uint8_t_to_uint16_t,
    [MAG_DTYPE_INT16] = mag_vcast_uint8_t_to_int16_t,
    [MAG_DTYPE_UINT32] = mag_vcast_uint8_t_to_uint32_t,
    [MAG_DTYPE_INT32] = mag_vcast_uint8_t_to_int32_t,
    [MAG_DTYPE_UINT64] = mag_vcast_uint8_t_to_uint64_t,
    [MAG_DTYPE_INT64] = mag_vcast_uint8_t_to_int64_t,
  },
  [MAG_DTYPE_UINT8] = {
    [MAG_DTYPE_FLOAT32] = mag_vcast_uint8_t_to_float,
    [MAG_DTYPE_FLOAT16] = mag_vcast_uint8_t_to_mag_float16_t,
    [MAG_DTYPE_BFLOAT16] = mag_vcast_uint8_t_to_mag_bfloat16_t,
    [MAG_DTYPE_BOOLEAN] = mag_vcast_uint8_t_to_boolean,
    [MAG_DTYPE_UINT8] = mag_vcast_uint8_t_to_uint8_t,
    [MAG_DTYPE_INT8] = mag_vcast_uint8_t_to_int8_t,
    [MAG_DTYPE_UINT16] = mag_vcast_uint8_t_to_uint16_t,
    [MAG_DTYPE_INT16] = mag_vcast_uint8_t_to_int16_t,
    [MAG_DTYPE_UINT32] = mag_vcast_uint8_t_to_uint32_t,
    [MAG_DTYPE_INT32] = mag_vcast_uint8_t_to_int32_t,
    [MAG_DTYPE_UINT64] = mag_vcast_uint8_t_to_uint64_t,
    [MAG_DTYPE_INT64] = mag_vcast_uint8_t_to_int64_t,
  },
  [MAG_DTYPE_INT8] = {
    [MAG_DTYPE_FLOAT32] = mag_vcast_int8_t_to_float,
    [MAG_DTYPE_FLOAT16] = mag_vcast_int8_t_to_mag_float16_t,
    [MAG_DTYPE_BFLOAT16] = mag_vcast_int8_t_to_mag_bfloat16_t,
    [MAG_DTYPE_BOOLEAN] = mag_vcast_int8_t_to_boolean,
    [MAG_DTYPE_UINT8] = mag_vcast_int8_t_to_uint8_t,
    [MAG_DTYPE_INT8] = mag_vcast_int8_t_to_int8_t,
    [MAG_DTYPE_UINT16] = mag_vcast_int8_t_to_uint16_t,
    [MAG_DTYPE_INT16] = mag_vcast_int8_t_to_int16_t,
    [MAG_DTYPE_UINT32] = mag_vcast_int8_t_to_uint32_t,
    [MAG_DTYPE_INT32] = mag_vcast_int8_t_to_int32_t,
    [MAG_DTYPE_UINT64] = mag_vcast_int8_t_to_uint64_t,
    [MAG_DTYPE_INT64] = mag_vcast_int8_t_to_int64_t,
  },
  [MAG_DTYPE_UINT16] = {
    [MAG_DTYPE_FLOAT32] = mag_vcast_uint16_t_to_float,
    [MAG_DTYPE_FLOAT16] = mag_vcast_uint16_t_to_mag_float16_t,
    [MAG_DTYPE_BFLOAT16] = mag_vcast_uint16_t_to_mag_bfloat16_t,
    [MAG_DTYPE_BOOLEAN] = mag_vcast_uint16_t_to_boolean,
    [MAG_DTYPE_UINT8] = mag_vcast_uint16_t_to_uint8_t,
    [MAG_DTYPE_INT8] = mag_vcast_uint16_t_to_int8_t,
    [MAG_DTYPE_UINT16] = mag_vcast_uint16_t_to_uint16_t,
    [MAG_DTYPE_INT16] = mag_vcast_uint16_t_to_int16_t,
    [MAG_DTYPE_UINT32] = mag_vcast_uint16_t_to_uint32_t,
    [MAG_DTYPE_INT32] = mag_vcast_uint16_t_to_int32_t,
    [MAG_DTYPE_UINT64] = mag_vcast_uint16_t_to_uint64_t,
    [MAG_DTYPE_INT64] = mag_vcast_uint16_t_to_int64_t,
  },
  [MAG_DTYPE_INT16] = {
    [MAG_DTYPE_FLOAT32] = mag_vcast_int16_t_to_float,
    [MAG_DTYPE_FLOAT16] = mag_vcast_int16_t_to_mag_float16_t,
    [MAG_DTYPE_BFLOAT16] = mag_vcast_int16_t_to_mag_bfloat16_t,
    [MAG_DTYPE_BOOLEAN] = mag_vcast_int16_t_to_boolean,
    [MAG_DTYPE_UINT8] = mag_vcast_int16_t_to_uint8_t,
    [MAG_DTYPE_INT8] = mag_vcast_int16_t_to_int8_t,
    [MAG_DTYPE_UINT16] = mag_vcast_int16_t_to_uint16_t,
    [MAG_DTYPE_INT16] = mag_vcast_int16_t_to_int16_t,
    [MAG_DTYPE_UINT32] = mag_vcast_int16_t_to_uint32_t,
    [MAG_DTYPE_INT32] = mag_vcast_int16_t_to_int32_t,
    [MAG_DTYPE_UINT64] = mag_vcast_int16_t_to_uint64_t,
    [MAG_DTYPE_INT64] = mag_vcast_int16_t_to_int64_t,
  },
  [MAG_DTYPE_UINT32] = {
    [MAG_DTYPE_FLOAT32] = mag_vcast_uint32_t_to_float,
    [MAG_DTYPE_FLOAT16] = mag_vcast_uint32_t_to_mag_float16_t,
    [MAG_DTYPE_BFLOAT16] = mag_vcast_uint32_t_to_mag_bfloat16_t,
    [MAG_DTYPE_BOOLEAN] = mag_vcast_uint32_t_to_boolean,
    [MAG_DTYPE_UINT8] = mag_vcast_uint32_t_to_uint8_t,
    [MAG_DTYPE_INT8] = mag_vcast_uint32_t_to_int8_t,
    [MAG_DTYPE_UINT16] = mag_vcast_uint32_t_to_uint16_t,
    [MAG_DTYPE_INT16] = mag_vcast_uint32_t_to_int16_t,
    [MAG_DTYPE_UINT32] = mag_vcast_uint32_t_to_uint32_t,
    [MAG_DTYPE_INT32] = mag_vcast_uint32_t_to_int32_t,
    [MAG_DTYPE_UINT64] = mag_vcast_uint32_t_to_uint64_t,
    [MAG_DTYPE_INT64] = mag_vcast_uint32_t_to_int64_t,
  },
  [MAG_DTYPE_INT32] = {
    [MAG_DTYPE_FLOAT32] = mag_vcast_int32_t_to_float,
    [MAG_DTYPE_FLOAT16] = mag_vcast_int32_t_to_mag_float16_t,
    [MAG_DTYPE_BFLOAT16] = mag_vcast_int32_t_to_mag_bfloat16_t,
    [MAG_DTYPE_BOOLEAN] = mag_vcast_int32_t_to_boolean,
    [MAG_DTYPE_UINT8] = mag_vcast_int32_t_to_uint8_t,
    [MAG_DTYPE_INT8] = mag_vcast_int32_t_to_int8_t,
    [MAG_DTYPE_UINT16] = mag_vcast_int32_t_to_uint16_t,
    [MAG_DTYPE_INT16] = mag_vcast_int32_t_to_int16_t,
    [MAG_DTYPE_UINT32] = mag_vcast_int32_t_to_uint32_t,
    [MAG_DTYPE_INT32] = mag_vcast_int32_t_to_int32_t,
    [MAG_DTYPE_UINT64] = mag_vcast_int32_t_to_uint64_t,
    [MAG_DTYPE_INT64] = mag_vcast_int32_t_to_int64_t,
  },
  [MAG_DTYPE_UINT64] = {
    [MAG_DTYPE_FLOAT32] = mag_vcast_uint64_t_to_float,
    [MAG_DTYPE_FLOAT16] = mag_vcast_uint64_t_to_mag_float16_t,
    [MAG_DTYPE_BFLOAT16] = mag_vcast_uint64_t_to_mag_bfloat16_t,
    [MAG_DTYPE_BOOLEAN] = mag_vcast_uint64_t_to_boolean,
    [MAG_DTYPE_UINT8] = mag_vcast_uint64_t_to_uint8_t,
    [MAG_DTYPE_INT8] = mag_vcast_uint64_t_to_int8_t,
    [MAG_DTYPE_UINT16] = mag_vcast_uint64_t_to_uint16_t,
    [MAG_DTYPE_INT16] = mag_vcast_uint64_t_to_int16_t,
    [MAG_DTYPE_UINT32] = mag_vcast_uint64_t_to_uint32_t,
    [MAG_DTYPE_INT32] = mag_vcast_uint64_t_to_int32_t,
    [MAG_DTYPE_UINT64] = mag_vcast_uint64_t_to_uint64_t,
    [MAG_DTYPE_INT64] = mag_vcast_uint64_t_to_int64_t,
  },
  [MAG_DTYPE_INT64] = {
    [MAG_DTYPE_FLOAT32] = mag_vcast_int64_t_to_float,
    [MAG_DTYPE_FLOAT16] = mag_vcast_int64_t_to_mag_float16_t,
    [MAG_DTYPE_BFLOAT16] = mag_vcast_int64_t_to_mag_bfloat16_t,
    [MAG_DTYPE_BOOLEAN] = mag_vcast_int64_t_to_boolean,
    [MAG_DTYPE_UINT8] = mag_vcast_int64_t_to_uint8_t,
    [MAG_DTYPE_INT8] = mag_vcast_int64_t_to_int8_t,
    [MAG_DTYPE_UINT16] = mag_vcast_int64_t_to_uint16_t,
    [MAG_DTYPE_INT16] = mag_vcast_int64_t_to_int16_t,
    [MAG_DTYPE_UINT32] = mag_vcast_int64_t_to_uint32_t,
    [MAG_DTYPE_INT32] = mag_vcast_int64_t_to_int32_t,
    [MAG_DTYPE_UINT64] = mag_vcast_int64_t_to_uint64_t,
    [MAG_DTYPE_INT64] = mag_vcast_int64_t_to_int64_t,
  },
};

static MAG_HOTPROC void mag_cast_generic(const mag_kernel_payload_t *payload) {
  mag_tensor_t *r = mag_cmd_out(0);
  const mag_tensor_t *x = mag_cmd_in(0);
  mag_dtype_t src = x->dtype;
  mag_dtype_t dst = r->dtype;
  const mag_type_traits_t *msrc = mag_type_trait(src);
  const mag_type_traits_t *mdst = mag_type_trait(dst);
  mag_vcast_fn_t *kernel = mag_cast_table_2D[src][dst];
  mag_assert(kernel, "No kernel found for type cast: %s -> %s", msrc->name, mdst->name);
  uint8_t *br = (uint8_t *)mag_tensor_data_ptr_mut(r);
  const uint8_t *bx = (const uint8_t *)mag_tensor_data_ptr(x);
  int64_t nbs = (int64_t)msrc->size;
  int64_t nbd = (int64_t)mdst->size;
  int64_t total = r->numel;
  int64_t tc = payload->thread_num;
  int64_t ti = payload->thread_idx;
  int64_t chunk = (total + tc - 1)/tc;
  int64_t ra = ti*chunk;
  int64_t rb = mag_xmin(ra + chunk, total);
  if (mag_unlikely(rb <= ra)) return;
  if (mag_all_shapes_equal_and_contig((const mag_tensor_t *[2]){r, x}, 2)) { /* TODO: can be relaxed to non-shape equality */
    void *pr = br + ra*nbd;
    const void *px = bx + ra*nbs;
    mag_bnd_chk(px, bx, mag_tensor_numbytes(x));
    mag_bnd_chk(pr, br, mag_tensor_numbytes(r));
    (*kernel)(rb-ra, pr, px);
    return;
  }
  /* We work in byte granularity and compute pointer offsets manually to avoid a generic for this stub function */
  mag_coords_iter_t cr, cx;
  mag_coords_iter_init(&cr, &r->coords);
  mag_coords_iter_init(&cx, &x->coords);
  for (int64_t i=ra; i < rb; ++i) { /* TODO: Optimize - Slow with the single indirect call for each element */
    int64_t ri, xi;
    mag_coords_iter_offset2(&cr, &cx, i, &ri, &xi);
    void *pr = br + ri*nbd;
    const void *px = bx + xi*nbs;
    mag_bnd_chk(px, bx, mag_tensor_numbytes(x));
    mag_bnd_chk(pr, br, mag_tensor_numbytes(r));
    (*kernel)(1, pr, px);
  }
}
