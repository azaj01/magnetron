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

#include "mag_cpu_specialization_detector.h"

#include <core/mag_context.h>
#include <core/mag_cpuid.h>
#include <core/mag_sstream.h>

extern void mag_cpu_blas_specialization_fallback(mag_kernel_registry_t *kernels); /* Generic any CPU impl */

typedef struct mag_cpu_specialization_t {
  const char *name;
  uint64_t (*get_feature_bitset)(void);
  void (*inject_kernels)(mag_kernel_registry_t *reg);
} mag_cpu_specialization_t;

#define mag_cpu_specialization_extern(arch, flag) \
extern uint64_t mag_cpu_blas_specialization_##arch##_##flag##_features(void); \
extern void mag_cpu_blas_specialization_##arch##_##flag(mag_kernel_registry_t *kernels)

#define mag_cpu_specialization_configure(arch, flag) \
(mag_cpu_specialization_t) { \
.name = #arch"-"#flag, \
.get_feature_bitset = &mag_cpu_blas_specialization_##arch##_##flag##_features, \
.inject_kernels = &mag_cpu_blas_specialization_##arch##_##flag \
}

#if defined(__x86_64__) || defined(_M_X64) /* Specialized impls for x86-64 with runtime CPU detection */

  static uint64_t mag_get_cpu_host_caps(const mag_context_t *ctx) { return ctx->machine.amd64_cpu_caps; }

  static const mag_cpu_specialization_t *mag_get_cpu_specializations(const mag_context_t *ctx, size_t *num) {
    #ifdef MAG_HAVE_CPU_ALDERLAKE
      mag_cpu_specialization_extern(amd64, alderlake);
    #endif
    #ifdef MAG_HAVE_CPU_ARROWLAKE
      mag_cpu_specialization_extern(amd64, arrowlake);
    #endif
    #ifdef MAG_HAVE_CPU_CANNONLAKE
      mag_cpu_specialization_extern(amd64, cannonlake);
    #endif
    #ifdef MAG_HAVE_CPU_CASCADELAKE
      mag_cpu_specialization_extern(amd64, cascadelake);
    #endif
    #ifdef MAG_HAVE_CPU_COOPERLAKE
      mag_cpu_specialization_extern(amd64, cooperlake);
    #endif
    #ifdef MAG_HAVE_CPU_CORE2
      mag_cpu_specialization_extern(amd64, core2);
    #endif
    #ifdef MAG_HAVE_CPU_HASWELL
      mag_cpu_specialization_extern(amd64, haswell);
    #endif
    #ifdef MAG_HAVE_CPU_ICELAKE
      mag_cpu_specialization_extern(amd64, icelake);
    #endif
    #ifdef MAG_HAVE_CPU_IVYBRIDGE
      mag_cpu_specialization_extern(amd64, ivybridge);
    #endif
    #ifdef MAG_HAVE_CPU_NEHALEM
      mag_cpu_specialization_extern(amd64, nehalem);
    #endif
    #ifdef MAG_HAVE_CPU_WESTMERE
      mag_cpu_specialization_extern(amd64, westmere);
    #endif
    #ifdef MAG_HAVE_CPU_SANDYBRIDGE
      mag_cpu_specialization_extern(amd64, sandybridge);
    #endif
    #ifdef MAG_HAVE_CPU_SAPPHIRERAPIDS
      mag_cpu_specialization_extern(amd64, sapphirerapids);
    #endif
    #ifdef MAG_HAVE_CPU_SIERRAFOREST
      mag_cpu_specialization_extern(amd64, sierraforest);
    #endif
    #ifdef MAG_HAVE_CPU_SKYLAKE_AVX512
      mag_cpu_specialization_extern(amd64, skylake_avx512);
    #endif
    #ifdef MAG_HAVE_CPU_TIGERLAKE
      mag_cpu_specialization_extern(amd64, tigerlake);
    #endif
    #ifdef MAG_HAVE_CPU_ZNVER1
      mag_cpu_specialization_extern(amd64, zn1);
    #endif
    #ifdef MAG_HAVE_CPU_ZNVER2
      mag_cpu_specialization_extern(amd64, zn2);
    #endif
    #ifdef MAG_HAVE_CPU_ZNVER3
      mag_cpu_specialization_extern(amd64, zn3);
    #endif
    #ifdef MAG_HAVE_CPU_ZNVER4
      mag_cpu_specialization_extern(amd64, zn4);
    #endif
    #ifdef MAG_HAVE_CPU_ZNVER5
      mag_cpu_specialization_extern(amd64, zn5);
    #endif

    static const mag_cpu_specialization_t specializations_intel[] = {
      #ifdef MAG_HAVE_CPU_SAPPHIRERAPIDS
        mag_cpu_specialization_configure(amd64, sapphirerapids),
      #endif
      #ifdef MAG_HAVE_CPU_ICELAKE
        mag_cpu_specialization_configure(amd64, icelake),
      #endif
      #ifdef MAG_HAVE_CPU_COOPERLAKE
        mag_cpu_specialization_configure(amd64, cooperlake),
      #endif
      #ifdef MAG_HAVE_CPU_CASCADELAKE
        mag_cpu_specialization_configure(amd64, cascadelake),
      #endif
      #ifdef MAG_HAVE_CPU_TIGERLAKE
        mag_cpu_specialization_configure(amd64, tigerlake),
      #endif
      #ifdef MAG_HAVE_CPU_SKYLAKE_AVX512
        mag_cpu_specialization_configure(amd64, skylake_avx512),
      #endif
      #ifdef MAG_HAVE_CPU_HASWELL
        mag_cpu_specialization_configure(amd64, haswell),
      #endif
      #ifdef MAG_HAVE_CPU_ALDERLAKE
        mag_cpu_specialization_configure(amd64, alderlake),
      #endif
      #ifdef MAG_HAVE_CPU_ARROWLAKE
        mag_cpu_specialization_configure(amd64, arrowlake),
      #endif
      #ifdef MAG_HAVE_CPU_CANNONLAKE
        mag_cpu_specialization_configure(amd64, cannonlake),
      #endif
      #ifdef MAG_HAVE_CPU_IVYBRIDGE
        mag_cpu_specialization_configure(amd64, ivybridge),
      #endif
      #ifdef MAG_HAVE_CPU_SANDYBRIDGE
        mag_cpu_specialization_configure(amd64, sandybridge),
      #endif
      #ifdef MAG_HAVE_CPU_WESTMERE
        mag_cpu_specialization_configure(amd64, westmere),
      #endif
      #ifdef MAG_HAVE_CPU_NEHALEM
        mag_cpu_specialization_configure(amd64, nehalem),
      #endif
      #ifdef MAG_HAVE_CPU_CORE2
        mag_cpu_specialization_configure(amd64, core2),
      #endif
      #ifdef MAG_HAVE_CPU_SIERRAFOREST
        mag_cpu_specialization_configure(amd64, sierraforest),
      #endif
    };

    static const mag_cpu_specialization_t specializations_amd[] = {
      #ifdef MAG_HAVE_CPU_ZNVER5
        mag_cpu_specialization_configure(amd64, zn5),
      #endif
      #ifdef MAG_HAVE_CPU_ZNVER4
        mag_cpu_specialization_configure(amd64, zn4),
      #endif
      #ifdef MAG_HAVE_CPU_ZNVER3
        mag_cpu_specialization_configure(amd64, zn3),
      #endif
      #ifdef MAG_HAVE_CPU_ZNVER2
        mag_cpu_specialization_configure(amd64, zn2),
      #endif
      #ifdef MAG_HAVE_CPU_ZNVER1
        mag_cpu_specialization_configure(amd64, zn1),
      #endif
    };
    bool is_amd = ctx->machine.amd64_cpu_caps&mag_amd64_cap(AMD);
    *num = is_amd ? sizeof(specializations_amd)/sizeof(*specializations_amd) : sizeof(specializations_intel)/sizeof(*specializations_intel);
    return is_amd ? specializations_amd : specializations_intel;
  }

#elif defined(__aarch64__) || defined(_M_ARM64)

    static uint64_t mag_get_cpu_host_caps(const mag_context_t *ctx) { return ctx->machine.arm64_cpu_caps; }

    static const mag_cpu_specialization_t *mag_get_cpu_specializations(const mag_context_t *ctx, size_t *num) {
        #ifdef MAG_HAVE_CPU_ARMV9_A_SVE2
            mag_cpu_specialization_extern(arm64, v9_sve2);
        #endif
        #ifdef MAG_HAVE_CPU_ARMV8_2_A_SVE
            mag_cpu_specialization_extern(arm64, v82_sve);
        #endif
        #ifdef MAG_HAVE_CPU_ARMV8_6_A_BF16_I8MM_FP16_DOTPROD_CRYPTO
            mag_cpu_specialization_extern(arm64, v86_crypto);
        #endif
        #ifdef MAG_HAVE_CPU_ARMV8_6_A_BF16_I8MM_FP16_DOTPROD
            mag_cpu_specialization_extern(arm64, v86);
        #endif
        #ifdef MAG_HAVE_CPU_ARMV8_2_A_DOTPROD_FP16
            mag_cpu_specialization_extern(arm64, v82);
        #endif

      static const mag_cpu_specialization_t specializations[] = { /* Dynamic selectable BLAS permutations, sorted from best to worst score. */
            #ifdef MAG_HAVE_CPU_ARMV9_A_SVE2
              mag_cpu_specialization_configure(arm64, v9_sve2),
            #endif
            #ifdef MAG_HAVE_CPU_ARMV8_2_A_SVE
              mag_cpu_specialization_configure(arm64, v82_sve),
            #endif
            #ifdef MAG_HAVE_CPU_ARMV8_6_A_BF16_I8MM_FP16_DOTPROD_CRYPTO
              mag_cpu_specialization_configure(arm64, v86_crypto),
            #endif
            #ifdef MAG_HAVE_CPU_ARMV8_6_A_BF16_I8MM_FP16_DOTPROD
              mag_cpu_specialization_configure(arm64, v86),
            #endif
            #ifdef MAG_HAVE_CPU_ARMV8_2_A_DOTPROD_FP16
              mag_cpu_specialization_configure(arm64, v82),
            #endif
        };
        *num = sizeof(specializations)/sizeof(*specializations);
        return specializations;
    }

#elif defined(__loongarch64) /* Loongson / Godson backend selection */

typedef struct mag_loongarch64_specialization_dispatch_t {
    const char *name;
    mag_loongarch64_cap_bitset_t (*get_cap_permutation)(void);
    void (*inject_kernels)(mag_kernel_registry_t *kernels);
} mag_loongarch64_specialization_dispatch_t;

#define mag_loongarch64_spec_extern(feat) \
    mag_loongarch64_cap_bitset_t mag_cpu_blas_specialization_loongarch64_##feat##_features(void); \
    extern void mag_cpu_blas_specialization_loongarch64_##feat(mag_kernel_registry_t* kernels)

#define mag_loongarch64_spec_dispatch(feat) \
    (mag_loongarch64_specialization_dispatch_t) { \
        .name = "loongarch64-"#feat, \
        .get_cap_permutation = &mag_cpu_blas_specialization_loongarch64_##feat##_features, \
        .inject_kernels = &mag_cpu_blas_specialization_loongarch64_##feat \
}

#ifdef MAG_HAVE_CPU_LSX
mag_loongarch64_spec_extern(lsx);
#endif
#ifdef MAG_HAVE_CPU_LASX
mag_loongarch64_spec_extern(lasx);
#endif

static bool mag_blas_detect_gen_optimal_spec(const mag_context_t *ctx, mag_kernel_registry_t *kernels) {
    const mag_loongarch64_specialization_dispatch_t impls[] = { /* Dynamic selectable BLAS permutations, sorted from best to worst score. */
        #ifdef MAG_HAVE_CPU_LASX
            mag_loongarch64_spec_dispatch(lasx),
        #endif
        #ifdef MAG_HAVE_CPU_LSX
            mag_loongarch64_spec_dispatch(lsx),
        #endif
    };

    mag_loongarch64_cap_bitset_t cap_avail = ctx->machine.loongarch64_cpu_caps;
    for (size_t i=0; i < sizeof(impls)/sizeof(*impls); ++i) { /* Find best blas spec for the host CPU */
        const mag_loongarch64_specialization_dispatch_t *spec = impls+i;
        mag_loongarch64_cap_bitset_t cap_required = (*spec->get_cap_permutation)(); /* Get requires features */
        if ((cap_avail & cap_required) == cap_required) { /* Since specializations are sorted by score, we found the perfect spec. */
            (*spec->inject_kernels)(kernels);
            mag_log_info("Using tuned BLAS specialization: %s", spec->name);
            return true;
        }
    }

    /* No matching specialization found, use generic */
    mag_cpu_blas_specialization_fallback(kernels);
    mag_log_info("Using fallback BLAS specialization");
    return false; /* No spec used, fallback is active */
}

#else

static uint64_t mag_get_cpu_host_caps(const mag_context_t *ctx) { (void)ctx; return 0; }

static const mag_cpu_specialization_t *mag_get_cpu_specializations(const mag_context_t *ctx, size_t *num) {
  (void)ctx;
  *num = 0;
  return NULL;
}

#endif

bool mag_blas_detect_optimal_specialization(const mag_context_t *ctx, mag_kernel_registry_t *kernels) {
  size_t num_impls=0;
  const mag_cpu_specialization_t *impls = mag_get_cpu_specializations(ctx, &num_impls);
  if (mag_unlikely(!num_impls || !impls)) goto fallback;
  mag_log_debug("Available CPU specializations: %zu", num_impls);
  uint64_t host_caps = mag_get_cpu_host_caps(ctx);
  for (size_t i=0; i < num_impls; ++i) { /* Find best blas spec for the host CPU */
    const mag_cpu_specialization_t *spec = impls+i;
    uint64_t spec_caps = (*spec->get_feature_bitset)(); /* Get requires features */
    bool matches = (host_caps&spec_caps) == spec_caps;
    mag_log_debug("Checked specialization %s: requires 0x%" PRIx64 ", machine caps 0x%" PRIx64 ", matches: %s", spec->name, spec_caps, host_caps, matches ? "yes" : "no");
    if (matches) { /* Since specializations are sorted by score, we found the perfect spec. */
      (*spec->inject_kernels)(kernels);
      mag_log_info("Found tuned specialization: %s", spec->name);
      return true;
    }
  }
fallback:  /* No matching specialization found, use generic */
  mag_cpu_blas_specialization_fallback(kernels);
  mag_log_info("Using fallback BLAS specialization");
  return false; /* No spec used, fallback is active */
}
