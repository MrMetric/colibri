/* Pure-C inference engine for Poolside Laguna (model_type "laguna").
 * Supports Laguna XS 2.1 (33B-A3B), Laguna S 2.1 (118B-A8B), Laguna M.1 (225B-A23B)
 * — all share the same architecture, differing only in scale.
 *
 * Architecture (from modeling_laguna.py):
 *  - GQA with per-layer head counts, explicit head_dim=128, no QKV bias
 *  - QK RMSNorm (per-head, before RoPE)
 *  - RoPE: standard for sliding-attention layers, YaRN for full-attention layers,
 *    with partial_rotary_factor (0.5 for full-attn, 1.0 for sliding)
 *  - Sliding window attention (window=512) interleaved with global attention (1:3)
 *  - Per-head softplus attention output gating (g_proj)
 *  - MoE: sigmoid router with norm_topk_prob, routed_scaling_factor, 1 shared expert
 *  - Layer 0 is dense MLP only; layers 1+ are MoE
 *
 * Dense weights (attn, norms, router, shared expert, dense MLP, embed, lm_head)
 * resident in RAM as bf16 (real checkpoint) or f32 (tiny oracle). Routed experts
 * streamed from disk per-expert with LRU cache, optionally int4-quantized.
 *
 * Validation: same ref_*.json oracle harness as inkling.c/olmoe.c.
 *   build: make -C c laguna   run: SNAP=./laguna_tiny ./laguna <cap> <bits> <ref.json>
 *   TF=1 -> teacher-forcing (validates prefill on the whole sequence)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
#include <sys/resource.h>
#include <sys/select.h>
#endif
#include "st.h"
#include "tok.h"
#include "json.h"
#include "compat.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef COLI_CUDA
#include "backend_cuda_ink.h"
#include "backend_cuda.h"
static int g_cuda = 0;
#endif

#define MAXL 128

/* ---------- config ---------- */
typedef struct {
    int hidden, n_layers, vocab;
    int n_heads;           /* base num_attention_heads (global layers) */
    int n_kv;               /* num_key_value_heads */
    int head_dim;           /* explicit, =128 */
    int swa_heads;          /* heads for sliding-window layers */
    int swa_kv;
    int swa_hd;
    int window;             /* sliding window size */
    int n_experts, topk, n_shared;
    int moe_inter;          /* moe_intermediate_size */
    int dense_inter;        /* intermediate_size (dense MLP) */
    int shared_inter;      /* shared_expert_intermediate_size */
    float eps, route_scale; /* rms_norm_eps, moe_routed_scaling_factor */
    int norm_topk;          /* norm_topk_prob */
    int eos_ids[8], n_eos;  /* config eos_token_id may be a list (Laguna: [〈|EOS|〉, </assistant>]) */
    int bos;
    /* RoPE: full-attention layers */
    float rope_theta_full;
    int   rope_type_full;   /* 0=default, 1=yarn */
    float yarn_factor_full, yarn_beta_slow_full, yarn_beta_fast_full;
    float yarn_attn_factor_full;
    int   yarn_orig_max_full;
    float partial_rotary_full;
    /* RoPE: sliding-attention layers */
    float rope_theta_swa;
    int   rope_type_swa;    /* 0=default */
    float partial_rotary_swa;
    unsigned char local[MAXL];  /* 1 = sliding_attention layer */
    unsigned char sparse[MAXL]; /* 1 = MoE layer, 0 = dense MLP */
    int heads_per_layer[MAXL];  /* per-layer head count override */
} Cfg;

/* per-layer dims: Laguna uses num_attention_heads_per_layer for ALL layers
 * (global and sliding). KV heads and head_dim are the same across layers. */
#define L_HEADS(c,i)  ((c)->heads_per_layer[i])
#define L_KV(c,i)     ((c)->n_kv)
#define L_HD(c,i)     ((c)->head_dim)

/* ---------- resident weights ---------- */
typedef struct { float *f; uint16_t *h; void *dev; } Wt;

typedef struct {
    float *in_ln, *post_ln;
    Wt q, k, v, g, o;          /* projections; g = per-head softplus gate */
    float *q_norm, *k_norm;     /* per-head RMSNorm weights [head_dim] */
    /* dense layers */
    Wt dg, du, dd;
    /* MoE layers */
    float *router;             /* [E, D] */
    Wt sh_g, sh_u, sh_d;       /* shared expert [I,D] etc. */
} Layer;

/* ---------- routed-expert cache: LRU + optional pins ---------- */
typedef struct {
    int eid; uint64_t used;
    int pinned, filled;
    uint8_t *p13, *p2; float *s13, *s2;  /* container: packed rows + scales */
    int8_t *q13, *q2;                    /* bits>0: runtime-quantized int8 */
    float *f13, *f2;                     /* bits==0: raw f32 */
} Slot;
typedef struct { Slot *slots; int n, cap; } LCache;

typedef struct {
    Cfg c;
    shards S;
    int quant_bits;
    int xq;                /* experts on disk are a colibri container */
    Wt embed, lm_head;
    float *final_norm;
    Layer *L;
    LCache *cache;
    int64_t rb13, rb2;
    uint32_t **eusage;
    int npin;
    uint64_t clock, hits, miss;
    double t_fill, t_expert, t_shared, t_attn, t_route;
    float **K, **V; int kv_len, max_t;
    double dense_load_s;
    /* precomputed RoPE tables: [max_t][rot_dim/2] for cos, sin */
    float *rope_cos_full, *rope_sin_full;  /* [max_t * (hd*prf/2)] */
    float *rope_cos_swa,  *rope_sin_swa;
    int rope_dim_full, rope_dim_swa;  /* hd * partial_rotary_factor */
} Model;

/* ---------- utility ---------- */
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec*1e-9; }
#if defined(__APPLE__)
static double rss_gb(void) { struct rusage r; getrusage(RUSAGE_SELF, &r); return r.ru_maxrss / (1024.0*1024.0*1024.0); }
#else
static double rss_gb(void) { struct rusage r; getrusage(RUSAGE_SELF, &r); return r.ru_maxrss / (1024.0*1024.0); }
#endif
static float *falloc(int64_t n) { float *p = malloc(n*sizeof(float)); if(!p){fprintf(stderr,"OOM %ld\n",(long)n);exit(1);} return p; }
static float sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }
static float siluf(float x) { return x / (1.f + expf(-x)); }
static float softplusf(float x) { return x > 20.f ? x : logf(1.f + expf(x)); }
static int is_eos(const Cfg *c, int t) {
    for (int i = 0; i < c->n_eos; i++) if (c->eos_ids[i] == t) return 1;
    return 0;
}

/* ---------- matmuls (same as inkling.c) ---------- */
static void matmul(float *y, const float *x, const float *W, int S, int I, int O) {
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const float *w = W + (int64_t)o * I;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            float acc = 0.f;
            for (int i = 0; i < I; i++) acc += xs[i] * w[i];
            y[(int64_t)s * O + o] = acc;
        }
    }
}

#if defined(__AVX512BF16__) && defined(__AVX512F__)
#include <immintrin.h>
#define HAVE_BF16_DOT 1
#endif
#if defined(__AVX2__)
#include <immintrin.h>
#endif

static void matmul_h(float *y, const float *x, const uint16_t *W, int S, int I, int O) {
#ifdef HAVE_BF16_DOT
    if (I % 32 == 0) {
        uint16_t *xh = malloc((size_t)S * I * sizeof(uint16_t));
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            uint16_t *xd = xh + (int64_t)s * I;
            for (int i = 0; i < I; i += 32) {
                __m512 a = _mm512_loadu_ps(xs + i), b = _mm512_loadu_ps(xs + i + 16);
                _mm512_storeu_si512(xd + i, (__m512i)_mm512_cvtne2ps_pbh(b, a));
            }
        }
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < O; o++) {
            const uint16_t *w = W + (int64_t)o * I;
            for (int s = 0; s < S; s++) {
                const uint16_t *xs = xh + (int64_t)s * I;
                __m512 acc = _mm512_setzero_ps();
                for (int i = 0; i < I; i += 32)
                    acc = _mm512_dpbf16_ps(acc, (__m512bh)_mm512_loadu_si512(xs + i),
                                                (__m512bh)_mm512_loadu_si512(w + i));
                y[(int64_t)s * O + o] = _mm512_reduce_add_ps(acc);
            }
        }
        free(xh);
        return;
    }
#endif
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const uint16_t *w = W + (int64_t)o * I;
        for (int s = 0; s < S; s++) {
            const float *xs = x + (int64_t)s * I;
            float acc = 0.f;
            for (int i = 0; i < I; i++) {
                union { uint32_t u; float f; } v = { (uint32_t)w[i] << 16 };
                acc += xs[i] * v.f;
            }
            y[(int64_t)s * O + o] = acc;
        }
    }
}

#ifdef COLI_CUDA
/* GPU_VERIFY=1 is the CPU-parity gate for the GPU path: keep the host copy of
 * every uploaded tensor, run both implementations on every matmul, and report
 * the worst deviation at the end of a generation. Costs a doubled dense-weight
 * RSS and roughly doubled decode time, so it is opt-in, but it is the only
 * thing that distinguishes "the GPU produced a different answer" from "the GPU
 * produced a wrong answer" — the two look identical in the output text. */
static int    g_gpu_verify;
static double g_gv_max, g_gv_ref;
static long   g_gv_calls;
/* the expert path is checked separately: one counter for two very different
 * code paths cannot say which of them is wrong */
static double g_gv_exp_max, g_gv_exp_ref, g_gv_exp_num, g_gv_exp_den;
static long   g_gv_exp_calls;
/* bytes of dense weight actually resident in VRAM, for the dashboard */
static size_t g_vram_bytes;

/* Routed experts held in VRAM for the whole run, indexed [layer][expert]. When
 * this is on, the LRU slot cache is bypassed entirely: there is nothing to
 * evict and nothing to stream, so the fill phase stops existing rather than
 * getting faster. It is all-or-nothing per model -- a partially resident set
 * would need the cache back for the remainder, which is the shape the larger
 * Laguna models will want and this is not. */
static int    g_exp_vram;
static size_t g_exp_bytes, g_exp_count;
static ColiCudaTensor ***g_eg, ***g_eu, ***g_ed;
#endif

static void matmul_w(float *y, const float *x, Wt W, int S, int I, int O) {
#ifdef COLI_CUDA
    if (W.dev) {
        if (ink_cuda_matmul_bf16(y, x, W.dev, S, I, O) != 0) {
            fprintf(stderr, "cuda matmul failed and host copy was freed\n"); exit(1);
        }
        if (g_gpu_verify && W.h) {
            float *ref = malloc((size_t)S * O * 4);
            if (!ref) { fprintf(stderr, "GPU_VERIFY: OOM on %dx%d reference\n", S, O); exit(1); }
            matmul_h(ref, x, W.h, S, I, O);
            for (int i = 0; i < S * O; i++) {
                double d = fabs((double)y[i] - (double)ref[i]);
                if (d > g_gv_max) { g_gv_max = d; g_gv_ref = fabs((double)ref[i]); }
            }
            g_gv_calls++;
            free(ref);
        }
        return;
    }
#endif
    if (W.f) matmul(y, x, W.f, S, I, O);
    else     matmul_h(y, x, W.h, S, I, O);
}

#if defined(__AVX2__)
static inline __m256i i8dot_block(__m256i acc, __m256i a, __m256i b) {
    __m256i ax = _mm256_sign_epi8(a, a);
    __m256i sy = _mm256_sign_epi8(b, a);
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    return _mm256_dpbusd_epi32(acc, ax, sy);
#else
    __m256i p = _mm256_maddubs_epi16(ax, sy);
    return _mm256_add_epi32(acc, _mm256_madd_epi16(p, _mm256_set1_epi16(1)));
#endif
}
#endif
static void matmul_q(float *y, const float *x, const int8_t *q, const float *scale, int I, int O) {
#if defined(__AVX2__)
    static int idot = -1;
    if (idot < 0) { const char *e = getenv("IDOT"); idot = !(e && *e == '0'); }
    if (idot && I % 32 == 0 && I <= 8192) {
        int nb = I / 32;
        int8_t xi[8192]; float xs[256];
        for (int b = 0; b < nb; b++) {
            const float *xb = x + b*32;
            float am = 0.f; for (int i = 0; i < 32; i++) { float a = fabsf(xb[i]); if (a > am) am = a; }
            float s = am/127.f; if (s < 1e-12f) s = 1e-12f;
            xs[b] = s; float inv = 1.f/s;
            for (int i = 0; i < 32; i++) xi[b*32+i] = (int8_t)lrintf(xb[i]*inv);
        }
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < O; o++) {
            const int8_t *w = q + (int64_t)o * I;
            float acc = 0.f;
            for (int b = 0; b < nb; b++) {
                __m256i vacc = i8dot_block(_mm256_setzero_si256(),
                                           _mm256_loadu_si256((const __m256i*)(xi + b*32)),
                                           _mm256_loadu_si256((const __m256i*)(w + b*32)));
                __m128i lo = _mm256_castsi256_si128(vacc), hi = _mm256_extracti128_si256(vacc, 1);
                __m128i s4 = _mm_add_epi32(lo, hi);
                s4 = _mm_hadd_epi32(s4, s4); s4 = _mm_hadd_epi32(s4, s4);
                acc += xs[b] * (float)_mm_cvtsi128_si32(s4);
            }
            y[o] = acc * scale[o];
        }
        return;
    }
#endif
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const int8_t *w = q + (int64_t)o * I;
        float acc = 0.f;
        for (int i = 0; i < I; i++) acc += x[i] * (float)w[i];
        y[o] = acc * scale[o];
    }
}

static void matmul_q4(float *y, const float *x, const uint8_t *p, const float *scale, int I, int O) {
#if defined(__AVX2__)
    static int idot = -1;
    if (idot < 0) { const char *e = getenv("IDOT"); idot = !(e && *e == '0'); }
    if (idot && I % 32 == 0 && I <= 8192) {
        int nb = I / 32;
        int8_t xi[8192]; float xs[256];
        for (int b = 0; b < nb; b++) {
            const float *xb = x + b*32;
            float am = 0.f; for (int i = 0; i < 32; i++) { float a = fabsf(xb[i]); if (a > am) am = a; }
            float s = am/127.f; if (s < 1e-12f) s = 1e-12f;
            xs[b] = s; float inv = 1.f/s;
            for (int i = 0; i < 32; i++) xi[b*32+i] = (int8_t)lrintf(xb[i]*inv);
        }
        const __m128i m4 = _mm_set1_epi8(0x0F);
        const __m256i b8 = _mm256_set1_epi8(8);
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < O; o++) {
            const uint8_t *w = p + (int64_t)o * (I/2);
            float acc = 0.f;
            for (int b = 0; b < nb; b++) {
                __m128i by = _mm_loadu_si128((const __m128i*)(w + b*16));
                __m128i lo = _mm_and_si128(by, m4);
                __m128i hi = _mm_and_si128(_mm_srli_epi16(by, 4), m4);
                __m256i nib = _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi));
                nib = _mm256_sub_epi8(nib, b8);
                __m256i vacc = i8dot_block(_mm256_setzero_si256(),
                                           _mm256_loadu_si256((const __m256i*)(xi + b*32)), nib);
                __m128i l = _mm256_castsi256_si128(vacc), h = _mm256_extracti128_si256(vacc, 1);
                __m128i s4 = _mm_add_epi32(l, h);
                s4 = _mm_hadd_epi32(s4, s4); s4 = _mm_hadd_epi32(s4, s4);
                acc += xs[b] * (float)_mm_cvtsi128_si32(s4);
            }
            y[o] = acc * scale[o];
        }
        return;
    }
#endif
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const uint8_t *w = p + (int64_t)o * (I/2);
        float acc = 0.f;
        for (int i = 0; i < I; i += 2) {
            uint8_t byte = w[i/2];
            acc += x[i]   * (float)((int)(byte & 0xF) - 8);
            acc += x[i+1] * (float)((int)(byte >> 4)  - 8);
        }
        y[o] = acc * scale[o];
    }
}

static void quantize_rows(const float *w, int8_t *q, float *scale, int O, int I, int bits) {
    int qmax = (1 << (bits - 1)) - 1;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < O; o++) {
        const float *wr = w + (int64_t)o * I;
        float amax = 0.f; for (int i = 0; i < I; i++) { float a = fabsf(wr[i]); if (a > amax) amax = a; }
        float s = amax / qmax; if (s < 1e-8f) s = 1e-8f;
        scale[o] = s;
        int8_t *qr = q + (int64_t)o * I;
        for (int i = 0; i < I; i++) {
            int v = (int)lrintf(wr[i] / s);
            if (v >  qmax) v =  qmax;
            if (v < -qmax-1) v = -qmax-1;
            qr[i] = (int8_t)v;
        }
    }
}

/* rmsnorm (f64 accumulate, same as inkling) */
static void rmsnorm_row(float *out, const float *x, const float *w, int D, float eps) {
    double ms = 0; for (int i = 0; i < D; i++) ms += (double)x[i]*x[i];
    float r = 1.f / sqrtf((float)(ms / D) + eps);
    for (int i = 0; i < D; i++) out[i] = x[i] * r * w[i];
}

static void softmax_row(float *x, int n) {
    float m = -1e30f; for (int i = 0; i < n; i++) if (x[i] > m) m = x[i];
    float s = 0; for (int i = 0; i < n; i++) { x[i] = expf(x[i]-m); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

/* ---------- RoPE (non-interleaved rotate_half style, partial rotary) ----------
 * Unlike GLM's interleaved RoPE, Laguna uses the standard GPT-J style:
 *   q_rot = q[..., :rot_dim]; q_pass = q[..., rot_dim:]
 *   q_embed = cat(q_rot * cos + rotate_half(q_rot) * sin, q_pass)
 * where rotate_half([x0..x(d/2-1), x(d/2)..x(d-1)]) = [-x(d/2)..-x(d-1), x0..x(d/2-1)]
 * and cos/sin have length rot_dim = head_dim * partial_rotary_factor.
 *
 * For YaRN, inv_freq is corrected by the YaRN formula; attention_factor scales cos/sin.
 * We precompute the full [max_t][rot_dim] tables at init. */
static void rope_compute_inv_freq(float *inv_freq, int half, float theta, int rope_type,
                                    float factor, float beta_slow, float beta_fast,
                                    int orig_max, int head_dim, float partial_rotary) {
    int dim = (int)(head_dim * partial_rotary);
    /* dim should equal 2*half */
    for (int i = 0; i < half; i++)
        inv_freq[i] = 1.0f / powf(theta, (float)(2*i) / (float)dim);
    if (rope_type == 1) {  /* YaRN */
        /* HF formula: find_correction_range + linear_ramp_factor
         * low  = floor(dim * log(orig_max / (beta_fast * 2*pi)) / (2*log(theta)))
         * high = ceil (dim * log(orig_max / (beta_slow * 2*pi)) / (2*log(theta)))
         * ramp[i] = clamp((i - low) / (high - low), 0, 1)
         * inv_freq[i] = interp[i] * (1-ramp) + extrap[i] * ramp
         * where interp = 1/(factor * theta^(2i/dim)), extrap = 1/theta^(2i/dim) */
        float lf = (float)dim * logf((float)orig_max / (beta_fast * 2.0f * (float)M_PI)) / (2.0f * logf(theta));
        float hf = (float)dim * logf((float)orig_max / (beta_slow * 2.0f * (float)M_PI)) / (2.0f * logf(theta));
        int low = (int)floorf(lf); if (low < 0) low = 0;
        int high = (int)ceilf(hf); if (high > half - 1) high = half - 1;
        if (low == high) high += 1;  /* prevent singularity */
        float range = (float)(high - low);
        for (int i = 0; i < half; i++) {
            float extrap = 1.0f / powf(theta, (float)(2*i) / (float)dim);
            float interp = extrap / factor;
            float ramp = (float)(i - low) / range;
            if (ramp < 0) ramp = 0; if (ramp > 1) ramp = 1;
            /* HF: inv_freq = interp * ramp + extrap * (1 - ramp)
             * (inv_freq_extrapolation_factor = 1 - ramp) */
            inv_freq[i] = interp * ramp + extrap * (1.0f - ramp);
        }
    }
}

static void rope_build_tables(float **cos_p, float **sin_p, int max_t, int head_dim,
                               float partial_rotary, int rope_type, float theta,
                               float factor, float beta_slow, float beta_fast,
                               int orig_max, float attn_factor) {
    int dim = (int)(head_dim * partial_rotary);
    int half = dim / 2;
    float *inv_freq = malloc(half * sizeof(float));
    rope_compute_inv_freq(inv_freq, half, theta, rope_type, factor, beta_slow, beta_fast,
                           orig_max, head_dim, partial_rotary);
    *cos_p = falloc((int64_t)max_t * dim);
    *sin_p = falloc((int64_t)max_t * dim);
    for (int pos = 0; pos < max_t; pos++) {
        float *c = *cos_p + (int64_t)pos * dim;
        float *s = *sin_p + (int64_t)pos * dim;
        /* emb = cat(freqs, freqs) so cos/sin = [cs0..cs(half-1), cs0..cs(half-1)] */
        for (int i = 0; i < half; i++) {
            float ang = (float)pos * inv_freq[i];
            float cs = cosf(ang) * attn_factor;
            float sn = sinf(ang) * attn_factor;
            c[i]       = cs;
            c[half + i] = cs;
            s[i]       = sn;
            s[half + i] = sn;
        }
    }
    free(inv_freq);
}

/* apply RoPE to a single [hd]-dim vector at position pos, in-place.
 * rot_dim = hd * partial_rotary (for this layer type). cos/sin precomputed. */
static void rope_apply(float *v, int pos, int hd, int rot_dim,
                        const float *cos_t, const float *sin_t) {
    const float *c = cos_t + (int64_t)pos * rot_dim;
    const float *s = sin_t + (int64_t)pos * rot_dim;
    int half = rot_dim / 2;
    float tmp[256];  /* rot_dim <= 128 */
    memcpy(tmp, v, rot_dim * sizeof(float));
    /* rotate_half: [-tmp[half:], tmp[:half]] */
    for (int i = 0; i < half; i++) {
        v[i]       = tmp[i] * c[i] - tmp[half + i] * s[i];
        v[half + i] = tmp[half + i] * c[i] + tmp[i] * s[i];
    }
    /* v[rot_dim..hd] passes through unchanged */
}

/* ---------- config loading ---------- */
static double jnum(jval *o, const char *k, double dflt) {
    jval *v = json_get(o, k);
    return (v && v->t == J_NUM) ? v->num : dflt;
}

static void load_cfg(Cfg *c, const char *snap) {
    char path[2048]; snprintf(path, sizeof(path), "%s/config.json", snap);
    FILE *f = fopen(path, "rb"); if(!f){perror(path);exit(1);}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *buf = malloc(n+1); if(fread(buf,1,n,f)!=(size_t)n){} buf[n]=0; fclose(f);
    char *arena=NULL; jval *root = json_parse(buf, &arena);

    c->hidden      = (int)jnum(root,"hidden_size",2048);
    c->n_layers    = (int)jnum(root,"num_hidden_layers",48);
    c->vocab       = (int)jnum(root,"vocab_size",100352);
    c->n_heads     = (int)jnum(root,"num_attention_heads",32);
    c->n_kv        = (int)jnum(root,"num_key_value_heads",8);
    c->head_dim    = (int)jnum(root,"head_dim",128);
    c->window      = (int)jnum(root,"sliding_window",512);
    c->n_experts   = (int)jnum(root,"num_experts",256);
    c->topk        = (int)jnum(root,"num_experts_per_tok",8);
    c->n_shared    = 1;  /* Laguna always has 1 shared expert */
    c->eps         = (float)jnum(root,"rms_norm_eps",1e-6);
    c->route_scale = (float)jnum(root,"moe_routed_scaling_factor",1.0);
    jval *nt = json_get(root,"norm_topk_prob");
    c->norm_topk = (nt && nt->t == J_BOOL) ? (int)nt->boolean : 1;
    c->dense_inter = (int)jnum(root,"intermediate_size",8192);
    c->moe_inter   = (int)jnum(root,"moe_intermediate_size",1024);
    c->shared_inter= (int)jnum(root,"shared_expert_intermediate_size",1024);
    c->swa_heads   = (int)jnum(root,"num_attention_heads",c->n_heads);
    /* Some configs use swa_num_attention_heads */
    jval *sh = json_get(root,"swa_num_attention_heads");
    if (sh && sh->t == J_NUM) c->swa_heads = (int)sh->num;
    c->swa_kv  = c->n_kv;
    c->swa_hd  = c->head_dim;

    /* eos + bos; every listed eos id ends generation (24 = </assistant> is the turn end) */
    jval *eo = json_get(root,"eos_token_id");
    c->n_eos = 0;
    if (eo) {
        if (eo->t == J_NUM) c->eos_ids[c->n_eos++] = (int)eo->num;
        else if (eo->t == J_ARR)
            for (int i = 0; i < eo->len && c->n_eos < 8; i++)
                c->eos_ids[c->n_eos++] = (int)eo->kids[i]->num;
    }
    jval *bo = json_get(root,"bos_token_id");
    c->bos = bo && bo->t == J_NUM ? (int)bo->num : -1;

    /* layer_types: "full_attention" / "sliding_attention" */
    jval *lt = json_get(root,"layer_types");
    for (int i = 0; i < c->n_layers; i++) {
        if (lt && lt->t == J_ARR && i < lt->len)
            c->local[i] = (strcmp(lt->kids[i]->str,"sliding_attention")==0);
        else
            c->local[i] = 0;  /* default: all global */
    }

    /* mlp_layer_types: "dense" / "sparse" */
    jval *mt = json_get(root,"mlp_layer_types");
    jval *ml = json_get(root,"mlp_only_layers");
    for (int i = 0; i < c->n_layers; i++) {
        if (mt && mt->t == J_ARR && i < mt->len)
            c->sparse[i] = (strcmp(mt->kids[i]->str,"sparse")==0);
        else if (ml && ml->t == J_ARR) {
            c->sparse[i] = 1;
            for (int j = 0; j < ml->len; j++)
                if ((int)ml->kids[j]->num == i) { c->sparse[i] = 0; break; }
        } else
            c->sparse[i] = (i > 0);  /* default: layer 0 dense, rest sparse */
    }

    /* per-layer head counts */
    jval *hpl = json_get(root,"num_attention_heads_per_layer");
    for (int i = 0; i < c->n_layers; i++)
        c->heads_per_layer[i] = (hpl && hpl->t == J_ARR && i < hpl->len)
            ? (int)hpl->kids[i]->num : c->n_heads;

    /* RoPE parameters: nested dict with "full_attention" and "sliding_attention" */
    jval *rp = json_get(root,"rope_parameters");
    /* defaults */
    c->rope_type_full = 0; c->rope_theta_full = 10000.0f;
    c->partial_rotary_full = 1.0f;
    c->yarn_factor_full = 1.0f; c->yarn_beta_slow_full = 1.0f;
    c->yarn_beta_fast_full = 32.0f; c->yarn_attn_factor_full = 1.0f;
    c->yarn_orig_max_full = 8192;
    c->rope_type_swa = 0; c->rope_theta_swa = 10000.0f;
    c->partial_rotary_swa = 1.0f;

    if (rp && rp->t == J_OBJ) {
        /* full_attention sub-dict (or top-level if not nested) */
        jval *fa = json_get(rp,"full_attention");
        jval *base = fa ? fa : rp;
        jval *rt = json_get(base,"rope_type");
        if (rt && rt->t == J_STR && !strcmp(rt->str,"yarn")) c->rope_type_full = 1;
        c->rope_theta_full = (float)jnum(base,"rope_theta",10000.0);
        c->partial_rotary_full = (float)jnum(base,"partial_rotary_factor",1.0);
        if (c->rope_type_full) {
            c->yarn_factor_full = (float)jnum(base,"factor",1.0);
            c->yarn_beta_slow_full = (float)jnum(base,"beta_slow",1.0);
            c->yarn_beta_fast_full = (float)jnum(base,"beta_fast",32.0);
            c->yarn_attn_factor_full = (float)jnum(base,"attention_factor",1.0);
            c->yarn_orig_max_full = (int)jnum(base,"original_max_position_embeddings",8192);
        }
        /* sliding_attention sub-dict */
        jval *sa = json_get(rp,"sliding_attention");
        if (sa) {
            jval *srt = json_get(sa,"rope_type");
            if (srt && srt->t == J_STR && !strcmp(srt->str,"yarn")) c->rope_type_swa = 1;
            c->rope_theta_swa = (float)jnum(sa,"rope_theta",10000.0);
            c->partial_rotary_swa = (float)jnum(sa,"partial_rotary_factor",1.0);
        }
    }
    if (c->n_layers > MAXL) { fprintf(stderr,"n_layers %d > MAXL\n", c->n_layers); exit(1); }
    free(buf); free(arena);
}

/* ---------- weight loading ---------- */
static float *load_t(Model *m, const char *name) {
    int64_t n = st_numel(&m->S, name);
    if (n < 0) { fprintf(stderr, "missing %s\n", name); exit(1); }
    float *p = falloc(n);
    st_read_f32(&m->S, name, p, 0);
    return p;
}
static float load_scalar(Model *m, const char *name, float dflt) {
    if (!st_has(&m->S, name)) return dflt;
    float v; st_read_f32(&m->S, name, &v, 0); return v;
}

static void pread_all(int fd, void *buf, int64_t nb, int64_t off) {
    char *p = buf;
    while (nb > 0) {
        int64_t chunk = nb < (1<<30) ? nb : (1<<30);
        ssize_t got = pread(fd, p, (size_t)chunk, off);
        if (got <= 0) { perror("pread chunk"); exit(1); }
        p += got; off += got; nb -= got;
    }
}

static Wt load_w(Model *m, const char *name, int gpu_ok) {
    Wt w = {0};
    st_tensor *t = st_find(&m->S, name);
    if (!t) { fprintf(stderr, "missing %s\n", name); exit(1); }
    if (t->dtype == 0) {  /* bf16/f16 on disk */
        w.h = malloc(t->nbytes); if (!w.h) { fprintf(stderr,"OOM %s\n",name); exit(1); }
        pread_all(t->fd, w.h, t->nbytes, t->off);
#ifdef COLI_CUDA
        if (g_cuda && gpu_ok && ink_cuda_free_bytes() > (size_t)t->nbytes + (3ULL<<30)) {
            w.dev = ink_cuda_upload(w.h, t->nbytes);
            if (w.dev) {
                g_vram_bytes += (size_t)t->nbytes;
                if (!g_gpu_verify) { free(w.h); w.h = NULL; }
            }
        }
#else
        (void)gpu_ok;
#endif
    } else {
        w.f = falloc(t->numel);
        st_read_f32(&m->S, name, w.f, 0);
    }
    return w;
}
static Wt wt_off(Wt w, int64_t off) {
    Wt r = { w.f ? w.f + off : NULL, w.h ? w.h + off : NULL,
             w.dev ? (char*)w.dev + off*2 : NULL };
    return r;
}
static void wt_row_f32(Wt w, int64_t off, float *out, int n) {
    if (w.f) memcpy(out, w.f + off, n * sizeof(float));
    else for (int i = 0; i < n; i++) { union { uint32_t u; float f; } v = { (uint32_t)w.h[off + i] << 16 }; out[i] = v.f; }
}

static void read_f32_slice(shards *S, const char *name, float *out, int64_t off, int64_t cnt) {
    st_tensor *t = st_find(S, name);
    if (!t) { fprintf(stderr, "missing tensor: %s\n", name); exit(1); }
    if (t->dtype == 3) { fprintf(stderr, "%s: U8 container has no f32 view\n", name); exit(1); }
    int esz = (t->dtype == 2) ? 4 : 2;
    void *raw = malloc((size_t)cnt * esz);
    if (!raw) { fprintf(stderr,"OOM slice %s\n",name); exit(1); }
    if (pread(t->fd, raw, (size_t)cnt*esz, t->off + off*esz) != (ssize_t)(cnt*esz)) { perror("pread slice"); exit(1); }
    if (t->dtype == 2) memcpy(out, raw, (size_t)cnt*4);
    else if (t->dtype == 0) { uint16_t *p = raw; for (int64_t i = 0; i < cnt; i++) out[i] = bf16_to_f32(p[i]); }
    else                    { uint16_t *p = raw; for (int64_t i = 0; i < cnt; i++) out[i] = f16_to_f32(p[i]); }
    free(raw);
    posix_fadvise(t->fd, t->off + off*esz, cnt*esz, POSIX_FADV_DONTNEED);
}

static void read_u8_slice(shards *S, const char *name, uint8_t *out, int64_t boff, int64_t nb) {
    st_tensor *t = st_find(S, name);
    if (!t) { fprintf(stderr, "missing tensor: %s\n", name); exit(1); }
    if (pread(t->fd, out, (size_t)nb, t->off + boff) != (ssize_t)nb) { perror("pread u8 slice"); exit(1); }
    posix_fadvise(t->fd, t->off + boff, nb, POSIX_FADV_DONTNEED);
}

static void unpack_rows(const uint8_t *raw, int8_t *q, int64_t rows, int64_t cols, int64_t rowb) {
    if (rowb == cols) { memcpy(q, raw, (size_t)(rows*cols)); return; }
    if (rowb*2 != cols) { fprintf(stderr, "container row size %ld vs cols %ld unsupported\n", (long)rowb, (long)cols); exit(1); }
    for (int64_t r = 0; r < rows; r++) {
        const uint8_t *b = raw + r*rowb;
        int8_t *qr = q + r*cols;
        for (int64_t j = 0; j < rowb; j++) {
            qr[2*j]   = (int8_t)((b[j] & 0xF) - 8);
            qr[2*j+1] = (int8_t)((b[j] >> 4) - 8);
        }
    }
}

static double mem_avail_bytes(void) {
#if defined(__linux__)
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char ln[256]; double kb = 0;
    while (fgets(ln, sizeof(ln), f)) if (sscanf(ln, "MemAvailable: %lf", &kb) == 1) break;
    fclose(f);
    return kb * 1024.0;
#else
    return 0;
#endif
}

#ifdef COLI_CUDA
/* Upload every routed expert to VRAM, or none.
 *
 * The container's int4 layout is byte-identical to backend_cuda.cu's fmt=2 --
 * verified against a double-precision decode of a real expert, with a
 * nibble-swapped null to prove the check discriminates: tests/
 * test_expert_int4_compat.c. So the bytes go straight from pread to cudaMalloc
 * with no repack. gate and up are the contiguous first and second halves of
 * gate_up, which is why they can be uploaded as two tensors from one read.
 *
 * Returns 1 if every expert made it. On any failure the partial set is freed
 * and the engine keeps its CPU cache path: half-resident is not a state this
 * code knows how to run. */
static int experts_upload(Model *m, int dev) {
	Cfg *c = &m->c;
	int64_t D = c->hidden, I = c->moe_inter, E = c->n_experts;
	int64_t wg = I * m->rb13, wd = D * m->rb2;   /* per-expert device bytes, per tensor */
	int nsp = 0;
	for (int i = 0; i < c->n_layers; i++) if (c->sparse[i]) nsp++;

	size_t need = (size_t)nsp * E * (2 * wg + wd)               /* weights  */
	            + (size_t)nsp * E * (2 * I + D) * sizeof(float); /* scales   */
	size_t freeb = 0, totb = 0;
	if (coli_cuda_mem_info(dev, &freeb, &totb) != 1) return 0;
	/* Leave headroom for the backend's own activation scratch and for the dense
	 * tensors the ink backend holds; running the card to the last byte turns a
	 * capacity problem into a mid-decode allocation failure. */
	size_t margin = 2ULL << 30;
	if (need + margin > freeb) {
		fprintf(stderr, "[cuda] experts stay on CPU: need %.1f GB + %.1f GB margin, %.1f GB free\n",
		        need / 1e9, margin / 1e9, freeb / 1e9);
		return 0;
	}

	g_eg = calloc(c->n_layers, sizeof(*g_eg));
	g_eu = calloc(c->n_layers, sizeof(*g_eu));
	g_ed = calloc(c->n_layers, sizeof(*g_ed));
	uint8_t *w13 = malloc((size_t)(2 * wg)), *w2 = malloc((size_t)wd);
	float *s13 = malloc((size_t)(2 * I) * sizeof(float)), *s2 = malloc((size_t)D * sizeof(float));
	if (!g_eg || !g_eu || !g_ed || !w13 || !w2 || !s13 || !s2) { fprintf(stderr, "[cuda] OOM staging experts\n"); return 0; }

	double t0 = now_s();
	int ok = 1;
	for (int layer = 0; layer < c->n_layers && ok; layer++) {
		if (!c->sparse[layer]) continue;
		g_eg[layer] = calloc(E, sizeof(**g_eg));
		g_eu[layer] = calloc(E, sizeof(**g_eu));
		g_ed[layer] = calloc(E, sizeof(**g_ed));
		if (!g_eg[layer] || !g_eu[layer] || !g_ed[layer]) { ok = 0; break; }
		char nm[320], qs[340];
		for (int64_t e = 0; e < E && ok; e++) {
			snprintf(nm, sizeof(nm), "model.layers.%d.mlp.experts.gate_up_proj", layer);
			read_u8_slice(&m->S, nm, w13, e * 2 * I * m->rb13, 2 * I * m->rb13);
			snprintf(qs, sizeof(qs), "%s.qs", nm);
			read_f32_slice(&m->S, qs, s13, e * 2 * I, 2 * I);
			snprintf(nm, sizeof(nm), "model.layers.%d.mlp.experts.down_proj", layer);
			read_u8_slice(&m->S, nm, w2, e * D * m->rb2, D * m->rb2);
			snprintf(qs, sizeof(qs), "%s.qs", nm);
			read_f32_slice(&m->S, qs, s2, e * D, D);
			/* gate = rows [0,I) of gate_up, up = rows [I,2I): contiguous halves */
			ok &= coli_cuda_tensor_upload(&g_eg[layer][e], w13,      s13,     2, (int)D, (int)I, dev);
			ok &= coli_cuda_tensor_upload(&g_eu[layer][e], w13 + wg, s13 + I, 2, (int)D, (int)I, dev);
			ok &= coli_cuda_tensor_upload(&g_ed[layer][e], w2,       s2,      2, (int)I, (int)D, dev);
			if (ok) { g_exp_count++; g_exp_bytes += (size_t)(2 * wg + wd) + (size_t)(2 * I + D) * sizeof(float); }
		}
		if ((layer & 7) == 1)
			fprintf(stderr, "\r[cuda] uploading experts: %zu/%d", g_exp_count, nsp * (int)E);
	}
	free(w13); free(w2); free(s13); free(s2);
	if (!ok) {
		fprintf(stderr, "\n[cuda] expert upload failed after %zu; falling back to the CPU cache\n", g_exp_count);
		for (int layer = 0; layer < c->n_layers; layer++)
			for (int64_t e = 0; g_eg[layer] && e < E; e++) {
				if (g_eg[layer][e]) coli_cuda_tensor_free(g_eg[layer][e]);
				if (g_eu[layer][e]) coli_cuda_tensor_free(g_eu[layer][e]);
				if (g_ed[layer][e]) coli_cuda_tensor_free(g_ed[layer][e]);
			}
		g_exp_count = g_exp_bytes = 0;
		return 0;
	}
	fprintf(stderr, "\r[cuda] %zu experts resident, %.1f GB, in %.0fs\n",
	        g_exp_count, g_exp_bytes / 1e9, now_s() - t0);
	return 1;
}
#endif

static void model_init(Model *m, const char *snap, int cap, int bits) {
    memset(m, 0, sizeof(*m));
    m->quant_bits = bits;
    load_cfg(&m->c, snap);
    st_init(&m->S, snap);
    Cfg *c = &m->c;
    double t0 = now_s();
#ifdef COLI_CUDA
    /* Same switches the rest of the project uses, so `coli --gpu none` and
     * COLI_GPU/COLI_GPUS reach this engine too. Only one device is used here
     * (the backend has no multi-GPU split), so a COLI_GPUS list means "the
     * first of these"; GPU_DEV stays as the engine-local override. */
    const char *gpus = getenv("COLI_GPUS") ? getenv("COLI_GPUS") : getenv("COLI_GPU");
    int gpu_off = getenv("NOGPU")
                || (getenv("COLI_CUDA") && !atoi(getenv("COLI_CUDA")))
                || (gpus && !strcmp(gpus, "none"));
    if (!gpu_off) {
        g_gpu_verify = getenv("GPU_VERIFY") && atoi(getenv("GPU_VERIFY"));
        int dev = 0;
        if (getenv("GPU_DEV"))                   dev = atoi(getenv("GPU_DEV"));
        else if (gpus && strcmp(gpus, "auto"))   dev = atoi(gpus);
        if (ink_cuda_init(dev) == 0) {
            char gname[128]; ink_cuda_device_name(gname, sizeof(gname));
            g_cuda = 1;
            /* name the silicon, not just the ordinal: with two GPUs installed,
             * device 0 is whichever one the runtime enumerated first. */
            fprintf(stderr, "[cuda] device %d = %s, %.1f GB free\n",
                    dev, gname[0] ? gname : "unknown", ink_cuda_free_bytes()/1e9);
        } else fprintf(stderr, "[cuda] init failed, running on CPU\n");
    }
#endif
    m->embed     = load_w(m, "model.embed_tokens.weight", 0);
    m->final_norm = load_t(m, "model.norm.weight");
    m->lm_head  = load_w(m, "lm_head.weight", 1);
    m->L = calloc(c->n_layers, sizeof(Layer));
    char nm[320];
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        #define LD(field, suffix)  snprintf(nm,sizeof(nm),"model.layers.%d." suffix,i); l->field = load_t(m,nm)
        #define LDW(field, suffix) snprintf(nm,sizeof(nm),"model.layers.%d." suffix,i); l->field = load_w(m,nm,1)
        LD(in_ln,  "input_layernorm.weight");
        LD(post_ln,"post_attention_layernorm.weight");
        LDW(q, "self_attn.q_proj.weight"); LDW(k, "self_attn.k_proj.weight");
        LDW(v, "self_attn.v_proj.weight"); LDW(g, "self_attn.g_proj.weight");
        LDW(o, "self_attn.o_proj.weight");
        LD(q_norm,"self_attn.q_norm.weight"); LD(k_norm,"self_attn.k_norm.weight");
        if (!c->sparse[i]) {
            LDW(dg, "mlp.gate_proj.weight"); LDW(du, "mlp.up_proj.weight"); LDW(dd, "mlp.down_proj.weight");
        } else {
            LD(router, "mlp.gate.weight");
            LDW(sh_g, "mlp.shared_expert.gate_proj.weight");
            LDW(sh_u, "mlp.shared_expert.up_proj.weight");
            LDW(sh_d, "mlp.shared_expert.down_proj.weight");
        }
        #undef LD
        #undef LDW
    }
    /* container detection */
    int64_t I = c->moe_inter, E = c->n_experts;
    for (int i = 0; i < c->n_layers; i++) if (c->sparse[i]) {
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.gate_up_proj",i);
        st_tensor *t = st_find(&m->S, nm);
        if (t && t->dtype == 3) {
            m->xq = 1;
            m->rb13 = t->nbytes / (E * 2*I);
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.down_proj",i);
            st_tensor *t2 = st_find(&m->S, nm);
            m->rb2 = t2->nbytes / (E * (int64_t)c->hidden);
            if (m->rb13 != c->hidden && m->rb13*2 != c->hidden) {
                fprintf(stderr,"unsupported container row size %lld\n",(long long)m->rb13); exit(1); }
        }
        break;
    }
    int nsp = 0; for (int i = 0; i < c->n_layers; i++) nsp += c->sparse[i];
    int64_t slotb = m->xq ? m->rb13*2*I + m->rb2*c->hidden + (2*I+c->hidden)*4
                  : m->quant_bits ? 3*I*c->hidden + (2*I+c->hidden)*4 : 3*I*c->hidden*4;
#ifdef COLI_CUDA
    /* Try residency before sizing the RAM cache: if every expert lives in VRAM
     * the cache is dead weight, and on this model it would otherwise reserve
     * ~16 GB of host RAM to hold a copy of what the GPU already has. */
    if (g_cuda && m->xq && m->rb13 * 2 == c->hidden &&
        (!getenv("EXPERTS_VRAM") || atoi(getenv("EXPERTS_VRAM")))) {
        int dev = getenv("GPU_DEV") ? atoi(getenv("GPU_DEV")) : 0;
        int devs[1] = { dev };
        if (coli_cuda_init(devs, 1) > 0 && experts_upload(m, dev)) g_exp_vram = 1;
        else coli_cuda_shutdown();
    }
    /* With every expert resident the RAM cache is dead weight, so it shrinks to
     * nothing -- EXCEPT under GPU_VERIFY, where the CPU chain is recomputed for
     * comparison. That needs a slot per DISTINCT expert routed across the whole
     * batch, not just per token: pass 1 acquires for every (token, k) before
     * pass 2 fills any of them, so a cache smaller than that aliases earlier
     * entries onto later fills and the checker reports its own eviction as an
     * engine fault. It did, twice. Hence the full expert set, and hence
     * GPU_VERIFY costing the host RAM this mode exists to save. */
    if (g_exp_vram) cap = g_gpu_verify ? c->n_experts : 1;
#endif
    if (cap <= 0) {
        double avail = mem_avail_bytes();
        cap = avail > 0 ? (int)((avail*0.80 - 4e9) / ((double)slotb * (nsp ? nsp : 1))) : 16;
        if (cap < 4) cap = 4;
        if (cap > c->n_experts) cap = c->n_experts;
        fprintf(stderr, "[cap auto] %d experts/layer (%.1f GB cache budget)\n",
                cap, (double)cap*slotb*nsp/1e9);
    }
    m->cache = calloc(c->n_layers, sizeof(LCache));
    for (int i = 0; i < c->n_layers; i++) { m->cache[i].cap = cap; m->cache[i].slots = calloc(cap, sizeof(Slot)); }
    m->eusage = calloc(c->n_layers, sizeof(uint32_t*));
    for (int i = 0; i < c->n_layers; i++) if (c->sparse[i]) m->eusage[i] = calloc(E, 4);
    m->dense_load_s = now_s() - t0;
}

/* ---------- expert slot management (same as inkling) ---------- */
static Slot *slot_find(Model *m, int layer, int eid) {
    LCache *lc = &m->cache[layer];
    for (int i = 0; i < lc->n; i++) if (lc->slots[i].eid == eid) {
        lc->slots[i].used = ++m->clock;
        return &lc->slots[i];
    }
    return NULL;
}

static Slot *slot_acquire(Model *m, int layer, int eid) {
    LCache *lc = &m->cache[layer]; Cfg *c = &m->c;
    int64_t D = c->hidden, I = c->moe_inter, n13 = 2*I*D, n2 = D*I;
    Slot *s;
    if (lc->n < lc->cap) {
        s = &lc->slots[lc->n++];
        if (m->xq)              { s->p13 = malloc((size_t)(m->rb13*2*I)); s->p2 = malloc((size_t)(m->rb2*D));
                                  s->s13 = falloc(2*I); s->s2 = falloc(D);
                                  if (!s->p13 || !s->p2) { fprintf(stderr,"OOM expert slot\n"); exit(1); } }
        else if (m->quant_bits) { s->q13 = malloc(n13); s->q2 = malloc(n2);
                                  s->s13 = falloc(2*I); s->s2 = falloc(D);
                                  if (!s->q13 || !s->q2) { fprintf(stderr,"OOM expert slot\n"); exit(1); } }
        else                    { s->f13 = falloc(n13); s->f2 = falloc(n2); }
    } else {
        int lru = -1;
        for (int i = 0; i < lc->n; i++)
            if (!lc->slots[i].pinned && (lru < 0 || lc->slots[i].used < lc->slots[lru].used)) lru = i;
        if (lru < 0) { fprintf(stderr, "layer %d: cache cap %d entirely pinned\n", layer, lc->cap); exit(1); }
        s = &lc->slots[lru];
    }
    s->eid = eid; s->used = ++m->clock; s->filled = 0; s->pinned = 0;
    return s;
}

static void slot_fill(Model *m, int layer, Slot *s) {
    Cfg *c = &m->c;
    int64_t D = c->hidden, I = c->moe_inter, n13 = 2*I*D, n2 = D*I;
    int64_t eid = s->eid;
    char nm[320], qs[340];
    if (m->xq) {
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.gate_up_proj",layer);
        read_u8_slice(&m->S, nm, s->p13, eid*2*I*m->rb13, 2*I*m->rb13);
        snprintf(qs,sizeof(qs),"%s.qs",nm);
        read_f32_slice(&m->S, qs, s->s13, eid*2*I, 2*I);
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.down_proj",layer);
        read_u8_slice(&m->S, nm, s->p2, eid*D*m->rb2, D*m->rb2);
        snprintf(qs,sizeof(qs),"%s.qs",nm);
        read_f32_slice(&m->S, qs, s->s2, eid*D, D);
    } else if (m->quant_bits) {
        float *tmp = falloc(n13 > n2 ? n13 : n2);
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.gate_up_proj",layer);
        if (!st_has(&m->S, nm)) {
            /* unfused HF format: separate gate_proj + up_proj */
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.%lld.gate_proj.weight",layer,(long long)eid);
            read_f32_slice(&m->S, nm, tmp, 0, I*D);
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.%lld.up_proj.weight",layer,(long long)eid);
            read_f32_slice(&m->S, nm, tmp + I*D, 0, I*D);
        } else {
            read_f32_slice(&m->S, nm, tmp, eid*n13, n13);
        }
        quantize_rows(tmp, s->q13, s->s13, 2*I, D, m->quant_bits);
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.down_proj",layer);
        if (!st_has(&m->S, nm)) {
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.%lld.down_proj.weight",layer,(long long)eid);
            read_f32_slice(&m->S, nm, tmp, 0, D*I);
        } else {
            read_f32_slice(&m->S, nm, tmp, eid*n2, n2);
        }
        quantize_rows(tmp, s->q2, s->s2, D, I, m->quant_bits);
        free(tmp);
    } else {
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.gate_up_proj",layer);
        if (!st_has(&m->S, nm)) {
            /* unfused HF format: separate gate_proj + up_proj */
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.%lld.gate_proj.weight",layer,(long long)eid);
            read_f32_slice(&m->S, nm, s->f13, 0, I*D);
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.%lld.up_proj.weight",layer,(long long)eid);
            read_f32_slice(&m->S, nm, s->f13 + I*D, 0, I*D);
        } else {
            read_f32_slice(&m->S, nm, s->f13, eid*n13, n13);
        }
        snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.down_proj",layer);
        if (!st_has(&m->S, nm)) {
            snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.%lld.down_proj.weight",layer,(long long)eid);
            read_f32_slice(&m->S, nm, s->f2, 0, D*I);
        } else {
            read_f32_slice(&m->S, nm, s->f2, eid*n2, n2);
        }
    }
    s->filled = 1;
}

static void pins_load(Model *m, const char *snap) {
    Cfg *c = &m->c; int E = c->n_experts;
    char up[2048];
    const char *env = getenv("PIN");
    if (env && (!strcmp(env, "off") || !strcmp(env, "0"))) { return; }
    if (env) snprintf(up, sizeof(up), "%s", env);
    else snprintf(up, sizeof(up), "%s/.coli_usage", snap);
    FILE *f = fopen(up, "rb");
    if (!f) return;
    uint32_t hdr[3];
    if (fread(hdr, 4, 3, f) != 3 || hdr[0] != 0x31554B49u ||
        (int)hdr[1] != c->n_layers || (int)hdr[2] != E) { fclose(f); return; }
    int cap = m->cache[0].cap;
    m->npin = getenv("PIN_N") ? atoi(getenv("PIN_N")) : cap/2;
    if (m->npin > cap - 8) m->npin = cap - 8;
    if (m->npin < 0) m->npin = 0;
    uint32_t *tmp = malloc((size_t)E * 4);
    Slot **ps = malloc((size_t)c->n_layers * m->npin * sizeof(Slot*));
    int *pl = malloc((size_t)c->n_layers * m->npin * sizeof(int));
    int np = 0;
    for (int i = 0; i < c->n_layers; i++) {
        if (fread(tmp, 4, E, f) != (size_t)E) break;
        if (!c->sparse[i] || !m->npin) continue;
        memcpy(m->eusage[i], tmp, (size_t)E * 4);
        for (int r = 0; r < m->npin; r++) {
            int best = -1; uint32_t bv = 0;
            for (int e = 0; e < E; e++) {
                int taken = 0;
                for (int z = 0; z < r; z++) if (ps[np-r+z]->eid == e) { taken = 1; break; }
                if (!taken && tmp[e] >= bv && tmp[e] > 0) { bv = tmp[e]; best = e; }
            }
            if (best < 0) break;
            Slot *s = slot_acquire(m, i, best);
            s->pinned = 1;
            ps[np] = s; pl[np] = i; np++;
        }
    }
    fclose(f);
    if (np) {
        double t0 = now_s();
        #pragma omp parallel for schedule(dynamic,1)
        for (int j = 0; j < np; j++) slot_fill(m, pl[j], ps[j]);
        fprintf(stderr, "[pin] %d experts pinned (%d/layer) from %s in %.1fs\n",
                np, m->npin, up, now_s()-t0);
    }
    free(tmp); free(ps); free(pl);
}

static int usage_save(Model *m, const char *snap) {
    Cfg *c = &m->c; int E = c->n_experts;
    char up[2048], tp[2060];
    const char *env = getenv("PIN");
    const char *sv = getenv("USAGE_SAVE");
    if (sv && *sv == '0') return 0;
    if (env && (!strcmp(env, "off") || !strcmp(env, "0"))) return 0;
    if (env) snprintf(up, sizeof(up), "%s", env);
    else snprintf(up, sizeof(up), "%s/.coli_usage", snap);
    snprintf(tp, sizeof(tp), "%s.tmp", up);
    FILE *f = fopen(tp, "wb");
    if (!f) return 0;
    uint32_t hdr[3] = { 0x31554B49u, (uint32_t)c->n_layers, (uint32_t)E };
    fwrite(hdr, 4, 3, f);
    uint32_t *zero = calloc(E, 4);
    for (int i = 0; i < c->n_layers; i++)
        fwrite(m->eusage[i] ? m->eusage[i] : zero, 4, E, f);
    free(zero); fclose(f);
    return rename(tp, up) == 0;
}

/* ---------- attention (GQA + sliding/global + QK-RMSNorm + RoPE + softplus gate) ---------- */
static void attention(Model *m, Layer *l, int li, float *x, int S, int pos0, float *out) {
    Cfg *c = &m->c;
    int D = c->hidden, H = L_HEADS(c,li), KV = L_KV(c,li), hd = L_HD(c,li);
    int local = c->local[li];
    int qdim = H*hd, kvdim = KV*hd, group = H/KV;
    int rot_dim = (int)(hd * (local ? c->partial_rotary_swa : c->partial_rotary_full));
    const float *cos_t = local ? m->rope_cos_swa : m->rope_cos_full;
    const float *sin_t = local ? m->rope_sin_swa : m->rope_sin_full;

    float *q  = falloc((int64_t)S*qdim);
    float *k  = falloc((int64_t)S*kvdim);
    float *vv = falloc((int64_t)S*kvdim);
    float *gate = falloc((int64_t)S*H);
    matmul_w(q,  x, l->q, S, D, qdim);
    matmul_w(k,  x, l->k, S, D, kvdim);
    matmul_w(vv, x, l->v, S, D, kvdim);
    matmul_w(gate, x, l->g, S, D, H);  /* per-head gate (before softplus) */

    /* QK RMSNorm (per-head, before RoPE) */
    for (int s = 0; s < S; s++) {
        for (int h = 0; h < H;  h++) rmsnorm_row(q + (int64_t)s*qdim  + h*hd, q + (int64_t)s*qdim  + h*hd, l->q_norm, hd, c->eps);
        for (int h = 0; h < KV; h++) rmsnorm_row(k + (int64_t)s*kvdim + h*hd, k + (int64_t)s*kvdim + h*hd, l->k_norm, hd, c->eps);
    }

    /* RoPE: apply to q and k (partial rotary) */
    for (int s = 0; s < S; s++) {
        int pos = pos0 + s;
        for (int h = 0; h < H; h++)  rope_apply(q + (int64_t)s*qdim + h*hd, pos, hd, rot_dim, cos_t, sin_t);
        for (int h = 0; h < KV; h++) rope_apply(k + (int64_t)s*kvdim + h*hd, pos, hd, rot_dim, cos_t, sin_t);
    }

    /* append K,V to cache */
    for (int s = 0; s < S; s++) for (int h = 0; h < KV; h++) {
        int t = pos0 + s;
        memcpy(m->K[li] + ((int64_t)h*m->max_t + t)*hd, k  + (int64_t)s*kvdim + h*hd, hd*sizeof(float));
        memcpy(m->V[li] + ((int64_t)h*m->max_t + t)*hd, vv + (int64_t)s*kvdim + h*hd, hd*sizeof(float));
    }

    /* softplus gate: gate[h] = softplus(g_proj(x)[h]) */
    for (int64_t i = 0; i < (int64_t)S*H; i++) gate[i] = softplusf(gate[i]);

    float scale = 1.f / sqrtf((float)hd);
    float *ctx = falloc((int64_t)S*qdim);
    #pragma omp parallel
    {
        float *sc = malloc((size_t)m->max_t * sizeof(float));
        #pragma omp for collapse(2) schedule(static)
        for (int h = 0; h < H; h++) {
            for (int s = 0; s < S; s++) {
                int qpos = pos0 + s;
                int t0 = local && qpos - c->window + 1 > 0 ? qpos - c->window + 1 : 0;
                const float *qv = q + (int64_t)s*qdim + h*hd;
                const float *Kh = m->K[li] + ((int64_t)(h/group)*m->max_t)*hd;
                for (int t = t0; t <= qpos; t++) {
                    const float *kv = Kh + (int64_t)t*hd;
                    float acc = 0.f;
                    for (int d = 0; d < hd; d++) acc += qv[d]*kv[d];
                    sc[t - t0] = acc * scale;
                }
                int n = qpos - t0 + 1;
                softmax_row(sc, n);
                float *cx = ctx + (int64_t)s*qdim + h*hd;
                for (int d = 0; d < hd; d++) cx[d] = 0.f;
                const float *Vh = m->V[li] + ((int64_t)(h/group)*m->max_t)*hd;
                for (int t = t0; t <= qpos; t++) {
                    const float *vrow = Vh + (int64_t)t*hd;
                    float a = sc[t - t0];
                    for (int d = 0; d < hd; d++) cx[d] += a * vrow[d];
                }
                /* per-head softplus gate: multiply attn output by gate[h] */
                float g = gate[(int64_t)s*H + h];
                for (int d = 0; d < hd; d++) cx[d] *= g;
            }
        }
        free(sc);
    }
    matmul_w(out, ctx, l->o, S, qdim, D);
    free(q); free(k); free(vv); free(gate); free(ctx);
}

/* ---------- dense MLP ---------- */
static void dense_mlp(Model *m, Layer *l, float *x, int S, float *out) {
    Cfg *c = &m->c; int D = c->hidden, I = c->dense_inter;
    float *g = falloc((int64_t)S*I), *u = falloc((int64_t)S*I);
    matmul_w(g, x, l->dg, S, D, I);
    matmul_w(u, x, l->du, S, D, I);
    for (int64_t i = 0; i < (int64_t)S*I; i++) g[i] = siluf(g[i]) * u[i];
    matmul_w(out, g, l->dd, S, I, D);
    free(g); free(u);
}

/* ---------- MoE: sigmoid router + norm_topk + shared expert ---------- */
static void moe(Model *m, Layer *l, int layer, float *x, int S, float *out) {
    Cfg *c = &m->c;
    int D = c->hidden, E = c->n_experts, K = c->topk, I = c->moe_inter;
    float *logits = falloc((int64_t)S*E);
    matmul(logits, x, l->router, S, D, E);
    memset(out, 0, (int64_t)S*D*sizeof(float));
    int   *idx  = malloc((size_t)S*K*sizeof(int));
    float *wgt  = malloc((size_t)S*K*sizeof(float));
    Slot **use  = malloc((size_t)S*K*sizeof(Slot*));
    Slot **fill = malloc((size_t)S*K*sizeof(Slot*));
    int  *fl    = malloc((size_t)S*K*sizeof(int));
    int nfill = 0;
    /* pass 1: routing + slot bookkeeping */
    for (int s = 0; s < S; s++) {
        float *lg = logits + (int64_t)s*E;
        int *si = idx + (int64_t)s*K;
        /* sigmoid scores, top-K selection */
        for (int kk = 0; kk < K; kk++) {
            int best = -1; float bv = -1e30f;
            for (int e = 0; e < E; e++) {
                int taken = 0; for (int j = 0; j < kk; j++) if (si[j]==e){taken=1;break;}
                float ch = sigmoidf(lg[e]);
                if (!taken && ch > bv) { bv = ch; best = e; }
            }
            si[kk] = best;
        }
        /* routing weights: sigmoid scores, normalized if norm_topk_prob */
        float *w = wgt + (int64_t)s*K; float sum = 0.f;
        for (int kk = 0; kk < K; kk++) { w[kk] = sigmoidf(lg[si[kk]]); sum += w[kk]; }
        if (c->norm_topk && sum > 0.f)
            for (int kk = 0; kk < K; kk++) w[kk] /= sum;
        for (int kk = 0; kk < K; kk++) {
            int eid = si[kk];
            if (m->eusage[layer]) m->eusage[layer][eid]++;
#ifdef COLI_CUDA
            if (g_exp_vram && !g_gpu_verify) { use[(int64_t)s*K + kk] = NULL; continue; }
#endif
            Slot *e = slot_find(m, layer, eid);
            if (e) m->hits++;
            else {
                m->miss++;
                e = slot_acquire(m, layer, eid);
                fill[nfill] = e; fl[nfill] = layer; nfill++;
            }
            use[(int64_t)s*K + kk] = e;
        }
    }
    /* pass 2: parallel fill */
    if (nfill) {
        double tf = now_s();
        #pragma omp parallel for schedule(dynamic,1)
        for (int j = 0; j < nfill; j++) slot_fill(m, fl[j], fill[j]);
        m->t_fill += now_s() - tf;
    }
    /* pass 3: compute */
    float *g = falloc(2*I), *u = g + I, *hh = falloc(D);
    int q4 = m->xq && m->rb13*2 == D;
    for (int s = 0; s < S; s++) {
        const float *xs = x + (int64_t)s*D;
        float *os = out + (int64_t)s*D;
        float *w = wgt + (int64_t)s*K;
        double te = now_s();
#ifdef COLI_CUDA
        if (g_exp_vram) {
            /* One launch for all K experts of this token. The backend wants
             * sum(rows) consecutive [D] rows in call order, so the same hidden
             * state is repeated K times; rows are 1 each because decode routes
             * one token at a time. */
            ColiCudaTensor *gs[64], *us[64], *ds[64];
            int rows[64], n = K > 64 ? 64 : K;
            float *xin = falloc((int64_t)n * D), *yout = falloc((int64_t)n * D);
            for (int kk = 0; kk < n; kk++) {
                int eid = idx[(int64_t)s*K + kk];
                gs[kk] = g_eg[layer][eid]; us[kk] = g_eu[layer][eid]; ds[kk] = g_ed[layer][eid];
                rows[kk] = 1;
                memcpy(xin + (int64_t)kk * D, xs, (size_t)D * sizeof(float));
            }
            if (!coli_cuda_expert_group(gs, us, ds, rows, n, yout, xin)) {
                fprintf(stderr, "coli_cuda_expert_group failed (layer %d)\n", layer); exit(1);
            }
            for (int kk = 0; kk < n; kk++) {
                const float *hg = yout + (int64_t)kk * D;
                if (g_gpu_verify) {
                    /* live-path parity: the CPU chain for the same expert, same
                     * input, run only under the gate because it costs the whole
                     * saving it exists to check */
                    Slot *e = use[(int64_t)s*K + kk];
                    if (e) {
                        matmul_q4(g, xs, e->p13, e->s13, D, 2*I);
                        for (int i = 0; i < I; i++) g[i] = siluf(g[i]) * u[i];
                        matmul_q4(hh, g, e->p2, e->s2, I, D);
                        /* Relative RMS over the whole [D] output, not a
                         * per-element max. Two simpler statistics both failed
                         * here: absolute difference is meaningless when expert
                         * activations span ~1 to ~1e6 across layers, and
                         * per-element RELATIVE difference is meaningless near
                         * zero, where a sign flip on a 0.001 element reads as
                         * 200%. The norm ratio is stable against both, and is
                         * what tests/test_expert_int4_compat.c uses. */
                        double num = 0, den = 0;
                        for (int d = 0; d < D; d++) {
                            double dv = (double)hg[d] - (double)hh[d];
                            num += dv * dv; den += (double)hh[d] * hh[d];
                        }
                        {
                            double rel = sqrt(num / (den + 1e-30));
                            if (rel > g_gv_exp_max) { g_gv_exp_max = rel; g_gv_exp_ref = sqrt(den / D); }
                            /* aggregate too: the worst SINGLE chain is often a
                             * near-zero output where any ratio is noise, so the
                             * magnitude-weighted total is the honest headline. */
                            g_gv_exp_num += num; g_gv_exp_den += den;
                        }
                        g_gv_exp_calls++;
                    }
                }
                for (int d = 0; d < D; d++) os[d] += w[kk] * hg[d];
            }
            free(xin); free(yout);
            m->t_expert += now_s() - te;
            goto shared;
        }
#endif
        for (int kk = 0; kk < K; kk++) {
            Slot *e = use[(int64_t)s*K + kk];
            if (m->xq) {
                if (q4) {
                    matmul_q4(g, xs, e->p13, e->s13, D, 2*I);
                    for (int i = 0; i < I; i++) g[i] = siluf(g[i]) * u[i];
                    matmul_q4(hh, g, e->p2, e->s2, I, D);
                } else {
                    matmul_q(g, xs, (int8_t*)e->p13, e->s13, D, 2*I);
                    for (int i = 0; i < I; i++) g[i] = siluf(g[i]) * u[i];
                    matmul_q(hh, g, (int8_t*)e->p2, e->s2, I, D);
                }
            } else if (m->quant_bits) {
                matmul_q(g, xs, e->q13, e->s13, D, 2*I);
                for (int i = 0; i < I; i++) g[i] = siluf(g[i]) * u[i];
                matmul_q(hh, g, e->q2, e->s2, I, D);
            } else {
                matmul(g, xs, e->f13, 1, D, 2*I);
                for (int i = 0; i < I; i++) g[i] = siluf(g[i]) * u[i];
                matmul(hh, g, e->f2, 1, I, D);
            }
            for (int d = 0; d < D; d++) os[d] += w[kk] * hh[d];
        }
        m->t_expert += now_s() - te;
shared: ;
        double ts = now_s();
        /* shared expert (unscaled — route_scale only applies to routed experts) */
        matmul_w(g, xs, l->sh_g, 1, D, I);
        matmul_w(u, xs, l->sh_u, 1, D, I);
        for (int i = 0; i < I; i++) g[i] = siluf(g[i]) * u[i];
        matmul_w(hh, g, l->sh_d, 1, I, D);
        /* os currently holds sum(w[kk] * expert_hh[kk]); apply route_scale, then add shared */
        for (int d = 0; d < D; d++) os[d] = os[d] * c->route_scale + hh[d];
        m->t_shared += now_s() - ts;
    }
    free(logits); free(idx); free(wgt); free(use); free(fill); free(fl);
    free(g); free(hh);
}

/* ---------- forward pass ---------- */
static float *step(Model *m, const int *ids, int S, int pos0, int *tf_out) {
    Cfg *c = &m->c; int D = c->hidden;
    float *x = falloc((int64_t)S*D);
    for (int s = 0; s < S; s++)
        wt_row_f32(m->embed, (int64_t)ids[s]*D, x + (int64_t)s*D, D);
    float *nrm = falloc((int64_t)S*D), *tmp = falloc((int64_t)S*D);
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s*D, x + (int64_t)s*D, l->in_ln, D, c->eps);
        double ta = now_s();
        attention(m, l, i, nrm, S, pos0, tmp);
        m->t_attn += now_s() - ta;
        for (int64_t j = 0; j < (int64_t)S*D; j++) x[j] += tmp[j];
        for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s*D, x + (int64_t)s*D, l->post_ln, D, c->eps);
        if (c->sparse[i]) moe(m, l, i, nrm, S, tmp);
        else dense_mlp(m, l, nrm, S, tmp);
        for (int64_t j = 0; j < (int64_t)S*D; j++) x[j] += tmp[j];
    }
    m->kv_len = pos0 + S;
    float *last = falloc(D);
    float *logit = falloc(c->vocab);
    if (tf_out) {
        for (int s = 0; s < S; s++) {
            rmsnorm_row(last, x + (int64_t)s*D, m->final_norm, D, c->eps);
            matmul_w(logit, last, m->lm_head, 1, D, c->vocab);
            int best = 0; for (int i = 1; i < c->vocab; i++) if (logit[i] > logit[best]) best = i;
            tf_out[pos0 + s] = best;
        }
    }
    rmsnorm_row(last, x + (int64_t)(S-1)*D, m->final_norm, D, c->eps);
    matmul_w(logit, last, m->lm_head, 1, D, c->vocab);
    free(x); free(nrm); free(tmp); free(last);
    return logit;
}

static void state_reset(Model *m) {
    m->kv_len = 0;
}

static void kv_alloc(Model *m, int max_t) {
    Cfg *c = &m->c;
    if (m->K && max_t <= m->max_t) return;
    if (m->K) for (int i = 0; i < c->n_layers; i++) { free(m->K[i]); free(m->V[i]); }
    if (m->rope_cos_full) { free(m->rope_cos_full); free(m->rope_sin_full); }
    if (m->rope_cos_swa)  { free(m->rope_cos_swa);  free(m->rope_sin_swa); }
    free(m->K); free(m->V);
    m->max_t = max_t;
    m->K = calloc(c->n_layers, sizeof(float*)); m->V = calloc(c->n_layers, sizeof(float*));
    for (int i = 0; i < c->n_layers; i++) {
        m->K[i] = falloc((int64_t)L_KV(c,i) * max_t * L_HD(c,i));
        m->V[i] = falloc((int64_t)L_KV(c,i) * max_t * L_HD(c,i));
    }
    /* precompute RoPE tables */
    m->rope_dim_full = (int)(c->head_dim * c->partial_rotary_full);
    m->rope_dim_swa  = (int)(c->head_dim * c->partial_rotary_swa);
    rope_build_tables(&m->rope_cos_full, &m->rope_sin_full, max_t, c->head_dim,
                      c->partial_rotary_full, c->rope_type_full, c->rope_theta_full,
                      c->yarn_factor_full, c->yarn_beta_slow_full, c->yarn_beta_fast_full,
                      c->yarn_orig_max_full, c->yarn_attn_factor_full);
    if (c->partial_rotary_swa != c->partial_rotary_full ||
        c->rope_theta_swa != c->rope_theta_full ||
        c->rope_type_swa != c->rope_type_full) {
        rope_build_tables(&m->rope_cos_swa, &m->rope_sin_swa, max_t, c->head_dim,
                          c->partial_rotary_swa, c->rope_type_swa, c->rope_theta_swa,
                          1.0f, 1.0f, 32.0f, 8192, 1.0f);
    } else {
        /* SWA uses the same rope as full-attention */
        int dim = m->rope_dim_full;
        m->rope_cos_swa = falloc((int64_t)max_t * dim);
        m->rope_sin_swa = falloc((int64_t)max_t * dim);
        memcpy(m->rope_cos_swa, m->rope_cos_full, (int64_t)max_t * dim * sizeof(float));
        memcpy(m->rope_sin_swa, m->rope_sin_full, (int64_t)max_t * dim * sizeof(float));
    }
}

/* ---------- generation ---------- */
static void generate(Model *m, const int *prompt, int np, int n_new, int *out) {
    for (int i = 0; i < np; i++) out[i] = prompt[i];
    float *logit = step(m, prompt, np, 0, NULL);
    int len = np;
    Cfg *c = &m->c;
    for (int s = 0; s < n_new; s++) {
        int best = 0; float bv = logit[0];
        for (int i = 1; i < c->vocab; i++) if (logit[i] > bv) { bv = logit[i]; best = i; }
        free(logit);
        out[len++] = best;
        if (s == n_new - 1) break;
        int one = best;
        logit = step(m, &one, 1, len - 1, NULL);
    }
}

static void generate_stream(Model *m, Tok *T, const char *prompt, int n_new) {
    Cfg *c = &m->c;
    int cap = (int)strlen(prompt) + 16;
    int *ids = malloc(cap * sizeof(int));
    int np = tok_encode(T, prompt, (int)strlen(prompt), ids, cap);
    if (np <= 0) { fprintf(stderr, "empty prompt after tokenization\n"); return; }
    /* prepend BOS if the model has one (Laguna was trained with BOS) — unless the
     * rendered template already starts with it (Laguna's emits 〈|EOS|〉 itself) */
    if (c->bos >= 0 && ids[0] != c->bos) {
        ids = realloc(ids, (np + 1) * sizeof(int));
        memmove(ids + 1, ids, np * sizeof(int));
        ids[0] = c->bos;
        np++;
    }
    kv_alloc(m, np + n_new + 8);
    printf("[%d prompt tokens] %s", np, prompt); fflush(stdout);
    double t0 = now_s(), t1 = 0;
    float *logit = step(m, ids, np, 0, NULL);
    int len = np;
    char buf[512];
    /* LOGIT_DUMP=<path> writes every step's raw logits. Kept because reasoning
     * about the SHAPE of this distribution rather than capturing one has now
     * been wrong twice: the sampler's candidate-pruning threshold looked
     * useless against synthesized logits and is 135x on real ones. */
    FILE *ldump = getenv("LOGIT_DUMP") ? fopen(getenv("LOGIT_DUMP"), "wb") : NULL;
    for (int s = 0; s < n_new; s++) {
        if (ldump) fwrite(logit, 4, (size_t)c->vocab, ldump);
        int best = 0; float bv = logit[0];
        for (int i = 1; i < c->vocab; i++) if (logit[i] > bv) { bv = logit[i]; best = i; }
        free(logit);
        if (s == 0) t1 = now_s();
        if (is_eos(c, best)) { printf("\n[eos after %d tokens]", s); break; }
        int nb = tok_decode(T, &best, 1, buf, sizeof(buf)-1);
        buf[nb] = 0; fputs(buf, stdout); fflush(stdout);
        int one = best; len++;
        if (s == n_new - 1) break;
        logit = step(m, &one, 1, len - 1, NULL);
    }
#ifdef COLI_CUDA
    if (g_gpu_verify)
    {
        printf("\n[gpu-verify] dense  : %ld matmuls, worst |gpu-cpu| = %.6g (|cpu| there = %.6g)\n",
               g_gv_calls, g_gv_max, g_gv_ref);
        if (g_gv_exp_calls)
            printf("[gpu-verify] experts: %ld chains, aggregate rel-RMS = %.4f%%"
                   " (worst single chain %.2f%% at magnitude %.3g)\n",
                   g_gv_exp_calls, sqrt(g_gv_exp_num / (g_gv_exp_den + 1e-30)) * 100.0,
                   g_gv_exp_max * 100.0, g_gv_exp_ref);
    }
#endif
    if (ldump) fclose(ldump);
    double dt = now_s() - t1;
    int gen = len - np;
    printf("\n[prefill %.1fs | %d tokens in %.1fs = %.2f tok/s | RSS %.1f GB]\n",
           t1 - t0, gen, dt, gen > 1 ? (gen-1)/dt : 0.0, rss_gb());
    printf("[phases] fill %.1fs | expert-mm %.1fs | shared %.1fs | attn %.1fs | other %.1fs\n",
           m->t_fill, m->t_expert, m->t_shared, m->t_attn,
           (now_s()-t0) - m->t_fill - m->t_expert - m->t_shared - m->t_attn);
    free(ids);
}

/* ---------- serve mode (same protocol as colibri.c / inkling.c) ---------- */
static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static double rng_next(void) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return (double)(g_rng >> 11) / 9007199254740992.0;
}

typedef struct { float p; int i; } PI;
static int pi_desc(const void *a, const void *b) {
    float d = ((const PI*)b)->p - ((const PI*)a)->p;
    return d > 0 ? 1 : d < 0 ? -1 : 0;
}
/* Nucleus sampling without sorting the whole vocabulary.
 *
 * The straightforward version qsorts all n probabilities every token. At
 * n=100352 that measured 13.0 ms/token against 0.04 ms for the greedy path --
 * with experts resident the whole token budget is ~50 ms, so a quarter of
 * decode was going into a sort whose result is almost entirely discarded.
 *
 * Only tokens near the max can matter: p_i = exp((l_i - max)/temp), so anything
 * more than -ln(EPS)*temp below the max has p < EPS individually. With
 * EPS = 1e-12 the entire excluded tail is under n*EPS = 1e-7 of a sum that is
 * at least 1 (the max token contributes exactly 1), which moves the top-p
 * cutoff by far less than the sampling noise it feeds. So: collect the
 * candidates in one pass, sort only those. A flat distribution defeats the
 * pruning, and for that -- and only that -- the full sort is still here. */
/* ln(1e-9). The excluded tail is then under n*1e-9 ~ 1e-4 of a sum that is at
 * least 1, i.e. 0.2% of the 0.05 slack that top_p=0.95 leaves -- comfortably
 * below the sampling noise it feeds. Measured on real Laguna logits, this keeps
 * a median of ~40 candidates out of 100352 at temp 0.7. */
#define SAMPLE_EPS_LN (-20.72f)

static int sample_pick(PI *c, int cnt, double sum, float top_p) {
    qsort(c, (size_t)cnt, sizeof(PI), pi_desc);
    double cut = (top_p > 0.f && top_p < 1.f) ? top_p * sum : sum;
    double acc = 0; int k = 0;
    while (k < cnt && acc < cut) acc += c[k++].p;
    double r = rng_next() * acc, run = 0;
    int pick = c[0].i;
    for (int i = 0; i < k; i++) { run += c[i].p; if (run >= r) { pick = c[i].i; break; } }
    return pick;
}

static int sample_logits(const float *logit, int n, float temp, float top_p) {
    int best = 0;
    for (int i = 1; i < n; i++) if (logit[i] > logit[best]) best = i;
    if (temp <= 0.f) return best;

    float floor_l = logit[best] + temp * SAMPLE_EPS_LN;   /* SAMPLE_EPS_LN is negative */
    int cnt = 0;
    for (int i = 0; i < n; i++) if (logit[i] >= floor_l) cnt++;

    /* No upper bound on cnt: sorting the candidates is never worse than sorting
     * everything, and the counting pass that buys it is one O(n) sweep. The
     * fallback below exists for cnt == n (a flat distribution prunes nothing)
     * and for allocation failure. */
    if (cnt > 0 && cnt < n) {
        PI *c = malloc((size_t)cnt * sizeof(PI));
        if (c) {
            double sum = 0; int k = 0;
            for (int i = 0; i < n && k < cnt; i++) {
                if (logit[i] < floor_l) continue;
                c[k].p = expf((logit[i] - logit[best]) / temp);
                c[k].i = i; sum += c[k].p; k++;
            }
            int pick = sample_pick(c, k, sum, top_p);
            free(c);
            return pick;
        }
    }
    /* flat distribution (or OOM): every token is a candidate, so sort them all */
    PI *c = malloc((size_t)n * sizeof(PI));
    if (!c) return best;
    double sum = 0;
    for (int i = 0; i < n; i++) { c[i].p = expf((logit[i]-logit[best])/temp); c[i].i = i; sum += c[i].p; }
    int pick = sample_pick(c, n, sum, top_p);
    free(c);
    return pick;
}

static void apply_rep_penalty(float *logit, int n, const int *hist, int nhist, float pen) {
    if (pen <= 1.f) return;
    for (int i = 0; i < nhist; i++) {
        int t = hist[i]; if (t < 0 || t >= n) continue;
        logit[t] = logit[t] > 0 ? logit[t] / pen : logit[t] * pen;
    }
}

static const char *prompt_reject(int np, int want) {
    const char *cm = getenv("CTX_MAX");
    int ctx_max = cm ? atoi(cm) : 8192;
    if (np + want > ctx_max) return "context exceeds CTX_MAX";
    return NULL;
}

typedef struct { char id[64]; int max_tok; float temp, top_p; char *payload; int plen; } SReq;
#define SRV_QMAX 16
static SReq g_q[SRV_QMAX]; static int g_qn = 0;

static int stdin_readable(void) {
    fd_set r; struct timeval tv = {0, 0};
    FD_ZERO(&r); FD_SET(0, &r);
    return select(1, &r, NULL, NULL, &tv) > 0;
}

static int serve_read_cmd(const char *cur_id) {
    char ln[512];
    if (!fgets(ln, sizeof(ln), stdin)) return -1;
    char cmd[16], id[64];
    if (sscanf(ln, "%15s %63s", cmd, id) < 2) return 0;
    if (!strcmp(cmd, "CANCEL")) return cur_id && !strcmp(id, cur_id);
    if (!strcmp(cmd, "SUBMIT")) {
        int slot, plen, max_tok; float temp, top_p;
        if (sscanf(ln, "%*s %*s %d %d %d %f %f", &slot, &plen, &max_tok, &temp, &top_p) != 5 ||
            plen < 0 || plen > (1<<22)) { printf("ERROR %s bad submit header\n", id); fflush(stdout); return 0; }
        (void)slot;
        char *pl = malloc((size_t)plen + 1);
        if (fread(pl, 1, (size_t)plen, stdin) != (size_t)plen) { free(pl); return -1; }
        int nl = fgetc(stdin); (void)nl;
        pl[plen] = 0;
        if (g_qn < SRV_QMAX) {
            SReq *q = &g_q[g_qn++];
            snprintf(q->id, sizeof(q->id), "%s", id);
            q->max_tok = max_tok; q->temp = temp; q->top_p = top_p;
            q->payload = pl; q->plen = plen;
        } else { printf("ERROR %s queue full\n", id); fflush(stdout); free(pl); }
    }
    return 0;
}

static void serve_one(Model *m, Tok *T, SReq *q) {
    Cfg *c = &m->c;
    int cap = q->plen + 16;
    int *ids = malloc((size_t)cap * sizeof(int));
    int np = tok_encode(T, q->payload, q->plen, ids, cap);
    if (np <= 0) { printf("ERROR %s empty prompt\n", q->id); fflush(stdout); free(ids); return; }
    /* prepend BOS if the model has one — unless the rendered template already starts with it */
    if (c->bos >= 0 && ids[0] != c->bos) {
        ids = realloc(ids, (size_t)(np + 1) * sizeof(int));
        memmove(ids + 1, ids, np * sizeof(int));
        ids[0] = c->bos;
        np++;
    }
    const char *bad = prompt_reject(np, q->max_tok);
    if (bad) { printf("ERROR %s %s\n", q->id, bad); fflush(stdout); free(ids); return; }
    state_reset(m);
    kv_alloc(m, np + q->max_tok + 8);
    double t0 = now_s();
    uint64_t h0 = m->hits, m0 = m->miss;
    double f0 = m->t_fill, e0 = m->t_expert, s0 = m->t_shared, a0 = m->t_attn;
    float *logit = step(m, ids, np, 0, NULL);
    int len = np, gen = 0, limited = 1, cancelled = 0;
    char buf[512];
    float rep = getenv("REP_PEN") ? atof(getenv("REP_PEN")) : 1.1f;
    int hist[128], nhist = 0;
    for (int i = (np > 128 ? np - 128 : 0); i < np; i++) hist[nhist++] = ids[i];
    for (int s = 0; s < q->max_tok && !cancelled; s++) {
        apply_rep_penalty(logit, c->vocab, hist, nhist, rep);
        int tk = sample_logits(logit, c->vocab, q->temp, q->top_p);
        free(logit); logit = NULL;
        if (is_eos(c, tk)) { limited = 0; break; }
        if (nhist < 128) hist[nhist++] = tk;
        else { memmove(hist, hist+1, 127*sizeof(int)); hist[127] = tk; }
        int nb = tok_decode(T, &tk, 1, buf, sizeof(buf)-1);
        printf("DATA %s %d\n", q->id, nb);
        fwrite(buf, 1, (size_t)nb, stdout);
        fputc('\n', stdout); fflush(stdout);
        gen++; len++;
        while (stdin_readable()) {
            int r = serve_read_cmd(q->id);
            if (r < 0) { free(ids); return; }
            if (r > 0) { cancelled = 1; limited = 0; }
        }
        if (cancelled || s == q->max_tok - 1) break;
        logit = step(m, &tk, 1, len - 1, NULL);
    }
    free(logit);
    double dt = now_s() - t0;
    double tot = (double)(m->hits - h0 + m->miss - m0);
    printf("DONE %s STAT %d %.3f %.1f %.2f %d %d\n", q->id, gen,
           dt > 0 ? gen/dt : 0.0, tot ? 100.0*(m->hits-h0)/tot : 0.0, rss_gb(), np, limited);
    printf("PROF %.3f %d %d %.3f %.3f %.3f %.3f %.3f %d\n", dt, np, gen,
           m->t_fill-f0, m->t_shared-s0, m->t_expert-e0, m->t_attn-a0, 0.0, gen+1);
    fflush(stdout);
    free(ids);
}

static void serve_hwinfo(Model *m) {
    char cpu[256] = ""; int cores = 0; double rt = 0, ra = 0;
    FILE *ci = fopen("/proc/cpuinfo", "r");
    if (ci) { char ln[256];
        while (fgets(ln, sizeof(ln), ci)) if (!strncmp(ln, "model name", 10)) {
            char *p = strchr(ln, ':'); if (p) { p++; while (*p == ' ') p++;
            int n = (int)strlen(p); if (n > 0 && p[n-1] == '\n') p[--n] = 0;
            snprintf(cpu, sizeof(cpu), "%s", p); } break; }
        fclose(ci); }
#ifdef _SC_NPROCESSORS_ONLN
    cores = (int)sysconf(_SC_NPROCESSORS_ONLN);
#endif
    FILE *mi = fopen("/proc/meminfo", "r");
    if (mi) { char ln[256]; double v = 0;
        while (fgets(ln, sizeof(ln), mi)) {
            if (sscanf(ln, "MemTotal: %lf", &v) == 1) rt = v/1e6;
            if (sscanf(ln, "MemAvailable: %lf", &v) == 1) ra = v/1e6;
        } fclose(mi); }
    int ngpu = 0; double vram = 0; const char *gpu = "";
#ifdef COLI_CUDA
    /* field 5 is VRAM *total*, matching telemetry.h's hwinfo_emit — the
     * dashboard prints it as the card's capacity. It used to send free bytes,
     * which reads plausibly (free-at-startup is close to total) and is wrong. */
    char gname[128] = "";
    if (g_cuda) { ngpu = 1; vram = ink_cuda_total_bytes()/1e9;
                  ink_cuda_device_name(gname, sizeof(gname));
                  gpu = gname[0] ? gname : "CUDA device"; }
#endif
    (void)m;
    printf("HWINFO %d %.1f %.1f %d %.1f %s|%s\n", cores, rt, ra, ngpu, vram, cpu[0]?cpu:"unknown", gpu);
#ifdef COLI_CUDA
    /* Resident dense weight, reported separately from TIERS on purpose: TIERS
     * is the EXPERT cortex (counts of experts per tier) and what laguna keeps
     * in VRAM is not an expert, it is the attention/lm_head/shared-expert
     * bf16. Folding the bytes into the expert row produced a "VRAM 0 · 3.6 GB"
     * legend — two taxonomies in one line. This belongs beside the GPU's
     * capacity in the runtime panel instead. */
    if (g_cuda) { printf("GPUMEM %.2f %.2f\n", (g_vram_bytes + g_exp_bytes)/1e9, vram); }
#endif
    fflush(stdout);
}

static void serve_tiers_emap(Model *m) {
    Cfg *c = &m->c; int E = c->n_experts;
    int nsp = 0, filled = 0;
    for (int i = 0; i < c->n_layers; i++) if (c->sparse[i]) { nsp++; filled += m->cache[i].n; }
    int64_t I = c->moe_inter, D = c->hidden;
    int64_t slotb = m->xq ? m->rb13*2*I + m->rb2*D + (2*I+D)*4
                  : m->quant_bits ? 3*I*D + (2*I+D)*4 : 3*I*D*4;
    /* Both VRAM figures are 0 because this row is EXPERT placement and laguna
     * puts no expert on the GPU unless EXPERTS_VRAM made the whole
     * set resident, in which case the counts say so. Either way the DENSE
     * weights in VRAM are a different thing and go out as GPUMEM, not here. */
    int vram_n = 0; double vram_gb = 0;
#ifdef COLI_CUDA
    if (g_exp_vram) { vram_n = (int)g_exp_count; vram_gb = g_exp_bytes / 1e9; filled = 0; }
#endif
    printf("TIERS %d %d %d %.2f %.2f\n", vram_n, filled, nsp*E - vram_n - filled,
           vram_gb, filled*(double)slotb/1e9);
    char *hex = malloc((size_t)nsp*E*2 + 1); int w = 0;
    for (int i = 0; i < c->n_layers; i++) {
        if (!c->sparse[i]) continue;
        LCache *lc = &m->cache[i];
        for (int e = 0; e < E; e++) {
            int tier = 0;
            for (int z = 0; z < lc->n; z++) if (lc->slots[z].eid == e && lc->slots[z].filled) { tier = 1; break; }
            uint32_t u = m->eusage[i] ? m->eusage[i][e] : 0;
            int heat = 0; while (u) { heat++; u >>= 1; } if (heat > 63) heat = 63;
            int b = (tier << 6) | heat;
            hex[w++] = "0123456789abcdef"[b >> 4];
            hex[w++] = "0123456789abcdef"[b & 15];
        }
    }
    hex[w] = 0;
    printf("EMAP %d %d %s\n", nsp, E, hex);
    fflush(stdout); free(hex);
}

static void serve_loop(Model *m, Tok *T) {
    setvbuf(stdin, NULL, _IONBF, 0);
    const char *sd = getenv("SEED");
    if (sd) g_rng ^= (uint64_t)strtoull(sd, NULL, 10);
    else g_rng ^= (uint64_t)time(NULL) * 2654435761u;
    fputs("\x01\x01READY\x01\x01\n", stdout);
    printf("STAT 0 0.0 0.0 %.2f 0 0\n", rss_gb());
    fflush(stdout);
    serve_hwinfo(m);
    serve_tiers_emap(m);
    for (;;) {
        while (!g_qn) if (serve_read_cmd(NULL) < 0) return;
        SReq q = g_q[0];
        memmove(g_q, g_q+1, (size_t)(--g_qn) * sizeof(SReq));
        serve_one(m, T, &q);
        serve_tiers_emap(m);
        free(q.payload);
    }
}

/* one thread per PHYSICAL core: SMT siblings share the vector units and the
 * per-expert matmul regions are tiny, so one thread per logical CPU measures
 * ~6x SLOWER than one per core (0.72 vs 4.1 tok/s on a 5950X). */
static int physical_cores(void) {
#ifdef __linux__
    int n = 0;
    for (int c = 0; ; c++) {
        char p[128];
        snprintf(p, sizeof(p), "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", c);
        FILE *f = fopen(p, "r");
        if (!f) break;
        int first = -1;
        if (fscanf(f, "%d", &first) == 1 && first == c) n++;
        fclose(f);
    }
    if (n > 0) return n;
#endif
#ifdef _SC_NPROCESSORS_ONLN
    long nc = sysconf(_SC_NPROCESSORS_ONLN);
    if (nc > 0) return (int)nc;
#endif
    return 0;
}

/* ---------- oracle harness ---------- */
static int *read_int_array(jval *o, const char *key, int *n_out) {
    jval *a = json_get(o, key);
    if (!a || a->t != J_ARR) { *n_out = 0; return NULL; }
    int *r = malloc(a->len * sizeof(int));
    for (int i = 0; i < a->len; i++) r[i] = (int)a->kids[i]->num;
    *n_out = a->len; return r;
}

int main(int argc, char **argv) {
    /* Unlike colibri.c, this block is NOT gated on COLI_CUDA. The GPU backend
     * here only takes the dense bf16 matmuls (attention, lm_head); every
     * routed expert stays on the CPU, so OMP tuning matters at least as much
     * with a GPU attached as without. Skipping it costs everything: measured
     * on a 5950X + MI50, GPU decode is 8.7 tok/s tuned and 0.24 tok/s untuned
     * (libgomp defaults to 32 SMT threads, and expert-mm goes 0.3s -> 130s). */
    if (!getenv("COLI_OMP_TUNED") && !getenv("COLI_NO_OMP_TUNE")) {
        /* If OMP_PROC_BIND/OMP_PLACES were already in the environment, libgomp
         * bound this initial thread to a single-CPU place BEFORE main() ran; an
         * execv would carry that one-CPU affinity mask into the new image, whose
         * libgomp would then size the whole team at 1 thread (the mask survives
         * exec). In that case skip the re-exec: binding is already configured,
         * and the team size can still be fixed through the API. */
        int bind_preset = getenv("OMP_PROC_BIND") || getenv("OMP_PLACES");
        char nt[16] = "";
        if (!getenv("OMP_NUM_THREADS")) {
            int pc = physical_cores();
            if (pc > 0) { snprintf(nt, sizeof(nt), "%d", pc); setenv("OMP_NUM_THREADS", nt, 0); }
        }
        setenv("OMP_WAIT_POLICY","active",0);
        setenv("GOMP_SPINCOUNT","200000",0);
        setenv("OMP_PLACES","cores",0);
        setenv("OMP_PROC_BIND","close",0);
        setenv("OMP_DYNAMIC","FALSE",0);
        setenv("COLI_OMP_TUNED","1",1);
        if (!bind_preset) {
#ifdef __linux__
            execv("/proc/self/exe", argv);
            perror("[OMP] execv self-reexec failed, running untuned");
#endif
        }
#ifdef _OPENMP
        else if (nt[0]) omp_set_num_threads(atoi(nt));
#endif
    }
    const char *snap = getenv("SNAP");
    if (!snap) { fprintf(stderr, "set SNAP=<snapshot directory>\n"); return 1; }
    const char *prompt = NULL, *pfile = NULL, *refpath = "ref_laguna.json";
    int cap = -1, bits = 0, n_new = 256, npos = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-p") && i+1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "-f") && i+1 < argc) pfile = argv[++i];
        else if (!strcmp(argv[i], "-n") && i+1 < argc) n_new = atoi(argv[++i]);
        else if (npos == 0) { cap = atoi(argv[i]); npos++; }
        else if (npos == 1) { bits = atoi(argv[i]); npos++; }
        else refpath = argv[i];
    }
    if (cap < 0) cap = (prompt || pfile) ? 0 : 16;
    if (bits && (bits < 2 || bits > 8)) { fprintf(stderr, "quant_bits must be 0 (f32) or 2..8\n"); return 1; }

    if (getenv("SERVE") && getenv("SERVE")[0] == '1') {
        Model m; model_init(&m, snap, cap, bits);
        pins_load(&m, snap);
        char tkp[2048]; snprintf(tkp, sizeof(tkp), "%s/tokenizer.json", snap);
        Tok T; tok_load(&T, tkp);
        serve_loop(&m, &T);
        usage_save(&m, snap);
        return 0;
    }

    if (prompt || pfile) {
        Model m; model_init(&m, snap, cap, bits);
        printf("== Laguna C engine, %d layers, experts @ %s, cache %d/layer ==\n",
               m.c.n_layers, m.xq ? "container" : bits ? "int" : "f32", m.cache[0].cap);
        pins_load(&m, snap);
        char tkp[2048]; snprintf(tkp, sizeof(tkp), "%s/tokenizer.json", snap);
        Tok T; tok_load(&T, tkp);
        if (prompt) generate_stream(&m, &T, prompt, n_new);
        else {
            FILE *pf = fopen(pfile, "rb"); if (!pf) { perror(pfile); return 1; }
            char ln[8192]; int np = 0;
            while (fgets(ln, sizeof(ln), pf)) {
                size_t n = strlen(ln); while (n && (ln[n-1]=='\n'||ln[n-1]=='\r')) ln[--n]=0;
                if (!n || ln[0]=='#') continue;
                printf("\n===== prompt %d =====\n", ++np);
                state_reset(&m);
                generate_stream(&m, &T, ln, n_new);
            }
            fclose(pf);
        }
        int saved = usage_save(&m, snap);
        double tot = m.hits + m.miss;
        printf("[cache] hit %.1f%% (%llu hit / %llu load)%s\n",
               tot ? 100.0*m.hits/tot : 0.0,
               (unsigned long long)m.hits, (unsigned long long)m.miss,
               saved ? " | usage history saved" : "");
        return 0;
    }

    FILE *f = fopen(refpath, "rb"); if(!f){perror(refpath);return 1;}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *buf=malloc(n+1); if(fread(buf,1,n,f)!=(size_t)n){} buf[n]=0; fclose(f);
    char *arena=NULL; jval *ref = json_parse(buf, &arena);
    int np, nfull, ntf;
    int *pids  = read_int_array(ref,"prompt_ids",&np);
    int *full  = read_int_array(ref,"full_ids",&nfull);
    int *tfref = read_int_array(ref,"tf_pred",&ntf);
    int ngen = nfull - np;

    Model m; model_init(&m, snap, cap, bits);
    printf("== Laguna C engine (Stage A), cache = %d experts/layer, experts @ %s ==\n",
           cap, m.xq ? "container (int4/int8 + .qs)" : bits ? "int (runtime quant)" : "f32");
    printf("cfg: D=%d L=%d V=%d heads=%d kv=%d hd=%d win=%d E=%d topk=%d shared=%d\n",
           m.c.hidden, m.c.n_layers, m.c.vocab, m.c.n_heads, m.c.n_kv,
           m.c.head_dim, m.c.window, m.c.n_experts, m.c.topk, m.c.n_shared);
    printf("resident weights loaded in %.1fs | RSS: %.2f GB\n", m.dense_load_s, rss_gb());
    kv_alloc(&m, nfull + 8);

    if (tfref && ntf == nfull) {
        int *tf = malloc(nfull * sizeof(int));
        float *lg = step(&m, full, nfull, 0, tf);
        free(lg);
        int ok = 0; for (int i = 0; i < nfull; i++) ok += (tf[i] == tfref[i]);
        printf("teacher-forced argmax: %d/%d match\n", ok, nfull);
        free(tf);
        state_reset(&m);
    }

    int *out = malloc(nfull * sizeof(int));
    double t = now_s();
    generate(&m, pids, np, ngen, out);
    double dt = now_s() - t;
    int match = 0;
    printf("Reference: "); for (int i=np;i<nfull;i++) printf("%d ", full[i]);
    printf("\nC engine : "); for (int i=np;i<nfull;i++) { printf("%d ", out[i]); if (out[i]==full[i]) match++; }
    printf("\nMatching tokens: %d/%d\n", match, ngen);
    double tot = m.hits + m.miss;
    printf("PEAK RSS: %.2f GB | expert cache hit %.1f%% | %.2f tok/s\n",
           rss_gb(), tot?100.0*m.hits/tot:0.0, ngen/dt);
    free(buf); free(arena);
    return (match == ngen) ? 0 : 1;
}