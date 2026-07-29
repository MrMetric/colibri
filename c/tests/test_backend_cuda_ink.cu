/* Correctness oracle for the inkling/laguna bf16 GPU matmul.
 *
 * The kernel gives ONE output row to a 32-thread group and finishes with a
 * shuffle reduction, so it is sensitive to the hardware wavefront width: CUDA
 * and RDNA run 32-lane waves, CDNA/GCN (gfx906, gfx90a) run 64-lane ones. A
 * width mistake there does not fail to compile and does not crash — it just
 * returns sums built from the wrong lanes. Hence an absolute reference
 * (double accumulation, written here from the documented contract and not
 * shared with the kernel) rather than a self-consistency check.
 *
 * Build+run:  make ink-test HIP=1 HIP_ARCH=gfx906   (or CUDA=1) */
#include "../backend_cuda_ink.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>

/* bf16 round-to-nearest-even — how the weights are stored on disk; the
 * activation stays f32 on both sides. */
static uint16_t f32_to_bf16(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    uint32_t r = (u >> 16) & 1u;
    u += 0x7fffu + r;
    return (uint16_t)(u >> 16);
}
static float bf16_to_f32(uint16_t h) {
    uint32_t u = (uint32_t)h << 16; float f; std::memcpy(&f, &u, 4); return f;
}

static uint32_t rng_state = 12345u;
static float rnd(void) {           /* xorshift, deterministic across runs */
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5;
    return (float)((int32_t)rng_state) / 2147483648.0f;
}

static double worst_ratio = 0;   /* worst observed |d| as a fraction of the tolerance */

static int one_case(int S, int I, int O) {
    uint16_t *W = (uint16_t *)std::malloc((size_t)O * I * 2);
    float *x = (float *)std::malloc((size_t)S * I * 4);
    float *y = (float *)std::malloc((size_t)S * O * 4);
    double *ref = (double *)std::malloc((size_t)S * O * sizeof(double));
    if (!W || !x || !y || !ref) { std::fprintf(stderr, "OOM\n"); return 0; }

    for (size_t i = 0; i < (size_t)O * I; i++) W[i] = f32_to_bf16(rnd());
    for (size_t i = 0; i < (size_t)S * I; i++) x[i] = rnd();
    /* reference: bf16 weights times f32 activations — the kernel's contract,
     * see backend_cuda_ink.cu — accumulated in double */
    for (int s = 0; s < S; s++)
        for (int o = 0; o < O; o++) {
            double acc = 0;
            for (int i = 0; i < I; i++)
                acc += (double)bf16_to_f32(W[(size_t)o * I + i]) *
                       (double)x[(size_t)s * I + i];
            ref[(size_t)s * O + o] = acc;
        }

    void *dW = ink_cuda_upload(W, (size_t)O * I * 2);
    if (!dW) { std::fprintf(stderr, "upload failed (S=%d I=%d O=%d)\n", S, I, O); return 0; }
    std::memset(y, 0xff, (size_t)S * O * 4);
    if (ink_cuda_matmul_bf16(y, x, dW, S, I, O) != 0) {
        std::fprintf(stderr, "matmul failed (S=%d I=%d O=%d)\n", S, I, O); return 0;
    }

    /* Only f32 summation order separates the kernel from the reference, so the
     * bound is the usual sqrt(n)-random-walk one over I terms of mean size
     * E|w*x| ~ 1/4 for these operands, times a 10x margin. A wrong-lane
     * reduction misses this by orders of magnitude (see the width=16 control
     * in the commit that added this test), so the margin costs no sharpness. */
    double tol = 10.0 * std::sqrt((double)I) * 6e-8 * 0.25 * I;
    int bad = 0;
    for (int i = 0; i < S * O; i++) {
        double d = std::fabs((double)y[i] - ref[i]);
        if (d > worst_ratio * tol) worst_ratio = d / tol;
        if (d > tol) {
            if (bad++ < 4)
                std::fprintf(stderr, "S=%d I=%d O=%d idx %d: got %.6f want %.6f (|d|=%.6g > %.6g)\n",
                             S, I, O, i, (double)y[i], ref[i], d, tol);
        }
    }
    std::free(W); std::free(x); std::free(y); std::free(ref);
    if (bad) { std::fprintf(stderr, "S=%d I=%d O=%d: %d/%d elements wrong\n", S, I, O, bad, S * O); return 0; }
    return 1;
}

int main(void) {
    int dev = std::getenv("GPU_DEV") ? std::atoi(std::getenv("GPU_DEV")) : 0;
    if (ink_cuda_init(dev) != 0) { std::fprintf(stderr, "ink_cuda_init(%d) failed\n", dev); return 1; }
    std::printf("ink matmul: device %d, %.1f GB free\n", dev, ink_cuda_free_bytes() / 1e9);

    /* I: multiples of the 64-element lane stride and deliberately not (66, 130);
     * O: below, at and above the 4-rows-per-block granularity;
     * S: single-token decode and a small prefill batch. */
    static const int Is[] = { 64, 66, 128, 130, 192, 2048 };
    static const int Os[] = { 1, 3, 4, 5, 7, 1024 };
    static const int Ss[] = { 1, 3 };
    int fail = 0, n = 0;
    for (size_t a = 0; a < sizeof(Is) / sizeof(*Is); a++)
        for (size_t b = 0; b < sizeof(Os) / sizeof(*Os); b++)
            for (size_t c = 0; c < sizeof(Ss) / sizeof(*Ss); c++) {
                n++;
                if (!one_case(Ss[c], Is[a], Os[b])) fail++;
            }

    /* odd I is rejected, not silently mis-computed */
    float y1 = 1.f, x1[3] = { 1.f, 2.f, 3.f };
    uint16_t w1[3] = { 0, 0, 0 };
    void *d1 = ink_cuda_upload(w1, sizeof(w1));
    if (d1 && ink_cuda_matmul_bf16(&y1, x1, d1, 1, 3, 1) == 0) {
        std::fprintf(stderr, "odd I=3 should have been refused\n"); fail++;
    }

    if (fail) { std::fprintf(stderr, "ink matmul: %d/%d cases FAILED\n", fail, n); return 1; }
    std::printf("ink matmul: %d shapes ok (worst deviation was %.1f%% of tolerance)\n", n, worst_ratio * 100.0);
    return 0;
}
