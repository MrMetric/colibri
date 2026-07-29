/* Can laguna's int4 expert container be handed to backend_cuda.cu's fmt=2 path
 * with no repacking?
 *
 * laguna.c stores each routed expert as colibri-container int4: one byte per two
 * weights along the contraction dim, low nibble first, offset-binary (-8), plus
 * one f32 scale per output row. backend_cuda.cu's fmt=2 claims the same layout.
 * Reading both decoders says they agree, which is exactly the kind of
 * form-matching that should not be believed — a packing mismatch does not fail
 * to compile and does not crash, it just returns confident garbage.
 *
 * So: take a REAL expert out of a REAL container (tools/dump_expert.py), decode
 * it here in double precision straight from the format description, and compare
 * that against both the GPU path and laguna's own matmul_q4. A deliberately
 * wrong decode (nibble halves swapped) runs alongside as the null, so "the
 * numbers agree" means something.
 *
 * Build+run:  make expert-compat-test HIP=1 HIP_ARCH=gfx906 EXPERT_BIN=... */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "../backend_cuda.h"

/* laguna's matmul_q4 is static; including the translation unit is how the test
 * exercises the REAL function rather than a copy of it that could drift. main()
 * is renamed away on the command line. */
#include "../laguna.c"
#undef main   /* laguna.c's main is renamed on the command line; the test keeps its own */

static double rel_rms(const float *a, const float *b, int n) {
    double num = 0, den = 0;
    for (int i = 0; i < n; i++) { double d = (double)a[i] - b[i]; num += d * d; den += (double)b[i] * b[i]; }
    return sqrt(num / (den + 1e-30));
}

/* The format, written from its description rather than shared with either
 * implementation. swap=1 decodes the nibbles the wrong way round: the null. */
static void ref_decode(double *y, const float *x, const uint8_t *w, const float *s,
                       int I, int O, int swap) {
    int rb = I / 2;
    for (int o = 0; o < O; o++) {
        const uint8_t *row = w + (size_t)o * rb;
        double acc = 0;
        for (int j = 0; j < rb; j++) {
            int lo = (row[j] & 0xF) - 8, hi = (row[j] >> 4) - 8;
            int q0 = swap ? hi : lo, q1 = swap ? lo : hi;
            acc += (double)q0 * x[2 * j] + (double)q1 * x[2 * j + 1];
        }
        y[o] = acc * s[o];
    }
}

static void to_f32(float *dst, const double *src, int n) {
    for (int i = 0; i < n; i++) dst[i] = (float)src[i];
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "expert.bin";
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 1; }
    int32_t hdr[4];
    if (fread(hdr, 4, 4, f) != 4) { fprintf(stderr, "short header\n"); return 1; }
    int D = hdr[0], I = hdr[1], O13 = hdr[2], O2 = hdr[3];
    printf("expert: D=%d I=%d  gate_up[%d,%d]  down[%d,%d]\n", D, I, O13, D, O2, I);

    size_t n13 = (size_t)O13 * (D / 2), n2 = (size_t)O2 * (I / 2);
    uint8_t *w13 = malloc(n13), *w2 = malloc(n2);
    float *s13 = malloc((size_t)O13 * 4), *s2 = malloc((size_t)O2 * 4);
    if (fread(w13, 1, n13, f) != n13 || fread(s13, 4, O13, f) != (size_t)O13 ||
        fread(w2, 1, n2, f) != n2 || fread(s2, 4, O2, f) != (size_t)O2) {
        fprintf(stderr, "short read\n"); return 1;
    }
    fclose(f);

    /* An activation with realistic magnitudes; the int4 paths quantize it, so a
     * flat or tiny x would understate their error rather than exercise it. */
    float *x = malloc((size_t)D * 4);
    uint32_t r = 22222;
    for (int i = 0; i < D; i++) { r ^= r << 13; r ^= r >> 17; r ^= r << 5; x[i] = (float)((int32_t)r) / 2147483648.0f; }

    /* every output buffer is sized for the LARGER of the two tensors: down has
     * 2048 rows to gate_up's 1024, and sizing on O13 overflows the heap on the
     * second pass (glibc caught it, which is the only reason this is a comment
     * and not a silent wrong answer). */
    int Omax = O13 > O2 ? O13 : O2;
    double *ref = malloc((size_t)Omax * sizeof(double)), *null = malloc((size_t)Omax * sizeof(double));
    float *reff = malloc((size_t)Omax * 4), *nullf = malloc((size_t)Omax * 4);
    float *cpu = malloc((size_t)Omax * 4), *gpu = malloc((size_t)Omax * 4);

    /* coli_cuda_init takes an explicit device list; it refuses NULL/0. GPU_DEV
     * keeps the selection consistent with the engine (device 0 is the MI50). */
    int want = getenv("GPU_DEV") ? atoi(getenv("GPU_DEV")) : 0;
    if (coli_cuda_init(&want, 1) <= 0) { fprintf(stderr, "coli_cuda_init(%d) failed\n", want); return 1; }
    int dev = coli_cuda_device_at(0);
    printf("device %d\n", dev);

    int fail = 0;
    struct { const char *name; uint8_t *w; float *s; int I, O; } t[2] = {
        { "gate_up", w13, s13, D, O13 },
        { "down",    w2,  s2,  I, O2  },
    };
    for (int k = 0; k < 2; k++) {
        int Ik = t[k].I, Ok = t[k].O;
        ref_decode(ref, x, t[k].w, t[k].s, Ik, Ok, 0);
        ref_decode(null, x, t[k].w, t[k].s, Ik, Ok, 1);
        to_f32(reff, ref, Ok); to_f32(nullf, null, Ok);

        matmul_q4(cpu, x, t[k].w, t[k].s, Ik, Ok);

        ColiCudaTensor *tensor = NULL;
        if (!coli_cuda_matmul(&tensor, gpu, x, t[k].w, t[k].s, 2, 1, Ik, Ok, dev, 0)) {
            fprintf(stderr, "%s: coli_cuda_matmul(fmt=2) refused\n", t[k].name); fail++; continue;
        }
        double e_gpu = rel_rms(gpu, reff, Ok);
        double e_cpu = rel_rms(cpu, reff, Ok);
        double e_null = rel_rms(nullf, reff, Ok);
        printf("\n%-8s rel-RMS vs the exact double decode\n", t[k].name);
        printf("  GPU  (backend_cuda fmt=2) : %.5f\n", e_gpu);
        printf("  CPU  (laguna matmul_q4)   : %.5f\n", e_cpu);
        printf("  NULL (nibbles swapped)    : %.5f   <- what a packing mismatch looks like\n", e_null);

        /* The int4 paths quantize the activation, so a few percent is expected and
         * correct; a packing mismatch lands at order 1. The null tells us the test
         * can tell those apart on THIS data rather than in principle. */
        if (e_null < 0.5) { printf("  !! null too small (%.5f) — this test cannot discriminate\n", e_null); fail++; }
        if (e_gpu > 0.10) { printf("  !! GPU disagrees with the format: REPACK NEEDED\n"); fail++; }
        if (e_cpu > 0.10) { printf("  !! CPU disagrees with the format — check the reference\n"); fail++; }
        if (e_gpu <= 0.10 && e_cpu <= 0.10)
            printf("  => same layout, no repack; GPU/CPU agree to %.5f\n", rel_rms(gpu, cpu, Ok));
        coli_cuda_tensor_free(tensor);
    }

    coli_cuda_shutdown();
    printf("\n%s\n", fail ? "EXPERT INT4 COMPAT: FAILED" : "expert int4 compat: ok");
    return fail ? 1 : 0;
}
