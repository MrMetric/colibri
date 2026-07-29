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

    /* ---- the fused expert pipeline, which is what the engine actually calls ----
     * y = down(silu(gate(x)) * up(x)) for ONE expert, against the same chain run
     * through laguna's CPU kernels. The per-matmul checks above can pass while
     * this fails, because this exercises a different thing: the gate/up split,
     * the argument layout, and whatever the backend does between the matmuls. */
    {
        int Iff = O13 / 2;                       /* gate and up are halves of gate_up */
        size_t half = (size_t)Iff * (D / 2);
        ColiCudaTensor *tg = NULL, *tu = NULL, *td = NULL;
        int up_ok = coli_cuda_tensor_upload(&tg, w13, s13, 2, D, Iff, dev)
                 && coli_cuda_tensor_upload(&tu, w13 + half, s13 + Iff, 2, D, Iff, dev)
                 && coli_cuda_tensor_upload(&td, w2, s2, 2, Iff, D, dev);
        if (!up_ok) { fprintf(stderr, "expert tensor upload failed\n"); fail++; }
        else {
            float *ycpu = malloc((size_t)D * 4), *ygpu = malloc((size_t)D * 4);
            float *gg = malloc((size_t)O13 * 4);
            matmul_q4(gg, x, w13, s13, D, O13);
            for (int i = 0; i < Iff; i++) gg[i] = siluf(gg[i]) * gg[Iff + i];
            matmul_q4(ycpu, gg, w2, s2, Iff, D);

            int rows[1] = { 1 };
            ColiCudaTensor *gs[1] = { tg }, *us[1] = { tu }, *ds[1] = { td };
            if (!coli_cuda_expert_group(gs, us, ds, rows, 1, ygpu, x)) {
                fprintf(stderr, "coli_cuda_expert_group refused\n"); fail++;
            } else {
                double e = rel_rms(ygpu, ycpu, D);
                printf("\nfused expert  rel-RMS GPU vs CPU chain : %.5f\n", e);
                printf("  cpu[0..3] % .5f % .5f % .5f % .5f\n", ycpu[0], ycpu[1], ycpu[2], ycpu[3]);
                printf("  gpu[0..3] % .5f % .5f % .5f % .5f\n", ygpu[0], ygpu[1], ygpu[2], ygpu[3]);
                if (e > 0.10) { printf("  !! fused path disagrees\n"); fail++; }
                else printf("  => fused expert path agrees\n");
            }
            /* ---- same thing with count=8, which is what decode actually does ----
             * The engine groups all K routed experts into one launch. Uploading
             * the SAME expert eight times gives eight distinct tensors with
             * identical contents, so every output row must equal the count=1
             * answer; anything else is the grouping, not the arithmetic. */
            {
                enum { N = 8 };
                ColiCudaTensor *g8[N], *u8[N], *d8[N];
                int r8[N], up8 = 1;
                for (int i = 0; i < N; i++) {
                    g8[i] = u8[i] = d8[i] = NULL; r8[i] = 1;
                    up8 &= coli_cuda_tensor_upload(&g8[i], w13, s13, 2, D, Iff, dev)
                        && coli_cuda_tensor_upload(&u8[i], w13 + half, s13 + Iff, 2, D, Iff, dev)
                        && coli_cuda_tensor_upload(&d8[i], w2, s2, 2, Iff, D, dev);
                }
                float *x8 = malloc((size_t)N * D * 4), *y8 = malloc((size_t)N * D * 4);
                for (int i = 0; i < N; i++) memcpy(x8 + (size_t)i * D, x, (size_t)D * 4);
                if (!up8 || !coli_cuda_expert_group(g8, u8, d8, r8, N, y8, x8)) {
                    fprintf(stderr, "  !! grouped expert_group(count=8) refused\n"); fail++;
                } else {
                    double worst = 0; int bad_row = -1;
                    for (int i = 0; i < N; i++) {
                        double e = rel_rms(y8 + (size_t)i * D, ycpu, D);
                        if (e > worst) { worst = e; bad_row = i; }
                    }
                    printf("\ngrouped x8    worst row rel-RMS vs CPU chain : %.5f (row %d)\n", worst, bad_row);
                    for (int i = 0; i < N; i++)
                        printf("  row %d [0..2] % .5f % .5f % .5f\n", i,
                               y8[(size_t)i*D], y8[(size_t)i*D+1], y8[(size_t)i*D+2]);
                    if (worst > 0.10) { printf("  !! grouping is where it breaks\n"); fail++; }
                    else printf("  => grouping agrees too\n");
                }
                free(x8); free(y8);
                for (int i = 0; i < N; i++) {
                    if (g8[i]) coli_cuda_tensor_free(g8[i]);
                    if (u8[i]) coli_cuda_tensor_free(u8[i]);
                    if (d8[i]) coli_cuda_tensor_free(d8[i]);
                }
            }
            free(ycpu); free(ygpu); free(gg);
            coli_cuda_tensor_free(tg); coli_cuda_tensor_free(tu); coli_cuda_tensor_free(td);
        }
    }

    coli_cuda_shutdown();
    printf("\n%s\n", fail ? "EXPERT INT4 COMPAT: FAILED" : "expert int4 compat: ok");
    return fail ? 1 : 0;
}
