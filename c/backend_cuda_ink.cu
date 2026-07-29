/* CUDA/HIP backend for inkling.c and laguna.c — see backend_cuda_ink.h for
 * scope. One 32-lane group per output row, weights bf16 (as stored), the
 * activation kept in f32, f32 accumulate.
 *
 * The activation used to be rounded to bf16 first, to copy the numeric
 * contract of matmul_h's _mm512_dpbf16_ps path so GPU and CPU runs stayed
 * comparable. That only holds on a host with AVX512-BF16: matmul_h's scalar
 * fallback multiplies f32 activations by bf16 weights, so on everything else
 * the rounding made the GPU the *less* accurate of the two and broke the
 * comparability it was there to provide. Measured on a 5950X + MI50, worst
 * per-matmul |gpu-cpu| over a 40-layer Laguna XS decode: 1.07 on an element
 * of magnitude 490 (0.22%) with the rounding, 1.4e-4 without — i.e. down to
 * plain f32 summation-order noise. Dropping it costs nothing (the kernel is
 * bound by the weight-row reads) and can only raise accuracy on a vdpbf16ps
 * host too, so it is unconditional rather than a build flag. */
#include <stdio.h>
#include <string.h>
#include "backend_gpu_compat.h"
#include "backend_cuda_ink.h"

static cudaStream_t g_st;
static float *g_dx = nullptr, *g_dy = nullptr;      /* device activation buffers */
static float *g_hx = nullptr, *g_hy = nullptr;      /* PINNED host staging: pageable
                                                     * memcpyAsync degrades to a
                                                     * synchronous bounce copy */
static size_t g_xcap = 0, g_ycap = 0;
static int g_ok = 0;

extern "C" int ink_cuda_init(int dev) {
    /* ~727 tiny matmul calls per decoded token: the default yield-based sync
     * costs ~0.5-1 ms of wakeup latency per call. Spin-sync in the driver is
     * one busy thread — affordable since the OMP team no longer spins. */
    cudaSetDeviceFlags(cudaDeviceScheduleSpin);
    if (cudaSetDevice(dev) != cudaSuccess) return -1;
    if (cudaStreamCreate(&g_st) != cudaSuccess) return -1;
    /* fail fast on a broken runtime instead of at the first matmul */
    void *probe = nullptr;
    if (cudaMalloc(&probe, 1 << 20) != cudaSuccess) return -1;
    cudaFree(probe);
    g_ok = 1;
    return 0;
}

extern "C" size_t ink_cuda_free_bytes(void) {
    size_t fr = 0, to = 0;
    if (cudaMemGetInfo(&fr, &to) != cudaSuccess) return 0;
    return fr;
}

extern "C" size_t ink_cuda_total_bytes(void) {
    size_t fr = 0, to = 0;
    if (cudaMemGetInfo(&fr, &to) != cudaSuccess) return 0;
    return to;
}

extern "C" void ink_cuda_device_name(char *buf, size_t n) {
    if (!n) return;
    buf[0] = 0;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return;
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) return;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
    snprintf(buf, n, "%s (%s)", p.name, p.gcnArchName);
#else
    snprintf(buf, n, "%s (sm_%d%d)", p.name, p.major, p.minor);
#endif
}

extern "C" void *ink_cuda_upload(const void *h, size_t n) {
    void *d = nullptr;
    if (cudaMalloc(&d, n) != cudaSuccess) return nullptr;
    if (cudaMemcpy(d, h, n, cudaMemcpyHostToDevice) != cudaSuccess) { cudaFree(d); return nullptr; }
    return d;
}

/* 4 warps per block, one output row per warp; lanes stride the contraction
 * dim two bf16 at a time (4-byte coalesced loads of the weight row). */
__global__ void mm_bf16_kernel(const __nv_bfloat16 * __restrict__ W,
                               const float * __restrict__ x,
                               float * __restrict__ y, int I, int O) {
    int o = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    int s = blockIdx.y;
    if (o >= O) return;
    const __nv_bfloat16 *w = W + (size_t)o * I;
    const float *xs = x + (size_t)s * I;
    float acc = 0.f;
    for (int i = lane * 2; i < I; i += 64) {
        __nv_bfloat162 wv = *(const __nv_bfloat162 *)(w + i);
        float xa = xs[i];
        float xb = xs[i + 1];
        acc += __bfloat162float(wv.x) * xa + __bfloat162float(wv.y) * xb;
    }
    for (int off = 16; off; off >>= 1) acc += coli_shfl_down32(acc, off);
    if (!lane) y[(size_t)s * O + o] = acc;
}

extern "C" int ink_cuda_matmul_bf16(float *y, const float *x, const void *W, int S, int I, int O) {
    if (!g_ok || (I & 1)) return -1;
    size_t xn = (size_t)S * I * 4, yn = (size_t)S * O * 4;
    if (xn > g_xcap) {
        cudaFree(g_dx); cudaFreeHost(g_hx);
        if (cudaMalloc(&g_dx, xn * 2) != cudaSuccess ||
            cudaMallocHost(&g_hx, xn * 2) != cudaSuccess) { g_dx = g_hx = nullptr; g_xcap = 0; return -1; }
        g_xcap = xn * 2;
    }
    if (yn > g_ycap) {
        cudaFree(g_dy); cudaFreeHost(g_hy);
        if (cudaMalloc(&g_dy, yn * 2) != cudaSuccess ||
            cudaMallocHost(&g_hy, yn * 2) != cudaSuccess) { g_dy = g_hy = nullptr; g_ycap = 0; return -1; }
        g_ycap = yn * 2;
    }
    memcpy(g_hx, x, xn);
    cudaMemcpyAsync(g_dx, g_hx, xn, cudaMemcpyHostToDevice, g_st);
    dim3 grid((unsigned)((O + 3) / 4), (unsigned)S);
    mm_bf16_kernel<<<grid, 128, 0, g_st>>>((const __nv_bfloat16 *)W, g_dx, g_dy, I, O);
    cudaMemcpyAsync(g_hy, g_dy, yn, cudaMemcpyDeviceToHost, g_st);
    if (cudaStreamSynchronize(g_st) != cudaSuccess) return -1;
    memcpy(y, g_hy, yn);
    return 0;
}
