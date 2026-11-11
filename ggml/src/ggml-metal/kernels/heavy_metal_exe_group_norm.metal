#include "common.h"

// GROUP_NORM over ggml's contiguous [W][H][C][N] f32 layout.
//
// Original Exe code; ggml's licence (MIT) applies to this file. It contains no code from any other project.
//
// One threadgroup of up to 1024 threads per (group, image). The statistics take one read: every thread sums
// d = x - k and d*d, with k the group's first value as a shift against cancellation when the mean lies far from
// zero; the threadgroup reduces both sums, and a second read writes (x - mean) * scale. The last group may hold
// fewer channels than the others; its mean and variance divide by its own size, as ggml's CPU backend does.

static inline float hm_gn_sum(float v)  { return v; }
static inline float hm_gn_sum(float4 v) { return v.x + v.y + v.z + v.w; }

template <typename T>
kernel void kernel_heavy_metal_exe_group_norm(
        constant ggml_metal_kargs_heavy_metal_exe_group_norm & args,
        device const float * src,
        device       float * dst,
        threadgroup float  * buf [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tpitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        uint2  ntg2  [[threads_per_threadgroup]]) {
    const ushort ntg = ntg2.x;

    const int64_t c0 = (int64_t) tgpig.x * args.cpg;
    const int64_t c1 = min(c0 + (int64_t) args.cpg, args.ne02);
    if (c0 >= c1) {
        return;
    }

    const int64_t base = (int64_t) tgpig.y * args.img + c0 * args.plane;
    const int64_t n    = (c1 - c0) * args.plane;
    const int64_t nv   = n / (int64_t) (sizeof(T) / sizeof(float));

    device const T * x = (device const T *) (src + base);
    device       T * y = (device       T *) (dst + base);

    const float k = src[base];

    float s1 = 0.0f;
    float s2 = 0.0f;
    for (int64_t i = tpitg; i < nv; i += ntg) {
        const T d = x[i] - k;
        s1 += hm_gn_sum(d);
        s2 += hm_gn_sum(d*d);
    }

    s1 = simd_sum(s1);
    s2 = simd_sum(s2);
    if (tiisg == 0) {
        buf[sgitg]               = s1;
        buf[N_SIMDWIDTH + sgitg] = s2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const ushort nsg = (ntg + N_SIMDWIDTH - 1) / N_SIMDWIDTH;
    s1 = simd_sum(tiisg < nsg ? buf[tiisg]               : 0.0f);
    s2 = simd_sum(tiisg < nsg ? buf[N_SIMDWIDTH + tiisg] : 0.0f);

    const float m1    = s1 / (float) n;
    const float var   = max(s2 / (float) n - m1*m1, 0.0f);
    const float mean  = k + m1;
    const float scale = rsqrt(var + args.eps);

    for (int64_t i = tpitg; i < nv; i += ntg) {
        y[i] = (x[i] - mean) * scale;
    }
}

typedef decltype(kernel_heavy_metal_exe_group_norm<float>) kernel_heavy_metal_exe_group_norm_t;

template [[host_name("kernel_heavy_metal_exe_group_norm_f32")]]   kernel kernel_heavy_metal_exe_group_norm_t kernel_heavy_metal_exe_group_norm<float>;
template [[host_name("kernel_heavy_metal_exe_group_norm_f32_4")]] kernel kernel_heavy_metal_exe_group_norm_t kernel_heavy_metal_exe_group_norm<float4>;
