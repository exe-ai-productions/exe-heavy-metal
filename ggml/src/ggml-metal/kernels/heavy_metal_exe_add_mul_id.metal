#include "common.h"

// ADD_MUL_ID: dst = (a + b*s[ids[row]]) + t[ids[row]], row by row.
//
// Original Exe code; ggml's licence (MIT) applies to this file. It contains no code from any other project.
//
// a and b share one shape, s and t hold one row per id with any row stride, ids holds one row id per a->ne[1];
// without ids, s and t are broadcast over a by their own shape and strides. The
// product is rounded before the sums, as in a MUL kernel followed by two ADD kernels, so the result has the same bits:
// contraction into a fused multiply-add is switched off for this file. One threadgroup covers one row; packed rows
// (16-byte aligned, a multiple of 4 values) are read as float4.

#pragma METAL fp contract(off)

template <typename T, bool shift>
kernel void kernel_heavy_metal_exe_add_mul_id(
        constant ggml_metal_kargs_heavy_metal_exe_add_mul_id & args,
        device const char * a,
        device const char * b,
        device const char * s,
        device const char * ids,
        device const char * t,
        device       char * dst,
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort3 ntg   [[threads_per_threadgroup]]) {
    const int64_t i1 = tgpig.x;
    const int64_t i2 = tgpig.y;
    const int64_t i3 = tgpig.z;

    device const T * pa = (device const T *) (a + i1*args.nb01 + i2*args.nb02 + i3*args.nb03);
    device const T * pb = (device const T *) (b + i1*args.nb11 + i2*args.nb12 + i3*args.nb13);
    device const T * ps;
    device const T * pt;
    if (args.has_ids) {
        const int32_t row = *((device const int32_t *) (ids + i1*args.nb_ids));
        ps = (device const T *) (s + row*args.nb_s[0]);
        pt = (device const T *) (t + row*args.nb_t[0]);
    } else {
        ps = (device const T *) (s + (i1 % args.ne_s[0])*args.nb_s[0] + (i2 % args.ne_s[1])*args.nb_s[1] +
                                 (i3 % args.ne_s[2])*args.nb_s[2]);
        pt = (device const T *) (t + (i1 % args.ne_t[0])*args.nb_t[0] + (i2 % args.ne_t[1])*args.nb_t[1] +
                                 (i3 % args.ne_t[2])*args.nb_t[2]);
    }
    device       T * pd = (device       T *) (dst + i1*args.nb1 + i2*args.nb2 + i3*args.nb3);

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        const T m = pb[i0]*ps[i0];
        T v = pa[i0] + m;
        if (shift) {
            v = v + pt[i0];
        }
        pd[i0] = v;
    }
}

typedef decltype(kernel_heavy_metal_exe_add_mul_id<float, true>) kernel_heavy_metal_exe_add_mul_id_t;

template [[host_name("kernel_heavy_metal_exe_add_mul_id_f32")]]         kernel kernel_heavy_metal_exe_add_mul_id_t kernel_heavy_metal_exe_add_mul_id<float,  false>;
template [[host_name("kernel_heavy_metal_exe_add_mul_id_f32_shift")]]   kernel kernel_heavy_metal_exe_add_mul_id_t kernel_heavy_metal_exe_add_mul_id<float,  true>;
template [[host_name("kernel_heavy_metal_exe_add_mul_id_f32_4")]]       kernel kernel_heavy_metal_exe_add_mul_id_t kernel_heavy_metal_exe_add_mul_id<float4, false>;
template [[host_name("kernel_heavy_metal_exe_add_mul_id_f32_4_shift")]] kernel kernel_heavy_metal_exe_add_mul_id_t kernel_heavy_metal_exe_add_mul_id<float4, true>;
