#include "common.h"

// Copies between two tensors of the same shape into a contiguous destination (CONT, CPY, DUP),
// for float and half. The generic copy kernel derives every destination index with four 64-bit
// divisions per element and reads a permuted source across memory; both kernels below avoid that.
//
// Original Exe code; ggml's licence (MIT) applies to this file.
//
//   kernel_cpy_row_*  : the source's dim 0 is also its fastest dim. One thread per element of a
//                       row, several rows per threadgroup when rows are short; no divisions
//                       except one by ne2 per thread.
//   kernel_cpy_tile_* : the source's fastest dim is a != 0 (a transpose). A 32 x 32 tile of the
//                       (dim 0, dim a) plane goes through threadgroup memory, so the reads are
//                       contiguous along a and the writes contiguous along dim 0.
//   kernel_cpy_inner_*: as tile, but dim a and a second dim b are so short (ne[a]*ne[b] <= 16)
//                       that each thread copies the whole (a, b) block of one element.
//   kernel_cpy_narrow_*: as tile, but dim 0 is short (a pixel shuffle's factor): each thread copies
//                       the ne[0] values of one element along a.

#define CPY_TILE 32
#define CPY_ROWS 8

template <typename T0, typename T1>
kernel void kernel_cpy_row(
        constant ggml_metal_kargs_cpy_fast & args,
        device const char * src,
        device       char * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const uint i0 = tgpig.x*ntg.x + tpitg.x;
    const uint i1 = tgpig.y*ntg.y + tpitg.y;
    if (i0 >= uint(args.ne[0]) || i1 >= uint(args.ne[1])) {
        return;
    }
    const uint i3 = tgpig.z / uint(args.ne[2]);
    const uint i2 = tgpig.z - i3*uint(args.ne[2]);

    device const T0 * s = (device const T0 *) (src + i0*args.nbs[0] + i1*args.nbs[1] + i2*args.nbs[2] + i3*args.nbs[3]);
    device       T1 * d = (device       T1 *) (dst + i0*args.nbd[0] + i1*args.nbd[1] + i2*args.nbd[2] + i3*args.nbd[3]);
    *d = (T1) *s;
}

template <typename T0, typename T1>
kernel void kernel_cpy_tile(
        constant ggml_metal_kargs_cpy_fast & args,
        device const char * src,
        device       char * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]]) {
    threadgroup T1 tile[CPY_TILE][CPY_TILE + 1];

    const int a = args.a, b = args.b, c = args.c;

    const uint n0 = args.ne[0];
    const uint na = args.ne[a];
    const uint nb = args.ne[b];

    const uint i00 = tgpig.x*CPY_TILE;
    const uint ia0 = tgpig.y*CPY_TILE;
    const uint ic  = tgpig.z / nb;
    const uint ib  = tgpig.z - ic*nb;

    device const char * s0 = src + ib*args.nbs[b] + ic*args.nbs[c];
    device       char * d0 = dst + ib*args.nbd[b] + ic*args.nbd[c];

    const uint tx = tpitg.x;
    const uint ty = tpitg.y;

    // read along a (the source's contiguous dim)
    for (uint k = 0; k < CPY_TILE; k += CPY_ROWS) {
        const uint i0 = i00 + ty + k;
        const uint ia = ia0 + tx;
        if (i0 < n0 && ia < na) {
            tile[ty + k][tx] = (T1) *((device const T0 *) (s0 + i0*args.nbs[0] + ia*args.nbs[a]));
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // write along dim 0 (the destination's contiguous dim)
    for (uint k = 0; k < CPY_TILE; k += CPY_ROWS) {
        const uint ia = ia0 + ty + k;
        const uint i0 = i00 + tx;
        if (i0 < n0 && ia < na) {
            *((device T1 *) (d0 + i0*args.nbd[0] + ia*args.nbd[a])) = tile[tx][ty + k];
        }
    }
}

// The source's contiguous dim a != 0 is short, and so is the dim b looped with it (ne[a]*ne[b] <= 16):
// a tile would leave most of its threads idle. Each thread takes one (dim 0, dim c) element and copies
// the whole small (a, b) block, which lies together in the source, into its planes of the destination;
// neighbouring threads write neighbouring elements of every plane.
template <typename T0, typename T1>
kernel void kernel_cpy_inner(
        constant ggml_metal_kargs_cpy_fast & args,
        device const char * src,
        device       char * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int a = args.a, b = args.b, c = args.c;

    // one flat index over (dim 0, dim c), dim 0 fastest: dim c can be far longer than a grid axis
    // (the host keeps ne[0]*ne[c] within 32 bits)
    const uint i  = tgpig.x*ntg.x + tpitg.x;
    const uint n0 = args.ne[0];
    if (i >= n0*uint(args.ne[c])) {
        return;
    }
    const uint ic = i / n0;
    const uint i0 = i - ic*n0;

    device const char * s0 = src + i0*args.nbs[0] + ic*args.nbs[c];
    device       char * d0 = dst + i0*args.nbd[0] + ic*args.nbd[c];

    for (int ib = 0; ib < args.ne[b]; ++ib) {
        for (int ia = 0; ia < args.ne[a]; ++ia) {
            *((device T1 *) (d0 + ia*args.nbd[a] + ib*args.nbd[b])) =
                (T1) *((device const T0 *) (s0 + ia*args.nbs[a] + ib*args.nbs[b]));
        }
    }
}

// Dim 0 is short (ne[0] <= GGML_METAL_CPY_NARROW on the host) and the source's contiguous dim a != 0 long: a tile
// would fill only ne[0] of its 32 rows. Each thread takes one element along a and copies its ne[0] values;
// neighbouring threads read neighbouring source elements.
template <typename T0, typename T1>
kernel void kernel_cpy_narrow(
        constant ggml_metal_kargs_cpy_fast & args,
        device const char * src,
        device       char * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int a = args.a, b = args.b, c = args.c;

    const uint ia = tgpig.x*ntg.x + tpitg.x;
    if (ia >= uint(args.ne[a])) {
        return;
    }
    const uint nb = args.ne[b];
    const uint ic = tgpig.z / nb;
    const uint ib = tgpig.z - ic*nb;

    device const char * s0 = src + ia*args.nbs[a] + ib*args.nbs[b] + ic*args.nbs[c];
    device       char * d0 = dst + ia*args.nbd[a] + ib*args.nbd[b] + ic*args.nbd[c];

    for (int i0 = 0; i0 < args.ne[0]; ++i0) {
        *((device T1 *) (d0 + i0*args.nbd[0])) = (T1) *((device const T0 *) (s0 + i0*args.nbs[0]));
    }
}

#define CPY_FAST_INSTANTIATE(S0, S1, T0, T1) \
template [[host_name("kernel_cpy_row_" S0 "_" S1)]] kernel void kernel_cpy_row<T0, T1>( \
    constant ggml_metal_kargs_cpy_fast & args, device const char * src, device char * dst, \
    uint3 tgpig[[threadgroup_position_in_grid]], uint3 tpitg[[thread_position_in_threadgroup]], \
    uint3 ntg[[threads_per_threadgroup]]); \
template [[host_name("kernel_cpy_tile_" S0 "_" S1)]] kernel void kernel_cpy_tile<T0, T1>( \
    constant ggml_metal_kargs_cpy_fast & args, device const char * src, device char * dst, \
    uint3 tgpig[[threadgroup_position_in_grid]], uint3 tpitg[[thread_position_in_threadgroup]]); \
template [[host_name("kernel_cpy_inner_" S0 "_" S1)]] kernel void kernel_cpy_inner<T0, T1>( \
    constant ggml_metal_kargs_cpy_fast & args, device const char * src, device char * dst, \
    uint3 tgpig[[threadgroup_position_in_grid]], uint3 tpitg[[thread_position_in_threadgroup]], \
    uint3 ntg[[threads_per_threadgroup]]); \
template [[host_name("kernel_cpy_narrow_" S0 "_" S1)]] kernel void kernel_cpy_narrow<T0, T1>( \
    constant ggml_metal_kargs_cpy_fast & args, device const char * src, device char * dst, \
    uint3 tgpig[[threadgroup_position_in_grid]], uint3 tpitg[[thread_position_in_threadgroup]], \
    uint3 ntg[[threads_per_threadgroup]]);

CPY_FAST_INSTANTIATE("f32", "f32", float, float)
CPY_FAST_INSTANTIATE("f32", "f16", float, half)
CPY_FAST_INSTANTIATE("f16", "f16", half,  half)
CPY_FAST_INSTANTIATE("f16", "f32", half,  float)
