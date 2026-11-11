#include "common.h"

// CONCAT of three contiguous tensors as a flat copy in 16-byte units.
//
// Original Exe code; ggml's licence (MIT) applies to this file. It contains no code from any other project.
//
// Along any dimension, a contiguous destination is n/(b0 + b1) blocks, each src0's next b0 units followed by
// src1's next b1 units, so every unit comes from one place found with one division. ggml's kernel computes the
// four-dimensional source index of every element instead.
kernel void kernel_heavy_metal_exe_concat(
        constant ggml_metal_kargs_heavy_metal_exe_concat & args,
        device const uint4 * src0,
        device const uint4 * src1,
        device       uint4 * dst,
        uint   tgpig [[threadgroup_position_in_grid]],
        ushort tpitg [[thread_position_in_threadgroup]],
        ushort ntg   [[threads_per_threadgroup]]) {
    const uint b = args.b0 + args.b1;

    for (uint k = 0; k < HEAVY_METAL_EXE_CONCAT_UPT; ++k) {
        const uint u = (tgpig*HEAVY_METAL_EXE_CONCAT_UPT + k)*ntg + tpitg;
        if (u >= args.n) {
            return;
        }

        const uint o = u / b;
        const uint r = u - o*b;

        dst[u] = r < args.b0 ? src0[o*args.b0 + r] : src1[o*args.b1 + (r - args.b0)];
    }
}
