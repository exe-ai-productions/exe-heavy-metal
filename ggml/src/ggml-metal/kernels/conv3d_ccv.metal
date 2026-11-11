#include "common.h"

// Layout conversions for the ccv conv3d path (see ../ccv/README.md).
//
// ggml keeps conv3d activations channel-major - in memory [N][IC][ID][IH][IW] -
// and conv3d weights in PyTorch OIDHW order, while ccv's Conv3DKernel reads and
// writes dense channel-last NDHWC. All three conversions are a transpose of a
// dense matrix, so a single shared-memory tiled kernel covers them:
//
//   activations in:  rows = IC,         cols = ID*IH*IW     -> [ID][IH][IW][IC]
//   weights:         rows = IC,         cols = KD*KH*KW, z = oc
//                                                           -> [OC][KD][KH][KW][IC]
//   result out:      rows = OD*OH*OW,   cols = OC           -> [OC][OD][OH][OW]
//
// The weights are F16 in ggml and F32 in ccv's kernel, hence two source types.
// ggml's licence (MIT) applies to this file: no ccv code, only the layout
// conversion around ccv's kernel.

template <typename T>
kernel void kernel_conv3d_ccv_transpose(
        constant ggml_metal_kargs_conv3d_ccv & args,
        device const char * src,
        device       float * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]]) {
    constexpr int TILE = 32;

    threadgroup float tile[TILE][TILE + 1];

    const int64_t row0 = (int64_t) tgpig.y * TILE;
    const int64_t col0 = (int64_t) tgpig.x * TILE;

    const int64_t src_base = (int64_t) tgpig.z * args.src_batch;
    const int64_t dst_base = (int64_t) tgpig.z * args.dst_batch;

    // load: consecutive threads walk the source's innermost dimension
    {
        const int64_t row = row0 + tpitg.y;
        const int64_t col = col0 + tpitg.x;

        float v = 0.0f;
        if (row < args.rows && col < args.cols) {
            device const T * s = (device const T *) src;
            v = (float) s[src_base + row*args.src_row_stride + col];
        }
        tile[tpitg.y][tpitg.x] = v;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // store: consecutive threads walk the destination's innermost dimension
    {
        const int64_t row = row0 + tpitg.x;
        const int64_t col = col0 + tpitg.y;

        if (row < args.rows && col < args.cols) {
            dst[dst_base + col*args.dst_row_stride + row] = tile[tpitg.x][tpitg.y];
        }
    }
}

template [[host_name("kernel_conv3d_ccv_transpose_f32")]]
kernel void kernel_conv3d_ccv_transpose<float>(
    constant ggml_metal_kargs_conv3d_ccv & args,
    device const char * src,
    device       float * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]]);

template [[host_name("kernel_conv3d_ccv_transpose_f16")]]
kernel void kernel_conv3d_ccv_transpose<half>(
    constant ggml_metal_kargs_conv3d_ccv & args,
    device const char * src,
    device       float * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]]);
