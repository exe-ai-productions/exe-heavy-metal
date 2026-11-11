#include "common.h"

// ROPE_APPLY: rotary position embedding from a precomputed table.
//
// Original Exe code; ggml's licence (MIT) applies to this file. It contains no code from any other project.
//
// x is [d, heads, tokens, batch] with any strides, pe is [2, 2, d/2, tokens] with any strides and holds one 2x2
// rotation per pair and token, dst is contiguous [d, tokens, heads*batch]. The first 2*npair values of a row are
// rotated and the rest copied: the interleaved form rotates the pairs (2j, 2j+1), the other form the pairs
// (j, j + npair). Every thread rotates one pair; a threadgroup covers whole rows
// of several tokens of one head, so the reads of x and the writes of dst are contiguous along a row. When the rows of
// x and the table are packed (the usual case), a pair is read as a float2 and its rotation as a float4.

template <bool interleaved, bool packed>
kernel void kernel_heavy_metal_exe_rope_apply(
        constant ggml_metal_kargs_heavy_metal_exe_rope_apply & args,
        device const char  * x,
        device const char  * pe,
        device       float * dst,
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort3 ntg   [[threads_per_threadgroup]]) {
    const int64_t token = (int64_t) tgpig.x * ntg.y + tpitg.y;
    if (token >= args.ne02) {
        return;
    }
    const int64_t head  = tgpig.y;
    const int64_t batch = tgpig.z;
    const int64_t npair = args.npair;

    device const char * x_row  = x  + head*args.nb01 + token*args.nb02 + batch*args.nb03;
    device const char * pe_row = pe + token*args.pe_nb3;
    device       float * out   = dst + ((batch*args.ne01 + head)*args.ne02 + token)*args.ne00;

    if (packed) {
        device const float  * xr = (device const float  *) x_row;
        device const float4 * mr = (device const float4 *) pe_row;
        for (int64_t j = tpitg.x; j < npair; j += ntg.x) {
            // the 2x2 rotation (column, row): x = (0,0), y = (1,0), z = (0,1), w = (1,1)
            const float4 m = mr[j];
            if (interleaved) {
                const float2 v = ((device const float2 *) xr)[j];
                ((device float2 *) out)[j] = float2(v.x*m.x + v.y*m.y, v.x*m.z + v.y*m.w);
            } else {
                const float a = xr[j];
                const float b = xr[j + npair];
                out[j]         = a*m.x - b*m.z;
                out[j + npair] = a*m.z + b*m.x;
            }
        }
        for (int64_t i = 2*npair + tpitg.x; i < args.ne00; i += ntg.x) {
            out[i] = xr[i];
        }
        return;
    }

    for (int64_t j = tpitg.x; j < npair; j += ntg.x) {
        const int64_t i0 = interleaved ? 2*j     : j;
        const int64_t i1 = interleaved ? 2*j + 1 : j + npair;

        const float a = *(device const float *) (x_row + i0*args.nb00);
        const float b = *(device const float *) (x_row + i1*args.nb00);

        device const char * m = pe_row + j*args.pe_nb2;
        const float c = *(device const float *) (m);
        const float s = *(device const float *) (m + args.pe_nb1);

        if (interleaved) {
            const float ns = *(device const float *) (m + args.pe_nb0);
            const float c1 = *(device const float *) (m + args.pe_nb0 + args.pe_nb1);
            out[i0] = a*c + b*ns;
            out[i1] = a*s + b*c1;
        } else {
            out[i0] = a*c - b*s;
            out[i1] = a*s + b*c;
        }
    }
    for (int64_t i = 2*npair + tpitg.x; i < args.ne00; i += ntg.x) {
        out[i] = *(device const float *) (x_row + i*args.nb00);
    }
}

typedef decltype(kernel_heavy_metal_exe_rope_apply<true, false>) kernel_heavy_metal_exe_rope_apply_t;

template [[host_name("kernel_heavy_metal_exe_rope_apply_f32")]]                    kernel kernel_heavy_metal_exe_rope_apply_t kernel_heavy_metal_exe_rope_apply<false, false>;
template [[host_name("kernel_heavy_metal_exe_rope_apply_f32_interleaved")]]        kernel kernel_heavy_metal_exe_rope_apply_t kernel_heavy_metal_exe_rope_apply<true,  false>;
template [[host_name("kernel_heavy_metal_exe_rope_apply_f32_packed")]]             kernel kernel_heavy_metal_exe_rope_apply_t kernel_heavy_metal_exe_rope_apply<false, true>;
template [[host_name("kernel_heavy_metal_exe_rope_apply_f32_interleaved_packed")]] kernel kernel_heavy_metal_exe_rope_apply_t kernel_heavy_metal_exe_rope_apply<true,  true>;
