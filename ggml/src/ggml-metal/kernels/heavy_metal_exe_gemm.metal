#include "common.h"

// conv3d for ggml's own layouts, half operands, float accumulator.
//
// Original Exe code, built up from the Exe lab kernel conv3d_mat_h; ggml's licence
// (MIT) applies to this file. It contains no code from ccv or from any other project.
//
// No im2col, no NDHWC copies. The convolution is an implicit GEMM:
//
//   M = flattened output positions of ONE output frame (64 per threadgroup)
//   N = output channels                                (BN per threadgroup)
//   K = taps x input channels
//
// ggml keeps the activations as [IC][ID][IH][IW] (W contiguous) and the result
// as [OC][OD][OH][OW].
//
// Tiling over the flattened padded plane. With stride 1 the (oh, ow) plane of an
// output frame is a one-dimensional convolution over the padded image read as
// one sequence of S = IW + 2*pad_w positions per row: output position
// f = oh*S + ow reads tap (kh, kw) at padded position g = f + kh*S + kw. A tile
// takes 64 consecutive f, wherever the rows break. The KW-1 trailing columns of
// each row (ow >= OW) are computed and not stored: 2 of 14, 26, 50, 98
// positions for the widths 12, 24, 48, 96, instead of padding every row to a
// multiple of the tile width. The padding itself is virtual: a staged position
// whose (ih, iw) lies outside the image is zero.
//
// Staging: the tile reads, per (kd, kh, kw0) block of KWC taps, one slice of
// 64 + kwc - 1 flat positions per input channel into threadgroup memory
// (converted to half on the way in, 8 channels per 16-byte store) and every tap reads its A fragments from it
// at a shifted row. The element offset of each staged position (one division
// per position, three positions per lane) is computed once per block and kept
// in registers for all channel blocks. A SIMD group stages whole channels: the
// 32 lanes walk consecutive positions, which are contiguous in device memory
// within an image row. The weights are not staged (a SIMD group needs only its
// own output channels); they are read from [OC][KD][KH][KW][IC] half, which a
// small pack kernel produces once per call from ggml's OIDHW weights.
//
// half x half -> float multiply-accumulate with a float accumulator: 27 taps x IC
// is far too long a sum for a half accumulator. Full fragments are written to
// the ggml destination with a transposed simdgroup_store; fragments that touch
// a row break, the dropped columns or the last channels are stored by hand.
//
// Layout of an 8x8 simdgroup_matrix fragment, needed for the hand-filled and
// hand-stored ragged edges (measured with a probe on M2 Max, not documented):
//   lane = (x&1) | ((y&1)<<1) | (((y>>1)&1)<<2) | (((x>>1)&1)<<3) | (((y>>2)&1)<<4)
//   slot = col & 1,   x = col / 2,   y = row
//
// Requirements (checked on the host): stride 1, dilation 1, no temporal
// padding, KH == KW, IC % 8 == 0, tensors contiguous. Apple7 floor: nothing
// here needs Metal 4, bfloat16 or the tensor API.

#define EXE_FM   8     // accumulator fragments per SIMD group along positions: 64 positions per tile
#define EXE_FN   2     // accumulator fragments per SIMD group along channels
#define EXE_BKC  64    // input channels per K block, a multiple of 8
#define EXE_KWC  3     // kw taps staged together

#ifndef EXE_LDA_PAD
#define EXE_LDA_PAD 8
#endif

// f32 -> half saturation at +-65504 in the staging: measured +2.5..4 % kernel time (test-backend-ops
// CONV_3D shapes), so it is off by default; compile with -DEXE_SATURATE=1 to enable
#ifndef EXE_SATURATE
#define EXE_SATURATE 0
#endif

constant constexpr uint BM   = EXE_FM * 8;             // positions per threadgroup
constant constexpr uint ROWS = BM + EXE_KWC - 1;       // rows of the staged slice
constant constexpr uint LDA  = EXE_BKC + EXE_LDA_PAD;  // row stride in halves: a multiple of 8 (16-byte staging
                                                       // stores). PAD 8 measured fastest in the lab (bank conflicts).

static inline uint exe_frag_row(ushort lane) {
    return ((lane >> 1) & 1u) | (((lane >> 2) & 1u) << 1) | (((lane >> 4) & 1u) << 2);
}

static inline uint exe_frag_col(ushort lane, uint slot) {
    const uint x = (lane & 1u) | (((lane >> 3) & 1u) << 1);
    return x * 2u + slot;
}

// One K block: kwc taps x up to BKC channels. A from the staged slice, B from
// the packed weights. OCFULL: all output channels of this SIMD group's tile
// exist, so B comes from plain simdgroup_loads.
template <bool OCFULL>
static inline void exe_mma_block(
        thread simdgroup_matrix<float, 8, 8> (&acc)[EXE_FM][EXE_FN],
        threadgroup const half * As,
        device const half * weights,
        uint bbase, uint ws, uint IC, uint OC, uint oc_base, uint cb,
        uint kwc, ushort lane) {
    device const half * bp0 = weights + (ulong) oc_base * ws + bbase;

    _Pragma("clang loop unroll(full)")
    for (uint kk = 0; kk < EXE_KWC; ++kk) {
        if (kk >= kwc) break;

        _Pragma("clang loop unroll(full)")
        for (uint k8 = 0; k8 < EXE_BKC / 8; ++k8) {
            if (cb + k8 * 8 >= IC) break;   // IC % 8 == 0: a group of 8 is in or out

            simdgroup_matrix<half, 8, 8> a[EXE_FM];
            simdgroup_matrix<half, 8, 8> b[EXE_FN];

            _Pragma("clang loop unroll(full)")
            for (uint i = 0; i < EXE_FM; ++i) {
                // rows = positions, cols = channels: a lane's two elements are adjacent
                simdgroup_load(a[i], As + (i * 8 + kk) * LDA + k8 * 8, LDA, ulong2(0, 0), false);
            }

            _Pragma("clang loop unroll(full)")
            for (uint j = 0; j < EXE_FN; ++j) {
                device const half * bp = bp0 + (ulong) j * 8 * ws + kk * IC + k8 * 8;
                if (OCFULL) {
                    simdgroup_load(b[j], bp, ws, ulong2(0, 0), true);
                } else {
                    // fragment B: row = input channel, col = output channel
                    const uint oc = oc_base + j * 8;
                    const uint k  = exe_frag_row(lane);
                    for (uint e = 0; e < 2; ++e) {
                        const uint ocx = oc + exe_frag_col(lane, e);
                        b[j].thread_elements()[e] = ocx < OC ? bp[(ulong) (ocx - oc) * ws + k] : half(0.0h);
                    }
                }
            }

            _Pragma("clang loop unroll(full)")
            for (uint i = 0; i < EXE_FM; ++i) {
                _Pragma("clang loop unroll(full)")
                for (uint j = 0; j < EXE_FN; ++j) {
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
                }
            }
        }
    }
}

// WN = SIMD groups per threadgroup along the output channels: BN = WN*16
// channels per threadgroup, WN*32 threads.
template <ushort WN>
kernel void kernel_heavy_metal_exe_gemm(
        constant ggml_metal_kargs_heavy_metal_exe_gemm & args,
        device const float * input,       // ggml [IC][ID][IH][IW], f32
        device const half  * weights,     // [OC][KD][KH][KW][IC], half
        device       float * dst,         // ggml [OC][OD][OH][OW], f32
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort sidx [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    constexpr uint BN  = WN * EXE_FN * 8;       // output channels per threadgroup
    constexpr uint GPS = EXE_BKC / 8 / WN;      // 8-channel groups staged per SIMD group and block

    threadgroup half As[ROWS * LDA] __attribute__((aligned(16)));

    const uint IC = args.IC;
    const uint OC = args.OC;
    const uint IW = args.IW;
    const uint OW = args.OW;
    const uint IH = args.IH;

    const uint oc0 = tgpig.x * BN;
    const uint od  = tgpig.y;
    const uint f0  = tgpig.z * BM;              // first flat position of this tile

    const uint S  = IW + 2 * args.pad_w;        // flat row length of the padded image
    const uint PH = IH + 2 * args.pad_h;        // padded rows

    const uint ws = args.KD * args.KH * args.KW * IC;   // weight row stride

    const ulong in_plane  = (ulong) args.ID * IH * IW;
    const ulong out_plane = (ulong) args.OD * args.OH * OW;
    const ulong out_frame = (ulong) od * args.OH * OW;

    const uint n0 = sidx * (EXE_FN * 8);        // this SIMD group's channels inside the tile

    simdgroup_matrix<float, 8, 8> acc[EXE_FM][EXE_FN];
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < EXE_FM; ++i) {
        _Pragma("clang loop unroll(full)")
        for (uint j = 0; j < EXE_FN; ++j) {
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    const bool oc_full = oc0 + BN <= OC;

    for (uint kd = 0; kd < args.KD; ++kd) {
        const uint id = od + kd;    // no temporal padding: ID == OD + KD - 1

        for (uint kh = 0; kh < args.KH; ++kh) {
            for (uint kw0 = 0; kw0 < args.KW; kw0 += EXE_KWC) {
                const uint kwc   = min((uint) EXE_KWC, args.KW - kw0);
                const uint arows = BM + kwc - 1;

                // element offset (inside one channel plane) of the three
                // staged positions of this lane, or -1: outside the image
                int soff[3];
                _Pragma("clang loop unroll(full)")
                for (uint ri = 0; ri < 3; ++ri) {
                    const uint r = lane + ri * 32;
                    int off = -1;
                    if (r < arows) {
                        const uint g   = f0 + kh * S + kw0 + r;
                        const uint ihp = g / S;
                        const int  ih  = int(ihp) - int(args.pad_h);
                        const int  iw  = int(g - ihp * S) - int(args.pad_w);
                        if (ihp < PH && ih >= 0 && ih < int(IH) && iw >= 0 && iw < int(IW)) {
                            off = int(((ulong) id * IH + (uint) ih) * IW + (uint) iw);
                        }
                    }
                    soff[ri] = off;
                }

                for (uint cb = 0; cb < IC; cb += EXE_BKC) {
                    // stage: SIMD group s takes the channel groups s, s+WN, ...
                    // (8 consecutive channels each). A lane owns three
                    // positions; for each it reads the 8 channels of a group
                    // (consecutive lanes = consecutive positions, contiguous in
                    // device memory), converts them to half and writes them as
                    // ONE 16-byte store into the staged row. All loads of a
                    // block are issued before the first store.
                    float v[GPS][3][8];
                    _Pragma("clang loop unroll(full)")
                    for (uint gi = 0; gi < GPS; ++gi) {
                        const uint ch0 = cb + (sidx + gi * WN) * 8;   // IC % 8 == 0: a group is in or out
                        _Pragma("clang loop unroll(full)")
                        for (uint ri = 0; ri < 3; ++ri) {
                            const bool ok = ch0 < IC && soff[ri] >= 0;
                            device const float * src = input + (ulong) ch0 * in_plane + (ok ? soff[ri] : 0);
                            _Pragma("clang loop unroll(full)")
                            for (uint e = 0; e < 8; ++e) {
                                v[gi][ri][e] = ok ? src[(ulong) e * in_plane] : 0.0f;
                            }
                        }
                    }
                    _Pragma("clang loop unroll(full)")
                    for (uint gi = 0; gi < GPS; ++gi) {
                        _Pragma("clang loop unroll(full)")
                        for (uint ri = 0; ri < 3; ++ri) {
                            const uint r = lane + ri * 32;
                            if (r < arows) {
                                // f32 -> half saturates at +-65504 instead of becoming +-inf (an inf
                                // would turn into NaN in the MMA); NaN passes through unchanged
                                float4 flo = float4(v[gi][ri][0], v[gi][ri][1], v[gi][ri][2], v[gi][ri][3]);
                                float4 fhi = float4(v[gi][ri][4], v[gi][ri][5], v[gi][ri][6], v[gi][ri][7]);
#if EXE_SATURATE
                                flo = clamp(flo, -65504.0f, 65504.0f);
                                fhi = clamp(fhi, -65504.0f, 65504.0f);
#endif
                                const half4 lo = half4(flo);
                                const half4 hi = half4(fhi);
                                *((threadgroup uint4 *) (As + r * LDA + (sidx + gi * WN) * 8)) =
                                    uint4(as_type<uint2>(lo), as_type<uint2>(hi));
                            }
                        }
                    }

                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    const uint bbase = ((kd * args.KH + kh) * args.KW + kw0) * IC + cb;
                    if (oc_full) {
                        exe_mma_block<true >(acc, As, weights, bbase, ws, IC, OC, oc0 + n0, cb, kwc, lane);
                    } else {
                        exe_mma_block<false>(acc, As, weights, bbase, ws, IC, OC, oc0 + n0, cb, kwc, lane);
                    }

                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
        }
    }

    // write the float tile into ggml's [OC][OD][OH][OW]. Fragment rows are flat
    // positions f -> (oh, ow = f - oh*S); columns are channels (stride out_plane).
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < EXE_FM; ++i) {
        const uint fu  = f0 + i * 8;                // fragment's first position (uniform)
        const uint ohu = fu / S;
        if (ohu >= args.OH) continue;
        const uint owu = fu - ohu * S;

        const uint fl  = fu + exe_frag_row(lane);   // this lane's position
        const uint ohl = fl / S;
        const uint owl = fl - ohl * S;
        const bool lane_in = ohl < args.OH && owl < OW;
        const ulong lpos = out_frame + (ulong) ohl * OW + owl;

        _Pragma("clang loop unroll(full)")
        for (uint j = 0; j < EXE_FN; ++j) {
            const uint oc = oc0 + n0 + j * 8;
            if (oc >= OC) continue;

            if (owu + 8 <= OW && oc + 8 <= OC) {
                simdgroup_store(acc[i][j], dst + (ulong) oc * out_plane + out_frame + (ulong) ohu * OW + owu,
                                out_plane, ulong2(0, 0), true);
            } else if (lane_in) {
                for (uint e = 0; e < 2; ++e) {
                    const uint ch = oc + exe_frag_col(lane, e);
                    if (ch < OC) {
                        dst[(ulong) ch * out_plane + lpos] = acc[i][j].thread_elements()[e];
                    }
                }
            }
        }
    }
}

#define EXE_INSTANTIATE(WNV) \
template [[host_name("kernel_heavy_metal_exe_gemm_wn" #WNV)]] \
kernel void kernel_heavy_metal_exe_gemm<WNV>( \
    constant ggml_metal_kargs_heavy_metal_exe_gemm & args, \
    device const float * input, \
    device const half  * weights, \
    device       float * dst, \
    uint3  tgpig[[threadgroup_position_in_grid]], \
    ushort sidx [[simdgroup_index_in_threadgroup]], \
    ushort lane [[thread_index_in_simdgroup]]);

EXE_INSTANTIATE(4)
EXE_INSTANTIATE(8)

// Weights: ggml OIDHW (f16 or f32) -> half [OC][KD][KH][KW][IC]. One 32x32 tile
// per threadgroup through threadgroup memory, z = output channel; rows = IC,
// cols = KD*KH*KW. Arguments as for kernel_conv3d_ccv_transpose.
template <typename T>
kernel void kernel_heavy_metal_exe_gemm_pack_w(
        constant ggml_metal_kargs_conv3d_ccv & args,
        device const char * src,
        device       half * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]]) {
    constexpr int TILE = 32;

    threadgroup half tile[TILE][TILE + 1];

    const int64_t row0 = (int64_t) tgpig.y * TILE;
    const int64_t col0 = (int64_t) tgpig.x * TILE;

    const int64_t src_base = (int64_t) tgpig.z * args.src_batch;
    const int64_t dst_base = (int64_t) tgpig.z * args.dst_batch;

    {
        const int64_t row = row0 + tpitg.y;
        const int64_t col = col0 + tpitg.x;

        half v = 0.0h;
        if (row < args.rows && col < args.cols) {
            device const T * s = (device const T *) src;
            v = (half) s[src_base + row*args.src_row_stride + col];
        }
        tile[tpitg.y][tpitg.x] = v;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        const int64_t row = row0 + tpitg.x;
        const int64_t col = col0 + tpitg.y;

        if (row < args.rows && col < args.cols) {
            dst[dst_base + col*args.dst_row_stride + row] = tile[tpitg.x][tpitg.y];
        }
    }
}

template [[host_name("kernel_heavy_metal_exe_gemm_pack_w_f32")]]
kernel void kernel_heavy_metal_exe_gemm_pack_w<float>(
    constant ggml_metal_kargs_conv3d_ccv & args,
    device const char * src,
    device       half * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]]);

template [[host_name("kernel_heavy_metal_exe_gemm_pack_w_f16")]]
kernel void kernel_heavy_metal_exe_gemm_pack_w<half>(
    constant ggml_metal_kargs_conv3d_ccv & args,
    device const char * src,
    device       half * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]]);
