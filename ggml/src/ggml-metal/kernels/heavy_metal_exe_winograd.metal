#include "common.h"

// conv3d by Winograd F(4x4, 3x3) over the spatial taps, for ggml's own layouts.
//
// Original Exe code, adapted from the Exe lab kernel conv3d_wino (Metal Labor); ggml's
// licence (MIT) applies to this file. It contains no code from ccv or from any other project.
//
// A 3x3 plane convolution that produces a 4x4 output tile from a 6x6 input tile needs
// 36 multiplications per (input channel, output channel) instead of 9*16 = 144. The
// depth taps (kd = 0..KD-1) are NOT transformed: they stay a plain sum, K = KD * IC.
//
// Four kernels, run back to back for every (output-channel chunk, depth chunk):
//   winograd_weight : U[p][oc][kd*IC + ic] = (G g G^T)[p]          once per output-channel chunk
//   winograd_input  : V[p][fl*T + t][ic]   = (B^T d B)[p]          reads ggml [IC][ID][IH][IW] directly
//   winograd_gemm   : Y[p][oc][m]          = sum_kd sum_ic V[p][m + kd*T][ic] * U[p][oc][kd*IC + ic]
//                 m = odl*T + t is the flat (output frame, tile) index of the chunk, so the depth
//                 tap kd is a shift of kd*T rows
//   winograd_output : dst[oc][od][oh][ow] = A^T Y A                writes ggml [OC][OD][OH][OW] directly
// T = ceil(OH/4) * ceil(OW/4) tiles per frame; a frame too large for the scratch budget is split into runs of
// consecutive tiles (t_base, tiles), which the GEMM sees as smaller frames. Spatial padding 0 or 1 (a caller
// that pads the input itself, like a causal conv, passes 0), no temporal padding, stride 1, KH=KW=3 and
// KD = 3 or 1 (a 2D conv seen as a 3D conv of depth 1).
// H and W need not be multiples of 4: the edge tiles read zeros outside the frame (the input
// transform already does that for the padding) and the output transform writes only the pixels
// inside the frame.
//
// All intermediates are float: with half intermediates F(4,3) is too inaccurate (the lab
// measured rms 1.7e-3 of the signal rms with half Y, against 2e-4 for float).
// Y is stored transposed ([oc][m], m padded to a multiple of 64): the GEMM stores it with a
// transposed simdgroup_store, and the output transform then reads it with lanes along m,
// which is also the contiguous direction of the destination row.

#define WINO_AL  6                       // input tile edge
#define WINO_NP  36                      // transform points
#define WINO_BKC 64                      // input channels per K block
#define WINO_PAD 8

static inline void winograd_bt(thread const float * d, thread float * t) {
    t[0] =  4.0f * d[0] - 5.0f * d[2] + d[4];
    t[1] = -4.0f * d[1] - 4.0f * d[2] + d[3] + d[4];
    t[2] =  4.0f * d[1] - 4.0f * d[2] - d[3] + d[4];
    t[3] = -2.0f * d[1] - d[2] + 2.0f * d[3] + d[4];
    t[4] =  2.0f * d[1] - d[2] - 2.0f * d[3] + d[4];
    t[5] =  4.0f * d[1] - 5.0f * d[3] + d[5];
}

static inline void winograd_g(thread const float * g, thread float * u) {
    u[0] = g[0] * 0.25f;
    u[1] = -(g[0] + g[1] + g[2]) * (1.0f / 6.0f);
    u[2] = -(g[0] - g[1] + g[2]) * (1.0f / 6.0f);
    u[3] = g[0] * (1.0f / 24.0f) + g[1] * (1.0f / 12.0f) + g[2] * (1.0f / 6.0f);
    u[4] = g[0] * (1.0f / 24.0f) - g[1] * (1.0f / 12.0f) + g[2] * (1.0f / 6.0f);
    u[5] = g[2];
}

static inline void winograd_at(thread const float * y, thread float * o) {
    o[0] = y[0] + y[1] + y[2] + y[3] + y[4];
    o[1] = y[1] - y[2] + 2.0f * y[3] - 2.0f * y[4];
    o[2] = y[1] + y[2] + 4.0f * y[3] + 4.0f * y[4];
    o[3] = y[1] - y[2] + 8.0f * y[3] - 8.0f * y[4] + y[5];
}

static inline uint winograd_frag_row(ushort lane) {
    return ((lane >> 1) & 1u) | (((lane >> 2) & 1u) << 1) | (((lane >> 4) & 1u) << 2);
}

// weights, ggml OIDHW (f16 or f32): element (oc, ic, kd, kh, kw) at (oc*IC + ic)*KD*9 + (kd*3 + kh)*3 + kw.
// grid: x = input channel, y = output channel of the chunk, z = kd
template <typename T>
kernel void kernel_heavy_metal_exe_winograd_weight(
        constant ggml_metal_kargs_heavy_metal_exe_winograd & args,
        device const T     * w,
        device       float * U,
        uint3 gid[[thread_position_in_grid]]) {
    const uint ic = gid.x;
    const uint ocl = gid.y;
    const uint kd = gid.z;
    if (ic >= args.IC || ocl >= args.OCc) {
        return;
    }

    const ulong oc = (ulong) args.oc_base + ocl;
    device const T * src = w + (oc * args.IC + ic) * (args.KD * 9) + kd * 9;

    float g[3][3], t[3][WINO_AL], u[WINO_AL][WINO_AL];
    for (uint kh = 0; kh < 3; ++kh) {
        for (uint kw = 0; kw < 3; ++kw) {
            g[kh][kw] = float(src[kh * 3 + kw]);
        }
    }
    for (uint kh = 0; kh < 3; ++kh) {
        float r[WINO_AL];
        winograd_g(g[kh], r);
        for (uint j = 0; j < WINO_AL; ++j) t[kh][j] = r[j];
    }
    for (uint j = 0; j < WINO_AL; ++j) {
        float col[3] = {t[0][j], t[1][j], t[2][j]};
        float r[WINO_AL];
        winograd_g(col, r);
        for (uint i = 0; i < WINO_AL; ++i) u[i][j] = r[i];
    }

    const ulong k3 = (ulong) args.KD * args.IC;
    for (uint i = 0; i < WINO_AL; ++i) {
        for (uint j = 0; j < WINO_AL; ++j) {
            U[((ulong) (i * WINO_AL + j) * args.OCc + ocl) * k3 + kd * args.IC + ic] = u[i][j];
        }
    }
}

#define WINOGRAD_WEIGHT_INSTANTIATE(NAME, T) \
template [[host_name(NAME)]] \
kernel void kernel_heavy_metal_exe_winograd_weight<T>( \
    constant ggml_metal_kargs_heavy_metal_exe_winograd & args, \
    device const T     * w, \
    device       float * U, \
    uint3 gid[[thread_position_in_grid]]);

WINOGRAD_WEIGHT_INSTANTIATE("kernel_heavy_metal_exe_winograd_weight_f32", float)
WINOGRAD_WEIGHT_INSTANTIATE("kernel_heavy_metal_exe_winograd_weight_f16", half)

// input transform. grid: x = input channel quad (4 channels), y = tile, z = local input frame.
// 4 channels per thread: consecutive lanes write consecutive 16-byte pieces of one V row.
kernel void kernel_heavy_metal_exe_winograd_input(
        constant ggml_metal_kargs_heavy_metal_exe_winograd & args,
        device const float * in,
        device       float * V,
        uint3 gid[[thread_position_in_grid]]) {
    const uint ic = gid.x * 4;
    const uint t  = gid.y;
    const uint fl = gid.z;
    if (ic >= args.IC || t >= args.tiles || fl >= args.odc + args.KD - 1) {
        return;
    }

    const uint tg = args.t_base + t;
    const uint th = tg / args.tiles_w;
    const uint tw = tg - th * args.tiles_w;
    const int h0 = int(th * 4) - args.pad_h;
    const int w0 = int(tw * 4) - args.pad_w;

    const ulong plane = (ulong) args.ID * args.IH * args.IW;
    const ulong frame = (ulong) (args.od_base + fl) * args.IH * args.IW;

    float d[WINO_AL][WINO_AL][4];
    for (uint c = 0; c < 4; ++c) {
        device const float * src = in + (ulong) (ic + c) * plane + frame;
        for (uint a = 0; a < WINO_AL; ++a) {
            const int ih = h0 + int(a);
            const bool hok = ih >= 0 && ih < int(args.IH);
            for (uint b = 0; b < WINO_AL; ++b) {
                const int iw = w0 + int(b);
                d[a][b][c] = (hok && iw >= 0 && iw < int(args.IW)) ? src[(ulong) ih * args.IW + iw] : 0.0f;
            }
        }
    }

    float o[WINO_AL][WINO_AL][4];
    for (uint c = 0; c < 4; ++c) {
        float tmp[WINO_AL][WINO_AL];
        for (uint a = 0; a < WINO_AL; ++a) {
            float row[WINO_AL], r[WINO_AL];
            for (uint b = 0; b < WINO_AL; ++b) row[b] = d[a][b][c];
            winograd_bt(row, r);
            for (uint j = 0; j < WINO_AL; ++j) tmp[a][j] = r[j];
        }
        for (uint j = 0; j < WINO_AL; ++j) {
            float col[WINO_AL], r[WINO_AL];
            for (uint a = 0; a < WINO_AL; ++a) col[a] = tmp[a][j];
            winograd_bt(col, r);
            for (uint i = 0; i < WINO_AL; ++i) o[i][j][c] = r[i];
        }
    }

    const ulong row = (ulong) fl * args.tiles + t;
    for (uint i = 0; i < WINO_AL; ++i) {
        for (uint j = 0; j < WINO_AL; ++j) {
            *reinterpret_cast<device float4 *>(V + ((ulong) (i * WINO_AL + j) * args.v_rows + row) * args.IC + ic) =
                float4(o[i][j][0], o[i][j][1], o[i][j][2], o[i][j][3]);
        }
    }
}

// GEMM over the 36 points: BM = FM*8 rows (positions) x BN = WN*FN*8 output channels per
// threadgroup, one SIMD group per FN*8 channels. A is staged in threadgroup memory, B is read
// straight from device memory; float x float -> float. grid: x = channel tile, y = row tile, z = point.
template <ushort FM, ushort FN, ushort WN, bool TAIL>
kernel void kernel_heavy_metal_exe_winograd_gemm(
        constant ggml_metal_kargs_heavy_metal_exe_winograd & args,
        device const float * V,
        device const float * U,
        device       float * Y,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tid  [[thread_index_in_threadgroup]],
        ushort sidx [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    constexpr uint BM  = FM * 8;
    constexpr uint BN  = WN * FN * 8;
    constexpr uint NT  = WN * 32;
    constexpr uint LDA = WINO_BKC + WINO_PAD;
    constexpr uint CV  = WINO_BKC / 4;                    // float4 items per staged row
    constexpr uint SLOTS = (BM * CV + NT - 1) / NT;

    threadgroup float As[BM * LDA] __attribute__((aligned(16)));

    const uint oc0 = tgpig.x * BN;
    const uint m0  = tgpig.y * BM;
    const uint pt  = tgpig.z;
    const uint IC  = args.IC;
    const uint OCc = args.OCc;
    const ulong k3 = (ulong) args.KD * IC;
    const uint nw0 = sidx * (FN * 8);

    device const float * Vp = V + (ulong) pt * args.v_rows * IC;
    device const float * Up = U + (ulong) pt * OCc * k3;

    simdgroup_matrix<float, 8, 8> acc[FM][FN];
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < FM; ++i) {
        _Pragma("clang loop unroll(full)")
        for (uint j = 0; j < FN; ++j) {
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    for (uint kd = 0; kd < uint(args.KD); ++kd) {
        for (uint cb = 0; cb < IC; cb += WINO_BKC) {
            // TAIL: IC is a multiple of 8, not of WINO_BKC; the last block holds kc channels and its K steps stop
            // there, so B is never read past this depth tap's channels. Without TAIL every block is full.
            const uint kc = TAIL ? min(uint(WINO_BKC), IC - cb) : uint(WINO_BKC);
            _Pragma("clang loop unroll(full)")
            for (uint s = 0; s < SLOTS; ++s) {
                const uint idx = tid + s * NT;
                const uint r = idx / CV;
                const uint cq = idx - r * CV;
                if (r < BM) {
                    float4 v = float4(0.0f);
                    const uint m = m0 + r;
                    if (m < args.m_total && (!TAIL || cq * 4 < kc)) {
                        v = *reinterpret_cast<device const float4 *>(Vp + (ulong) (m + kd * args.tiles) * IC + cb + cq * 4);
                    }
                    *reinterpret_cast<threadgroup float4 *>(As + r * LDA + cq * 4) = v;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            device const float * bp0 = Up + (ulong) (oc0 + nw0) * k3 + kd * IC + cb;

            _Pragma("clang loop unroll(full)")
            for (uint k8 = 0; k8 < WINO_BKC / 8; ++k8) {
                if (TAIL && k8 * 8 >= kc) {
                    break;
                }
                simdgroup_matrix<float, 8, 8> a[FM];
                simdgroup_matrix<float, 8, 8> b[FN];
                _Pragma("clang loop unroll(full)")
                for (uint i = 0; i < FM; ++i) {
                    simdgroup_load(a[i], As + (i * 8) * LDA + k8 * 8, LDA, ulong2(0, 0), false);
                }
                _Pragma("clang loop unroll(full)")
                for (uint j = 0; j < FN; ++j) {
                    simdgroup_load(b[j], bp0 + (ulong) (j * 8) * k3 + k8 * 8, k3, ulong2(0, 0), true);
                }
                _Pragma("clang loop unroll(full)")
                for (uint i = 0; i < FM; ++i) {
                    _Pragma("clang loop unroll(full)")
                    for (uint j = 0; j < FN; ++j) {
                        simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Y[p][oc][m], m padded to a multiple of 64 (a whole fragment is always inside the buffer;
    // the rows beyond m_total came from zero-staged A rows and are never read)
    device float * Yp = Y + (ulong) pt * OCc * args.m_pad;
    _Pragma("clang loop unroll(full)")
    for (uint i = 0; i < FM; ++i) {
        _Pragma("clang loop unroll(full)")
        for (uint j = 0; j < FN; ++j) {
            simdgroup_store(acc[i][j], Yp + (ulong) (oc0 + nw0 + j * 8) * args.m_pad + m0 + i * 8,
                            args.m_pad, ulong2(0, 0), true);
        }
    }
}

#define WINOGRAD_GEMM_INSTANTIATE(NAME, FM, FN, WN, TAIL) \
template [[host_name(NAME)]] \
kernel void kernel_heavy_metal_exe_winograd_gemm<FM, FN, WN, TAIL>( \
    constant ggml_metal_kargs_heavy_metal_exe_winograd & args, \
    device const float * V, \
    device const float * U, \
    device       float * Y, \
    uint3  tgpig[[threadgroup_position_in_grid]], \
    ushort tid  [[thread_index_in_threadgroup]], \
    ushort sidx [[simdgroup_index_in_threadgroup]], \
    ushort lane [[thread_index_in_simdgroup]]);

WINOGRAD_GEMM_INSTANTIATE("kernel_heavy_metal_exe_winograd_gemm_bn128",      8, 2, 8, false)   // BM 64, BN 128
WINOGRAD_GEMM_INSTANTIATE("kernel_heavy_metal_exe_winograd_gemm_bn64",       8, 2, 4, false)   // BM 64, BN 64 (SD/SDXL UNet: 320, 960)
WINOGRAD_GEMM_INSTANTIATE("kernel_heavy_metal_exe_winograd_gemm_bn48",       4, 2, 3, false)   // BM 32, BN 48 (the decoder's conv_out)
WINOGRAD_GEMM_INSTANTIATE("kernel_heavy_metal_exe_winograd_gemm_bn128_tail", 8, 2, 8, true)    // the same with IC % 64 != 0
WINOGRAD_GEMM_INSTANTIATE("kernel_heavy_metal_exe_winograd_gemm_bn64_tail",  8, 2, 4, true)
WINOGRAD_GEMM_INSTANTIATE("kernel_heavy_metal_exe_winograd_gemm_bn48_tail",  4, 2, 3, true)

// output transform. grid: x = flat (frame, tile) index m of the chunk (lanes along m), y = output channel of the chunk.
kernel void kernel_heavy_metal_exe_winograd_output(
        constant ggml_metal_kargs_heavy_metal_exe_winograd & args,
        device const float * Y,
        device       float * dst,
        uint3 gid[[thread_position_in_grid]]) {
    const uint m = gid.x;
    const uint ocl = gid.y;
    if (m >= args.m_total || ocl >= args.OCc) {
        return;
    }

    float y[WINO_AL][WINO_AL];
    for (uint p = 0; p < WINO_NP; ++p) {
        y[p / WINO_AL][p % WINO_AL] = Y[((ulong) p * args.OCc + ocl) * args.m_pad + m];
    }

    float tmp[WINO_AL][4];
    for (uint a = 0; a < WINO_AL; ++a) {
        float r[4];
        winograd_at(y[a], r);
        for (uint j = 0; j < 4; ++j) tmp[a][j] = r[j];
    }

    const uint odl = m / args.tiles;
    const uint t = args.t_base + (m - odl * args.tiles);
    const uint th = t / args.tiles_w;
    const uint tw = t - th * args.tiles_w;

    const ulong oc = (ulong) args.oc_base + ocl;
    const ulong od = (ulong) args.od_base + odl;
    device float * out = dst + ((oc * args.OD + od) * args.OH + th * 4) * args.OW + tw * 4;

    float o[4][4];
    for (uint j = 0; j < 4; ++j) {
        float col[WINO_AL], r[4];
        for (uint a = 0; a < WINO_AL; ++a) col[a] = tmp[a][j];
        winograd_at(col, r);
        for (uint i = 0; i < 4; ++i) o[i][j] = r[i];
    }

    // stride 2: the tile's even rows and columns are the 2x2 block (th*2, tw*2) of the strided destination
    if (args.ostride == 2) {
        device float * out2 = dst + ((oc * args.OD + od) * args.SOH + th * 2) * args.SOW + tw * 2;
        const uint nh2 = min(2u, uint(args.SOH) - th * 2);
        const uint nw2 = min(2u, uint(args.SOW) - tw * 2);
        for (uint i = 0; i < nh2; ++i) {
            for (uint j = 0; j < nw2; ++j) {
                out2[(ulong) i * args.SOW + j] = o[2 * i][2 * j];
            }
        }
        return;
    }

    // edge tiles: only the rows and columns inside the frame; a row of 4 is one 16-byte store
    // when the row pitch keeps it aligned (OW % 4 == 0), else 4 single stores
    const uint nh = min(4u, uint(args.OH) - th * 4);
    const uint nw = min(4u, uint(args.OW) - tw * 4);
    const bool vec = nw == 4 && (args.OW & 3) == 0;
    for (uint i = 0; i < nh; ++i) {
        device float * row = out + (ulong) i * args.OW;
        if (vec) {
            *reinterpret_cast<device float4 *>(row) = float4(o[i][0], o[i][1], o[i][2], o[i][3]);
        } else {
            for (uint j = 0; j < nw; ++j) row[j] = o[i][j];
        }
    }
}
