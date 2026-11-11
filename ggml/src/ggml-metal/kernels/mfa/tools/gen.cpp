// Writes the forward attention source that ccv's Metal FlashAttention generator produces for one configuration to
// mfa_attention.metal, and prints the block it chose (OP_FLASH_ATTN_EXT_MFA_* in ggml-metal-impl.h must match).
//
//   gen R H D LOW_PRECISION_INTERMEDIATES SYNC_LOADS F32_QUERIES
//
// R and H only seed the descriptor: the generated kernel takes sequence lengths and strides as function constants.
// Built by build_mfa.sh against a ccv checkout.
#define private public
#include "nnc/mfa/kernels/AttentionDescriptor.hpp"
#include "nnc/mfa/kernels/AttentionKernel.hpp"
#include "nnc/mfa/kernels/AttentionKernelDescriptor.hpp"
#include "nnc/mfa/ccv_nnc_mfa_error.hpp"
#undef private
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>

int main(int argc, char ** argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: %s R H D LOW_PRECISION_INTERMEDIATES SYNC_LOADS F32_QUERIES\n", argv[0]);
        return 2;
    }
    const unsigned R = atoi(argv[1]), H = atoi(argv[2]), D = atoi(argv[3]);
    const bool low_intermediates = atoi(argv[4]) != 0, sync_loads = atoi(argv[5]) != 0, f32_queries = atoi(argv[6]) != 0;

    MTL::Device * device = MTL::CreateSystemDefaultDevice();

    AttentionDescriptor a;
    a.lowPrecisionInputs = true;
    a.isBF16 = false;
    a.lowPrecisionIntermediates = low_intermediates;
    a.matrixDimensions = simd::uint3{R, R, D};
    a.transposeState = simd::uchar4{0, 0, 0, 0};
    a.Hq = H;
    a.Hk = H;
    a.batchDimension = 1;
    a.scale = 1.0f / sqrtf(float(D));
    a.leadingDimensions = simd::uint4{H * D, H * D, H * D, H * D};
    a.type = AttentionKernelType::forward;

    DeviceProperties dprops = DeviceProperties();
    AttentionKernelDescriptor kd = a.kernelDescriptor(device, dprops);
    if (f32_queries) {
        kd.memoryPrecisions[AttentionOperand::Q] = GEMMOperandPrecision::FP32;
        kd.registerPrecisions[AttentionOperand::Q] = GEMMOperandPrecision::FP32;
    }
    // for tuning: MFA_BLOCK="R,C,D" sets the block, MFA_CACHE_Q and MFA_CACHE_O=0/1 whether Q and O stay in registers,
    // MFA_S and MFA_P
    // (16 or 32) the register precision of the scores and of the softmax output
    if (const char * b = getenv("MFA_BLOCK")) {
        unsigned r, c, d;
        if (sscanf(b, "%u,%u,%u", &r, &c, &d) == 3) {
            kd.blockDimensions = simd::ushort3{(unsigned short) r, (unsigned short) c, (unsigned short) d};
        }
    }
    if (const char * c = getenv("MFA_CACHE_Q")) {
        kd.cacheState[AttentionOperand::Q] = atoi(c) != 0;
    }
    if (const char * c = getenv("MFA_CACHE_O")) {
        kd.cacheState[AttentionOperand::O] = atoi(c) != 0;
    }
    if (const char * e = getenv("MFA_S")) {
        kd.registerPrecisions[AttentionOperand::S] = atoi(e) == 16 ? GEMMOperandPrecision::FP16 : GEMMOperandPrecision::FP32;
    }
    if (const char * e = getenv("MFA_P")) {
        kd.registerPrecisions[AttentionOperand::P] = atoi(e) == 16 ? GEMMOperandPrecision::FP16 : GEMMOperandPrecision::FP32;
    }

    AttentionKernel k(kd, device);
    if (sync_loads) {
        k.preferAsyncCache = false;
        k.preferAsyncLoad = false;
        k.disableAsyncCopy = true;
    } else {
        // the constructor's runtime compile rejects the async-copy declarations and falls back to plain loads; the
        // offline build handles them, so regenerate with the descriptor's choice
        k.preferAsyncCache = kd.preferAsyncCache;
        k.preferAsyncLoad = kd.preferAsyncLoad;
        k.disableAsyncCopy = false;
    }
    k.source = k.createSource();

    const auto & cache = kd.cacheState;
    const auto & regs  = kd.registerPrecisions;
    auto prec = [&](AttentionOperand op) {
        const auto p = regs[op];
        return p && p->value == GEMMOperandPrecision::FP16 ? "f16" : "f32";
    };
    printf("block %u %u %u  threads %u  threadgroup memory %u  async cache %d  async load %d  cache Q %d O %d  S %s  P %s\n",
           k.blockDimensions[0], k.blockDimensions[1], k.blockDimensions[2], k.threadgroupSize,
           k.threadgroupMemoryAllocation, k.preferAsyncCache, k.preferAsyncLoad,
           (int) cache[AttentionOperand::Q].value_or(false), (int) cache[AttentionOperand::O].value_or(false),
           prec(AttentionOperand::S), prec(AttentionOperand::P));
    std::ofstream("mfa_attention.metal") << k.source;
    return 0;
}

namespace ccv { namespace nnc { namespace mfa {
void fatal_error(NS::Error * error, int line, const char * file_name, const char * function_name) {
    fprintf(stderr, "error %s:%d %s: %s\n", file_name, line, function_name,
            error ? error->localizedDescription()->utf8String() : "");
    exit(1);
}
void precondition_failure(const char * message, int line, const char * file_name, const char * function_name) {
    fprintf(stderr, "precondition %s:%d %s %s\n", file_name, line, function_name, message ? message : "");
    exit(1);
}
}}}
