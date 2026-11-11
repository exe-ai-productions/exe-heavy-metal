// C interface to ccv's Conv3DKernel for ggml's Metal backend.
//
// The kernel itself is generated as Metal source at runtime by
// Conv3DKernel::createSource() (liuliu/ccv, BSD-3, see COPYING in this
// directory). This header hides everything Metal from the ggml side: callers
// pass opaque pointers and a plain descriptor.
//
// Adapter code derived from ccv_nnc_mfa_conv3d.cpp, licensed BSD-3 like the
// sources it wraps - see COPYING in this directory.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_metal_conv3d_ccv;

// Output volume of the convolution, NDHWC everywhere. Strides, dilation and
// groups are fixed by the kernel: 1, 1 and 1; the temporal extent is not
// padded (ID = OD + KD - 1), the spatial padding is free.
struct ggml_metal_conv3d_ccv_params {
    int32_t IC;
    int32_t OC;
    int32_t KD;
    int32_t KH;
    int32_t KW;
    int32_t OD;
    int32_t OH;
    int32_t OW;
    int32_t pad_left;
    int32_t pad_right;
    int32_t pad_top;
    int32_t pad_bottom;
};

// Returns the cached kernel for this descriptor and device, or NULL if the
// source could not be compiled. The first call for a descriptor compiles the
// generated Metal source; later calls reuse it.
struct ggml_metal_conv3d_ccv * ggml_metal_conv3d_ccv_get(void * device,
                                                        const struct ggml_metal_conv3d_ccv_params * params);

// Encodes one dispatch into a command encoder: input and output are dense F32
// NDHWC, the weights are dense F32 in [OC][KD][KH][KW][IC] order.
void ggml_metal_conv3d_ccv_encode(struct ggml_metal_conv3d_ccv * kernel,
                                  void * encoder,
                                  void * src, size_t src_offs,
                                  void * w,   size_t w_offs,
                                  void * dst, size_t dst_offs);

#ifdef __cplusplus
}
#endif
