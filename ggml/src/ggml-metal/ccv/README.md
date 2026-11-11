# ccv sources, vendored into ggml's Metal backend

These files come from [liuliu/ccv](https://github.com/liuliu/ccv), BSD-3
(`COPYING` is the upstream licence, unchanged), and implement the conv3d kernel
that ggml's Metal backend now uses for `GGML_OP_CONV_3D`.

The files are copies of the upstream sources at revision
`3564f9f53643d664800f1105781efde18b8f1c38`. Eleven are byte-identical; four
differ in one `#include` line each, because the directory layout is flat here
(`Conv3DKernel.cpp`, `GEMMHeaders.cpp`, `ccv_nnc_mfa_error.cpp`: the umbrella
header `ccv_nnc_mfa.hpp` replaced by `ccv_nnc_mfa_error.hpp`;
`Conv3DKernelDescriptor.cpp`: `../ccv_nnc_mfa_hash.hpp` → `ccv_nnc_mfa_hash.hpp`).

| file | role |
|---|---|
| `Conv3DKernel.cpp/.hpp` | generates the Metal source for one descriptor, builds library and pipeline |
| `Conv3DDescriptor.hpp` | descriptor, unused here except as the type `Conv3DKernel` reads |
| `Conv3DKernelDescriptor.cpp/.hpp` | key of the generator (channels, filter, padding, block size) |
| `CodeWriter.cpp/.hpp` | the template engine used by `createSource()` |
| `GEMMHeaders.cpp/.hpp` | the simdgroup matrix prelude that ends up in the generated source |
| `DeviceProperties.hpp`, `PipelineValue.hpp` | minimal depends of the headers above |
| `ccv_nnc_mfa_error.cpp/.hpp`, `ccv_nnc_mfa_hash.hpp` | error macros, hashing |
| `Metal.cpp` | defines metal-cpp's `*_PRIVATE_IMPLEMENTATION`, one TU only |
| `nnc/mfa/3rdparty/metal-cpp/Metal.hpp` | Apple's metal-cpp (Apache-2.0), single-header amalgamation |

Only the float32 path is used (`dataType = 3`). ccv's half path accumulates in
half and was off by up to 19 absolute on the decoder shapes; a half-input,
float-accumulator variant was accurate but only 2–6 % faster before the
mandatory f32→half cast, so it is not included.

## What is ours, not ccv's

- `ggml-metal-conv3d-ccv.h` / `.mm` — the adapter: caches one kernel per
  descriptor and dispatches it into ggml's command encoder. Same licence.
- `../kernels/conv3d_ccv.metal` — the layout transposes around the kernel.

Apart from the four include lines above, ccv's code is untouched, so a future
upstream update can replace these files and re-apply those lines.
