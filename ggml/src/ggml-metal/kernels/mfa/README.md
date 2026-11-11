# Metal FlashAttention forward kernel

`fa_mfa_dk128.metal`, `fa_mfa_dk64.metal` and `fa_mfa_dk512.metal` are the forward attention kernels that ccv's
Metal FlashAttention generator (`lib/nnc/mfa`, https://github.com/liuliu/ccv, commit 3564f9f) writes for head size
128 (most DiT models), 64 (the SD/SDXL UNet) and 512 (the mid-block attention of the SD/SDXL VAE), f32 queries, f16
keys and values, f32 scores, softmax and output; they differ only in the head size. `fa_mfa.metallib` holds all
three, as `kernel_flash_attn_ext_mfa_dk128`, `_dk64` and `_dk512`,
built for macOS 13 and later; `ggml-metal` embeds the library as is (`CMakeLists.txt`) because the runtime compiler
rejects the async-copy declarations the kernel loads its key and value blocks with.

`ggml_metal_op_flash_attn_ext` takes this kernel for unmasked attention without sinks, ALiBi or soft-capping,
with head size 128, 64 or 512, a whole number of query heads per key/value head, at least 32 queries, 16-byte aligned rows, and on the Apple7 and Apple8
GPU families (M1, M2). All three use one block, set in `tools/build_mfa.sh` instead of the generator's table: 32
queries, 128 keys, 32 head columns per step, Q not held in registers, all f32; 128 threads, 8192 bytes of threadgroup
memory (`OP_FLASH_ATTN_EXT_MFA_*` in `ggml-metal-impl.h`). Against the table's 32/80/16 on M2 Max: head size 128
-18 to -28 %, 64 -13 to -35 %, 512 -34 %. `tools/gen.cpp` takes `MFA_BLOCK`, `MFA_CACHE_Q`, `MFA_CACHE_O`, `MFA_S`
and `MFA_P` for such sweeps.
`GGML_METAL_FA_MFA=0` keeps every call on ggml's kernels.

Every head runs as one batch entry of the kernel (`Hq = 1`), so ggml's head strides become its batch strides and
the row strides its leading dimensions; sequence lengths, strides and the scale are function constants
(`ggml_metal_library_get_pipeline_flash_attn_ext_mfa`). The output is written in ggml's `[D, head, query]`
layout. With grouped keys and values (Krea 2: 48 query heads on 12 key/value heads) the key and value batch
strides are 0 and each key/value head gets its own dispatch over its query heads, so query head h reads key/value
head h / (query heads per key/value head), as ggml's other kernels do. The log-sum-exp store, needed only by a backward pass, is removed from the source.

Rebuild with `tools/build_mfa.sh <ccv>/lib`; with Xcode 26.5 its 128 source is the one built before head size 64
was added, apart from the kernel name.
Licences: `LICENSE` (ccv, BSD 3-Clause; Metal FlashAttention, MIT).
