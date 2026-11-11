#!/bin/zsh
# Regenerates fa_mfa_dk128.metal, fa_mfa_dk64.metal, fa_mfa_dk512.metal and fa_mfa.metallib in kernels/mfa from a ccv checkout.
#
#   build_mfa.sh CCV_LIB_DIR [OUT_DIR]
#
# CCV_LIB_DIR is ccv's lib directory (https://github.com/liuliu/ccv, commit 3564f9f was used); it needs lib/*.h,
# lib/nnc/*.h, lib/nnc/cmd/*.h and lib/nnc/mfa/. Needs Xcode's Metal toolchain (xcodebuild -downloadComponent
# MetalToolchain).
#
# The generator declares its async copies with __asm labels, which the current Metal compiler rejects. The source is
# therefore compiled to LLVM IR without the labels, the declarations are renamed to the AIR intrinsics they stand for,
# and the IR is built into the library. The log-sum-exp store (only needed by a backward pass) is dropped and the
# kernel for head size D renamed to kernel_flash_attn_ext_mfa_dk<D>. Both head sizes go into one library.
set -e
L=${1:?ccv lib directory}
OUT=${2:-$(dirname $0)/..}
M=$L/nnc/mfa
T=$(mktemp -d)
HERE=$(cd $(dirname $0) && pwd)

for f in AttentionDescriptor AttentionKernel AttentionKernelDescriptor CodeWriter GEMMHeaders "AttentionKernel+Precompiled"; do
  clang++ -std=c++17 -O1 -I$L -I$L/nnc -I$M/3rdparty/metal-cpp -c "$M/kernels/$f.cpp" -o "$T/$f.o"
done
clang++ -std=c++17 -O1 -I$L -I$M/3rdparty/metal-cpp -c $M/Metal.cpp -o $T/metal.o
clang++ -std=c++17 -O1 -I$L -I$L/nnc -I$M/3rdparty/metal-cpp -c $HERE/gen.cpp -o $T/gen.o
clang++ $T/*.o -framework Metal -framework Foundation -framework QuartzCore -o $T/gen

# f32 scores and softmax, f32 queries read directly, async loads; 128 for most DiT models, 64 for the SD/SDXL UNet,
# 512 for the mid-block attention of the SD/SDXL VAE
# The block is set, not taken from the generator's table: 32 queries, 128 keys, 32 head columns per step, Q not held in
# registers, all f32. Measured on M2 Max against the table's 32/80/16 (26.10.): head size 128 -18 to -28 %, 64 -13 to
# -35 %, 512 -34 %.
for D in 128 64 512; do
  (cd $T && MFA_BLOCK=32,128,32 MFA_CACHE_Q=0 ./gen 2304 32 $D 0 0 1)

  sed -E 's/^  __asm\("air\.[^"]+"\);/  ;/' $T/mfa_attention.metal | python3 -c "
import re, sys
s = sys.stdin.read()
s = re.sub(r'\)\n  ;', ');', s)
store = r'\n\s*\(L\)\[[^\n]*= L_sram;'
assert len(re.findall(store, s)) == 1
s = re.sub(store, '', s)
s = s.replace('kernel void attention(', 'kernel void kernel_flash_attn_ext_mfa_dk$D(')
sys.stdout.write('#include <metal_stdlib>\n' + s)" > $OUT/fa_mfa_dk$D.metal

  xcrun -sdk macosx metal -O3 -mmacosx-version-min=13.0 -S -emit-llvm $OUT/fa_mfa_dk$D.metal -o $T/fa$D.ll
  sed -i '' -e 's/@_Z29__metal_wait_simdgroup_eventsiPP18_simdgroup_event_t/@air.wait_simdgroup_events/g' \
    -e 's/@_Z31__metal_simdgroup_async_copy_2dmmPU14MTLthreadgroupvmmDv2_mPU9MTLdeviceKvmmS1_Dv2_li/@air.simdgroup_async_copy_2d.p3i8.p1i8/g' \
    -e 's/@_Z31__metal_simdgroup_async_copy_2dmmPU9MTLdevicevmmDv2_mPU14MTLthreadgroupKvmmS1_Dv2_li/@air.simdgroup_async_copy_2d.p1i8.p3i8/g' $T/fa$D.ll
  if grep -q "__metal_simdgroup_async\|__metal_wait_simdgroup" $T/fa$D.ll; then echo "an async-copy declaration was not renamed" >&2; exit 1; fi
  xcrun -sdk macosx metal -mmacosx-version-min=13.0 -c $T/fa$D.ll -o $T/fa$D.air
done
xcrun -sdk macosx metallib $T/fa128.air $T/fa64.air $T/fa512.air -o $OUT/fa_mfa.metallib
rm -rf $T
