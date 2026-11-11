#include <metal_stdlib>

// -*- Metal -*-
//===-- metal_simdgroup_event ---------------------------------------------===//
// Copyright (c) 2024 Philip Turner. See MIT LICENSE
//===----------------------------------------------------------------------===//

#ifndef __METAL_SIMDGROUP_EVENT
#define __METAL_SIMDGROUP_EVENT

// Invoking the generation of LLVM bitcode for async copies.
//
//   %struct._simdgroup_event_t = type opaque
//
struct _simdgroup_event_t;

// Invoking the generation of LLVM bitcode for async copies.
//
//   Bitcode: TBD
//
thread _simdgroup_event_t*
__metal_simdgroup_async_copy_1d(
  ulong, ulong, threadgroup void *, const device void *, ulong);

// Invoking the generation of LLVM bitcode for async copies.
//
//   Bitcode: TBD
//
thread _simdgroup_event_t*
__metal_simdgroup_async_copy_1d(
  ulong, ulong, device void *, const threadgroup void *, ulong);

// Invoking the generation of LLVM bitcode for async copies.
//
//   ; Function Attrs: argmemonly convergent nounwind
//   declare %struct._simdgroup_event_t*
//     @air.simdgroup_async_copy_2d.p3i8.p1i8(
//       i64, i64,
//       i8 addrspace(3)* nocapture writeonly, i64, i64, <2 x i64>,
//       i8 addrspace(1)* nocapture readonly,  i64, i64, <2 x i64>,
//       <2 x i64>, i32)
//     local_unnamed_addr #4
//
thread _simdgroup_event_t*
__metal_simdgroup_async_copy_2d(
  ulong, ulong,
  threadgroup void *, ulong, ulong, ulong2,
  const device void *, ulong, ulong, ulong2,
  long2, int);

// Invoking the generation of LLVM bitcode for async copies.
//
//   ; Function Attrs: argmemonly convergent nounwind
//   declare %struct._simdgroup_event_t*
//     @air.simdgroup_async_copy_2d.p1i8.p3i8(
//       i64, i64,
//       i8 addrspace(1)* nocapture writeonly, i64, i64, <2 x i64>,
//       i8 addrspace(3)* nocapture readonly,  i64, i64, <2 x i64>,
//       <2 x i64>, i32)
//     local_unnamed_addr #4
//
thread _simdgroup_event_t*
__metal_simdgroup_async_copy_2d(
  ulong, ulong,
  device void *, ulong, ulong, ulong2,
  const threadgroup void *, ulong, ulong, ulong2,
  long2, int);

// Invoking the generation of LLVM bitcode for async copies.
//
//   ; Function Attrs: convergent nounwind
//   declare void
//     @air.wait_simdgroup_events(i32, %struct._simdgroup_event_t** nocapture)
//     local_unnamed_addr #3
//
void __metal_wait_simdgroup_events(
  int, thread _simdgroup_event_t**);

#pragma METAL internals : enable
namespace metal
{
  enum class simdgroup_async_copy_clamp_mode {
    clamp_to_zero = 0,
    clamp_to_edge = 1
  };

  struct simdgroup_event {
    METAL_FUNC simdgroup_event() thread {}

    template <ushort dst_elements_per_row, ushort threadgroup_size, simdgroup_async_copy_clamp_mode clamp_mode = simdgroup_async_copy_clamp_mode::clamp_to_zero, typename T>
    METAL_FUNC void async_copy(
      // Description of the destination.
      threadgroup T *dst,
      ushort2 dst_tile_dimensions,

      // Description of the source.
      const device T *src,
      uint src_elements_per_row,
      ushort2 src_tile_dimensions,
      bool transpose_matrix = false
    ) thread {
      if (transpose_matrix) {
        src_tile_dimensions = src_tile_dimensions.yx;
        dst_tile_dimensions = dst_tile_dimensions.yx;
      }
      event = __metal_simdgroup_async_copy_2d(
        // Description of the data type.
        sizeof(T),
        alignof(T),

        // Description of the destination.
        reinterpret_cast<threadgroup void *>(dst),
        ushort(dst_elements_per_row),
        1,
        ulong2(dst_tile_dimensions),

        // Description of the source.
        reinterpret_cast<const device void *>(src),
        uint(src_elements_per_row),
        1,
        ulong2(src_tile_dimensions),

        // Other arguments.
        long2(0),
        static_cast<int>(clamp_mode));
    }

    template <ushort src_elements_per_row, ushort threadgroup_size, simdgroup_async_copy_clamp_mode clamp_mode = simdgroup_async_copy_clamp_mode::clamp_to_zero, typename T>
    METAL_FUNC void async_copy(
      // Description of the destination.
      device T *dst,
      uint dst_elements_per_row,
      ushort2 dst_tile_dimensions,

      // Description of the source.
      const threadgroup T *src,
      ushort2 src_tile_dimensions,
      bool transpose_matrix = false
    ) thread {
      if (transpose_matrix) {
        src_tile_dimensions = src_tile_dimensions.yx;
        dst_tile_dimensions = dst_tile_dimensions.yx;
      }
      event = __metal_simdgroup_async_copy_2d(
        // Description of the data type.
        sizeof(T),
        alignof(T),

        // Description of the destination.
        reinterpret_cast<device void *>(dst),
        uint(dst_elements_per_row),
        1,
        ulong2(dst_tile_dimensions),

        // Description of the source.
        reinterpret_cast<const threadgroup void *>(src),
        ushort(src_elements_per_row),
        1,
        ulong2(src_tile_dimensions),

        // Other arguments.
        long2(0),
        0);
    }

    template <ushort threadgroup_size, typename T>
    METAL_FUNC void async_copy(
      // Description of the destination.
      threadgroup T *dst,
      ushort dst_tile_dimensions,

      // Description of the source.
      const device T *src,
      ushort src_tile_dimensions
    ) thread {
      async_copy<1, threadgroup_size>(dst, ushort2(dst_tile_dimensions, 1), src, 1, ushort2(src_tile_dimensions, 1));
    }

    template <ushort threadgroup_size, typename T>
    METAL_FUNC void async_copy(
      // Description of the destination.
      device T *dst,
      ushort dst_tile_dimensions,

      // Description of the source.
      const threadgroup T *src,
      ushort src_tile_dimensions
    ) thread {
      async_copy<1, threadgroup_size>(dst, 1, ushort2(dst_tile_dimensions, 1), src, ushort2(src_tile_dimensions, 1));
    }

    METAL_FUNC static void wait(int count, thread simdgroup_event *events) {
      __metal_wait_simdgroup_events(
        count, reinterpret_cast<thread _simdgroup_event_t**>(events));
    }

  private:
    // Invoking the generation of LLVM bitcode for async copies.
    //
    //   %"struct.metal::simdgroup_event" = type { %struct._simdgroup_event_t* }
    //
    thread _simdgroup_event_t* event;
  };
} // namespace metal
#pragma METAL internals : disable

#endif // __METAL_SIMDGROUP_EVENT



// -*- Metal -*-
//===-- metal_simdgroup_matrix_storage ------------------------------------===//
// Copyright (c) 2024 Philip Turner. See MIT LICENSE
//===----------------------------------------------------------------------===//

#ifndef __METAL_SIMDGROUP_MATRIX_STORAGE
#define __METAL_SIMDGROUP_MATRIX_STORAGE

// The layout of threads within a SIMD matrix.
//
//  0  0  1  1  8  8  9  9
//  2  2  3  3 10 10 11 11
//  4  4  5  5 12 12 13 13
//  6  6  7  7 14 14 15 15
// 16 16 17 17 24 24 25 25
// 18 18 19 19 26 26 27 27
// 20 20 21 21 28 28 29 29
// 22 22 23 23 30 30 31 31
//
// This is Morton order, a method for coalescing data accesses. It is used
// in a variety of contexts, from ray tracing acceleration structures, to
// nodal-point Laplacians, to sorting large lattices of atoms.
//
// Source: https://patents.google.com/patent/US11256518B2
METAL_FUNC static ushort2 morton_order(ushort thread_index_in_simdgroup) {
  ushort lane_id = thread_index_in_simdgroup;
  ushort quad_id = lane_id / 4;

  constexpr ushort QUADRANT_SPAN_M = 4;
  constexpr ushort THREADS_PER_QUADRANT = 8;
  ushort M_floor_of_quadrant = (quad_id / 4) * QUADRANT_SPAN_M;
  ushort M_in_quadrant = (lane_id / 2) % (THREADS_PER_QUADRANT / 2);
  ushort M_in_simd = M_floor_of_quadrant + M_in_quadrant;

  ushort N_floor_of_quadrant = (quad_id & 2) * 2; // 0 or 4
  ushort N_in_quadrant = (lane_id % 2) * 2; // 0 or 2
  ushort N_in_simd = N_floor_of_quadrant + N_in_quadrant;

  return ushort2(N_in_simd, M_in_simd);
}

#pragma METAL internals : enable
namespace metal
{
  template <typename T>
  struct simdgroup_matrix_storage {
    typedef vec<T, 64> storage_type;

    storage_type t;

    METAL_FUNC thread vec<T, 2>* thread_elements() thread {
      return reinterpret_cast<thread vec<T, 2>*>(&t);
    }

    METAL_FUNC simdgroup_matrix_storage() thread = default;

    METAL_FUNC simdgroup_matrix_storage(vec<T, 2> thread_elements) thread {
      *(this->thread_elements()) = thread_elements;
    }

    METAL_FUNC static device T* apply_offset(device T *src, uint elements_per_row, uint2 matrix_origin, bool transpose_matrix = false) {
      if (transpose_matrix) {
        return src + ulong(matrix_origin.x * elements_per_row) + matrix_origin.y;
      } else {
        return src + ulong(matrix_origin.y * elements_per_row) + matrix_origin.x;
      }
    }

    METAL_FUNC static threadgroup T* apply_offset(threadgroup T *src, ushort elements_per_row, ushort2 matrix_origin, bool transpose_matrix = false) {
      if (transpose_matrix) {
        return src + matrix_origin.x * elements_per_row + matrix_origin.y;
      } else {
        return src + matrix_origin.y * elements_per_row + matrix_origin.x;
      }
    }

    template <typename U>
    METAL_FUNC void load(const device U *src, uint elements_per_row, ushort2 matrix_origin, bool transpose_matrix = false) thread {
      if (transpose_matrix) {
        uint address0 = uint(matrix_origin.x + 0) * elements_per_row + uint(matrix_origin.y);
        uint address1 = uint(matrix_origin.x + 1) * elements_per_row + uint(matrix_origin.y);
        U memoryForm0 = src[address0];
        U memoryForm1 = src[address1];
        ((thread T*)thread_elements())[0] = T(memoryForm0);
        ((thread T*)thread_elements())[1] = T(memoryForm1);
      } else if (elements_per_row % 2 != 0) {
        uint address0 = uint(matrix_origin.y) * elements_per_row + uint(matrix_origin.x + 0);
        uint address1 = uint(matrix_origin.y) * elements_per_row + uint(matrix_origin.x + 1);
        U memoryForm0 = src[address0];
        U memoryForm1 = src[address1];
        ((thread T*)thread_elements())[0] = T(memoryForm0);
        ((thread T*)thread_elements())[1] = T(memoryForm1);
      } else {
        auto combinedAddress = uint(matrix_origin.y) * elements_per_row + uint(matrix_origin.x + 0);
        vec<U, 2> memoryForm = *(const device vec<U, 2>*)(src + combinedAddress);
        *(thread_elements()) = vec<T, 2>(memoryForm);
      }
    }

    template <typename U>
    METAL_FUNC void load(const threadgroup U *src, ushort elements_per_row, ushort2 matrix_origin, bool transpose_matrix = false) thread {
      if (transpose_matrix) {
        ushort address0 = ushort(matrix_origin.x + 0) * elements_per_row + ushort(matrix_origin.y);
        ushort address1 = ushort(matrix_origin.x + 1) * elements_per_row + ushort(matrix_origin.y);
        U memoryForm0 = src[address0];
        U memoryForm1 = src[address1];
        ((thread T*)thread_elements())[0] = T(memoryForm0);
        ((thread T*)thread_elements())[1] = T(memoryForm1);
      } else if (elements_per_row % 2 != 0) {
        ushort address0 = ushort(matrix_origin.y) * elements_per_row + ushort(matrix_origin.x + 0);
        ushort address1 = ushort(matrix_origin.y) * elements_per_row + ushort(matrix_origin.x + 1);
        U memoryForm0 = src[address0];
        U memoryForm1 = src[address1];
        ((thread T*)thread_elements())[0] = T(memoryForm0);
        ((thread T*)thread_elements())[1] = T(memoryForm1);
      } else {
        auto combinedAddress = ushort(matrix_origin.y) * elements_per_row + ushort(matrix_origin.x + 0);
        vec<U, 2> memoryForm = *(const threadgroup vec<U, 2>*)(src + combinedAddress);
        *(thread_elements()) = vec<T, 2>(memoryForm);
      }
    }

    template <typename U>
    METAL_FUNC void store(device U *dst, uint elements_per_row, ushort2 matrix_origin, bool transpose_matrix = false) thread {
      if (transpose_matrix) {
        uint address0 = uint(matrix_origin.x + 0) * elements_per_row + uint(matrix_origin.y);
        uint address1 = uint(matrix_origin.x + 1) * elements_per_row + uint(matrix_origin.y);
        T registerForm0 = ((thread T*)thread_elements())[0];
        T registerForm1 = ((thread T*)thread_elements())[1];
        dst[address0] = U(registerForm0);
        dst[address1] = U(registerForm1);
      } else if (elements_per_row % 2 != 0) {
        uint address0 = uint(matrix_origin.y) * elements_per_row + uint(matrix_origin.x + 0);
        uint address1 = uint(matrix_origin.y) * elements_per_row + uint(matrix_origin.x + 1);
        T registerForm0 = ((thread T*)thread_elements())[0];
        T registerForm1 = ((thread T*)thread_elements())[1];
        dst[address0] = U(registerForm0);
        dst[address1] = U(registerForm1);
      } else {
        auto combinedAddress = uint(matrix_origin.y) * elements_per_row + uint(matrix_origin.x + 0);
        vec<T, 2> registerForm = *(thread_elements());
        *(device vec<U, 2>*)(dst + combinedAddress) = vec<U, 2>(registerForm);
      }
    }

    template <typename U>
    METAL_FUNC void store(threadgroup U *dst, ushort elements_per_row, ushort2 matrix_origin, bool transpose_matrix = false) thread {
      if (transpose_matrix) {
        ushort address0 = ushort(matrix_origin.x + 0) * elements_per_row + ushort(matrix_origin.y);
        ushort address1 = ushort(matrix_origin.x + 1) * elements_per_row + ushort(matrix_origin.y);
        T registerForm0 = ((thread T*)thread_elements())[0];
        T registerForm1 = ((thread T*)thread_elements())[1];
        dst[address0] = U(registerForm0);
        dst[address1] = U(registerForm1);
      } else if (elements_per_row % 2 != 0) {
        ushort address0 = ushort(matrix_origin.y) * elements_per_row + ushort(matrix_origin.x + 0);
        ushort address1 = ushort(matrix_origin.y) * elements_per_row + ushort(matrix_origin.x + 1);
        T registerForm0 = ((thread T*)thread_elements())[0];
        T registerForm1 = ((thread T*)thread_elements())[1];
        dst[address0] = U(registerForm0);
        dst[address1] = U(registerForm1);
      } else {
        auto combinedAddress = ushort(matrix_origin.y) * elements_per_row + ushort(matrix_origin.x + 0);
        vec<T, 2> registerForm = *(thread_elements());
        *(threadgroup vec<U, 2>*)(dst + combinedAddress) = vec<U, 2>(registerForm);
      }
    }


    template <typename U, typename V>
    METAL_FUNC void multiply(simdgroup_matrix_storage<U> a, simdgroup_matrix_storage<V> b, bool accumulate = true) thread {
      if (!accumulate) {
        *(thread_elements()) = vec<T, 2>(0);
      }
      t = __metal_simdgroup_matrix_8x8_multiply_accumulate(a.t, b.t, t, typename simdgroup_matrix_storage<T>::storage_type());
    }
  };
} // namespace metal
#pragma METAL internals : disable

#endif // __METAL_SIMDGROUP_MATRIX_STORAGE



using namespace metal;


constant uint R [[function_constant(0)]];
constant uint C [[function_constant(1)]];


    // R = row dimension (output sequence)
    // C = column dimension (input sequence)
    // Hq = number of query heads.

    constant uint Hq [[function_constant(2)]];
    constant uint H_Hk_ratio [[function_constant(3)]];

	constant float dot_product_scale_derivative [[function_constant(4)]];
	constant float dot_product_scale = dot_product_scale_derivative * 1.442695041;

  constant uint Q_batch_stride [[function_constant(5)]];
  constant uint Q_leading_dimension [[function_constant(15)]];
  constant uint K_batch_stride [[function_constant(6)]];
  constant uint K_leading_dimension [[function_constant(16)]];
  constant uint V_batch_stride [[function_constant(7)]];
  constant uint V_leading_dimension [[function_constant(17)]];
  constant uint O_batch_stride [[function_constant(8)]];
  constant uint O_leading_dimension [[function_constant(18)]];
  constant bool attention_sinks [[function_constant(28)]];



    
    // Declare the function.
    kernel void kernel_flash_attn_ext_mfa_dk64(

  device float* Q [[buffer(0)]],
  device half* K [[buffer(1)]],
  device half* V [[buffer(2)]],
  device float* O [[buffer(3)]],
  device float* L [[buffer(4)]],
  device const float* Sinks_buf [[buffer(19), function_constant(attention_sinks)]],
  constant uint& Sink_head_stride [[buffer(20), function_constant(attention_sinks)]],



      threadgroup uchar *threadgroup_block [[threadgroup(0)]],
      
      uint3 gid [[threadgroup_position_in_grid]],
      ushort sidx [[simdgroup_index_in_threadgroup]],
      ushort lane_id [[thread_index_in_simdgroup]]
    ) {
      ushort2 morton_offset = morton_order(lane_id);
      gid = { gid.x % ((R + 32 - 1) / 32), (gid.x / ((R + 32 - 1) / 32)) % Hq, gid.x / (Hq * ((R + 32 - 1) / 32))};
      uint parallelization_group_offset = gid.x;
      parallelization_group_offset *= 32;
      
      // Return early if the entire SIMD is out of bounds.
      if (parallelization_group_offset >= R) {
        return;
      }


    Q = Q + gid.z * Q_batch_stride + gid.y * 64;


    K = K + gid.z * K_batch_stride + gid.y / H_Hk_ratio * 64;


    V = V + gid.z * V_batch_stride + gid.y / H_Hk_ratio * 64;


    O = O + gid.z * O_batch_stride + gid.y * 64;


    L = L + (gid.z * Hq + gid.y) * R;





    float m = attention_sinks ? (float)Sinks_buf[gid.y * Sink_head_stride] * 1.442695041 : -numeric_limits<float>::max();
    float l = attention_sinks ? 1 : numeric_limits<float>::denorm_min();






  // Outer loop over the traversal dimension.
  for (uint c = 0; c < C; c += 128) {
    // S = Q * K^T
    


    simdgroup_matrix_storage<float> S_sram[128 / 8];





    #pragma clang loop unroll(full)
    for (ushort c = 0; c < 128; c += 8) {
      auto S = S_sram + c / 8;
      *S = simdgroup_matrix_storage<float>(0);
    }





    #pragma clang loop unroll(disable)
    for (
      ushort d_outer = 0;
      d_outer < 64;
      d_outer += 32
    ) {
      

    if ((
        (C % 128 == 0) ||
        (c + 128 <= C)
      ) && (
        (64 % 8 == 0) ||
        (d_outer + 32 <= 64)
      )) {
      


    simdgroup_matrix_storage<float> Q_sram[32 / 8];





      

      uint2 Q_src_offset(
        morton_offset.x + d_outer,
        min(parallelization_group_offset + sidx * 8 + morton_offset.y, R - 1));
      auto Q_src = simdgroup_matrix_storage<float>
      ::apply_offset(
        Q, Q_leading_dimension,
        Q_src_offset, false);




      #pragma clang loop unroll(full)
      for (ushort d = 0; d < 32; d += 8) {
        ushort2 Q_origin(d, 0);
        Q_sram[d / 8].load(
          Q_src, Q_leading_dimension,
          Q_origin, false);
      }





      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        uint2 K_offset(d_outer, c);
        auto src = simdgroup_matrix_storage<half>
        ::apply_offset(
          K, K_leading_dimension,
          K_offset, false);
        auto dst = (threadgroup half*)(threadgroup_block);
 
        ushort D_src_dimension = min(
          ushort(32),
          ushort(64 - d_outer));
        ushort D_dst_dimension = 32;
        ushort C_src_dimension = min(
          uint(128),
          uint(C - c));
        ushort C_dst_dimension = max(
          ushort((((C % 128 == 0) ? 128 : C % 128) + 7) / 8 * 8),
          ushort(C_src_dimension));
        ushort2 tile_src(D_src_dimension, C_src_dimension);
        ushort2 tile_dst(D_dst_dimension, C_dst_dimension);

        simdgroup_event event;
        event.async_copy<32, 32>(
          dst, tile_dst,
          src, K_leading_dimension, tile_src, false);
        simdgroup_event::wait(1, &event);
      }

      

      ushort2 K_block_offset(
        morton_offset.x,
        morton_offset.y);
      auto K_src = (threadgroup half*)(threadgroup_block);
      K_src = simdgroup_matrix_storage<half>
      ::apply_offset(
        K_src, 32,
        K_block_offset, true);
      threadgroup_barrier(mem_flags::mem_threadgroup);








      #pragma clang loop unroll(full)
      for (ushort d = 0; d < 32; d += 8) {
        

    #pragma clang loop unroll(full)
    for (ushort c = 0; c < 128; c += 8) {
      
      // Load the RHS from memory.
      ushort2 K_origin(c, d);
      simdgroup_matrix_storage<half> K;
      K.load(
        K_src, 32,
        K_origin, true);

      // Issue one SIMD matmul instruction.
      S_sram[c / 8].multiply(
        Q_sram[(0 + d) / 8],
        K, true);
    }



      }




    } else {
      


    simdgroup_matrix_storage<float> Q_sram[32 / 8];





      

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sidx == 0) {
      uint2 Q_offset(d_outer, parallelization_group_offset);
      auto src = simdgroup_matrix_storage<float>
      ::apply_offset(
        Q, Q_leading_dimension,
        Q_offset, false);
      auto dst = (threadgroup float*)(threadgroup_block);

      ushort D_src_dimension = min(
        ushort(32),
        ushort(64 - d_outer));
      ushort D_dst_dimension = 32;
      ushort R_dimension = min(
        uint(32),
        uint(R - parallelization_group_offset));
      ushort2 tile_src(D_src_dimension, R_dimension);
      ushort2 tile_dst(D_dst_dimension, R_dimension);

      simdgroup_event event;
      event.async_copy<32, 32>(
        dst, tile_dst,
        src, Q_leading_dimension, tile_src, false);
      simdgroup_event::wait(1, &event);
    }



      

      ushort2 Q_block_offset(
        morton_offset.x, 
        morton_offset.y + sidx * 8);
      auto Q_src = (threadgroup float*)(threadgroup_block);
      Q_src = simdgroup_matrix_storage<float>
      ::apply_offset(
        Q_src, 32,
        Q_block_offset, false);
      threadgroup_barrier(mem_flags::mem_threadgroup);




      #pragma clang loop unroll(full)
      for (ushort d = 0; d < 32; d += 8) {
        ushort2 Q_origin(d, 0);
        Q_sram[d / 8].load(
          Q_src, 32,
          Q_origin, false);
      }





      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        uint2 K_offset(d_outer, c);
        auto src = simdgroup_matrix_storage<half>
        ::apply_offset(
          K, K_leading_dimension,
          K_offset, false);
        auto dst = (threadgroup half*)(threadgroup_block);
 
        ushort D_src_dimension = min(
          ushort(32),
          ushort(64 - d_outer));
        ushort D_dst_dimension = 32;
        ushort C_src_dimension = min(
          uint(128),
          uint(C - c));
        ushort C_dst_dimension = max(
          ushort((((C % 128 == 0) ? 128 : C % 128) + 7) / 8 * 8),
          ushort(C_src_dimension));
        ushort2 tile_src(D_src_dimension, C_src_dimension);
        ushort2 tile_dst(D_dst_dimension, C_dst_dimension);

        simdgroup_event event;
        event.async_copy<32, 32>(
          dst, tile_dst,
          src, K_leading_dimension, tile_src, false);
        simdgroup_event::wait(1, &event);
      }

      

      ushort2 K_block_offset(
        morton_offset.x,
        morton_offset.y);
      auto K_src = (threadgroup half*)(threadgroup_block);
      K_src = simdgroup_matrix_storage<half>
      ::apply_offset(
        K_src, 32,
        K_block_offset, true);
      threadgroup_barrier(mem_flags::mem_threadgroup);








      #pragma clang loop unroll(full)
      for (ushort d = 0; d < 32; d += 8) {
        

    #pragma clang loop unroll(full)
    for (ushort c = 0; c < (((C % 128 == 0) ? 128 : C % 128) + 7) / 8 * 8; c += 8) {
      
      // Load the RHS from memory.
      ushort2 K_origin(c, d);
      simdgroup_matrix_storage<half> K;
      K.load(
        K_src, 32,
        K_origin, true);

      // Issue one SIMD matmul instruction.
      S_sram[c / 8].multiply(
        Q_sram[(0 + d) / 8],
        K, true);
    }



        if (c + 128
            < C) {
          

    #pragma clang loop unroll(full)
    for (ushort c = (((C % 128 == 0) ? 128 : C % 128) + 7) / 8 * 8; c < 128; c += 8) {
      
      // Load the RHS from memory.
      ushort2 K_origin(c, d);
      simdgroup_matrix_storage<half> K;
      K.load(
        K_src, 32,
        K_origin, true);

      // Issue one SIMD matmul instruction.
      S_sram[c / 8].multiply(
        Q_sram[(0 + d) / 8],
        K, true);
    }



        }
      }




    }



    }





    if (false) {
      ushort d_outer = 64;
      

    if ((
        (C % 128 == 0) ||
        (c + 128 <= C)
      ) && (
        (64 % 8 == 0) ||
        (d_outer + 32 <= 64)
      )) {
      


    simdgroup_matrix_storage<float> Q_sram[32 / 8];





      

      uint2 Q_src_offset(
        morton_offset.x + d_outer,
        min(parallelization_group_offset + sidx * 8 + morton_offset.y, R - 1));
      auto Q_src = simdgroup_matrix_storage<float>
      ::apply_offset(
        Q, Q_leading_dimension,
        Q_src_offset, false);




      #pragma clang loop unroll(full)
      for (ushort d = 0; d < 32; d += 8) {
        ushort2 Q_origin(d, 0);
        Q_sram[d / 8].load(
          Q_src, Q_leading_dimension,
          Q_origin, false);
      }





      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        uint2 K_offset(d_outer, c);
        auto src = simdgroup_matrix_storage<half>
        ::apply_offset(
          K, K_leading_dimension,
          K_offset, false);
        auto dst = (threadgroup half*)(threadgroup_block);
 
        ushort D_src_dimension = min(
          ushort(32),
          ushort(64 - d_outer));
        ushort D_dst_dimension = 32;
        ushort C_src_dimension = min(
          uint(128),
          uint(C - c));
        ushort C_dst_dimension = max(
          ushort((((C % 128 == 0) ? 128 : C % 128) + 7) / 8 * 8),
          ushort(C_src_dimension));
        ushort2 tile_src(D_src_dimension, C_src_dimension);
        ushort2 tile_dst(D_dst_dimension, C_dst_dimension);

        simdgroup_event event;
        event.async_copy<32, 32>(
          dst, tile_dst,
          src, K_leading_dimension, tile_src, false);
        simdgroup_event::wait(1, &event);
      }

      

      ushort2 K_block_offset(
        morton_offset.x,
        morton_offset.y);
      auto K_src = (threadgroup half*)(threadgroup_block);
      K_src = simdgroup_matrix_storage<half>
      ::apply_offset(
        K_src, 32,
        K_block_offset, true);
      threadgroup_barrier(mem_flags::mem_threadgroup);








      #pragma clang loop unroll(full)
      for (ushort d = 0; d < 32; d += 8) {
        

    #pragma clang loop unroll(full)
    for (ushort c = 0; c < 128; c += 8) {
      
      // Load the RHS from memory.
      ushort2 K_origin(c, d);
      simdgroup_matrix_storage<half> K;
      K.load(
        K_src, 32,
        K_origin, true);

      // Issue one SIMD matmul instruction.
      S_sram[c / 8].multiply(
        Q_sram[(0 + d) / 8],
        K, true);
    }



      }




    } else {
      


    simdgroup_matrix_storage<float> Q_sram[32 / 8];





      

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sidx == 0) {
      uint2 Q_offset(d_outer, parallelization_group_offset);
      auto src = simdgroup_matrix_storage<float>
      ::apply_offset(
        Q, Q_leading_dimension,
        Q_offset, false);
      auto dst = (threadgroup float*)(threadgroup_block);

      ushort D_src_dimension = min(
        ushort(32),
        ushort(64 - d_outer));
      ushort D_dst_dimension = 32;
      ushort R_dimension = min(
        uint(32),
        uint(R - parallelization_group_offset));
      ushort2 tile_src(D_src_dimension, R_dimension);
      ushort2 tile_dst(D_dst_dimension, R_dimension);

      simdgroup_event event;
      event.async_copy<32, 32>(
        dst, tile_dst,
        src, Q_leading_dimension, tile_src, false);
      simdgroup_event::wait(1, &event);
    }



      

      ushort2 Q_block_offset(
        morton_offset.x, 
        morton_offset.y + sidx * 8);
      auto Q_src = (threadgroup float*)(threadgroup_block);
      Q_src = simdgroup_matrix_storage<float>
      ::apply_offset(
        Q_src, 32,
        Q_block_offset, false);
      threadgroup_barrier(mem_flags::mem_threadgroup);




      #pragma clang loop unroll(full)
      for (ushort d = 0; d < 32; d += 8) {
        ushort2 Q_origin(d, 0);
        Q_sram[d / 8].load(
          Q_src, 32,
          Q_origin, false);
      }





      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        uint2 K_offset(d_outer, c);
        auto src = simdgroup_matrix_storage<half>
        ::apply_offset(
          K, K_leading_dimension,
          K_offset, false);
        auto dst = (threadgroup half*)(threadgroup_block);
 
        ushort D_src_dimension = min(
          ushort(32),
          ushort(64 - d_outer));
        ushort D_dst_dimension = 32;
        ushort C_src_dimension = min(
          uint(128),
          uint(C - c));
        ushort C_dst_dimension = max(
          ushort((((C % 128 == 0) ? 128 : C % 128) + 7) / 8 * 8),
          ushort(C_src_dimension));
        ushort2 tile_src(D_src_dimension, C_src_dimension);
        ushort2 tile_dst(D_dst_dimension, C_dst_dimension);

        simdgroup_event event;
        event.async_copy<32, 32>(
          dst, tile_dst,
          src, K_leading_dimension, tile_src, false);
        simdgroup_event::wait(1, &event);
      }

      

      ushort2 K_block_offset(
        morton_offset.x,
        morton_offset.y);
      auto K_src = (threadgroup half*)(threadgroup_block);
      K_src = simdgroup_matrix_storage<half>
      ::apply_offset(
        K_src, 32,
        K_block_offset, true);
      threadgroup_barrier(mem_flags::mem_threadgroup);








      #pragma clang loop unroll(full)
      for (ushort d = 0; d < 32; d += 8) {
        

    #pragma clang loop unroll(full)
    for (ushort c = 0; c < (((C % 128 == 0) ? 128 : C % 128) + 7) / 8 * 8; c += 8) {
      
      // Load the RHS from memory.
      ushort2 K_origin(c, d);
      simdgroup_matrix_storage<half> K;
      K.load(
        K_src, 32,
        K_origin, true);

      // Issue one SIMD matmul instruction.
      S_sram[c / 8].multiply(
        Q_sram[(0 + d) / 8],
        K, true);
    }



        if (c + 128
            < C) {
          

    #pragma clang loop unroll(full)
    for (ushort c = (((C % 128 == 0) ? 128 : C % 128) + 7) / 8 * 8; c < 128; c += 8) {
      
      // Load the RHS from memory.
      ushort2 K_origin(c, d);
      simdgroup_matrix_storage<half> K;
      K.load(
        K_src, 32,
        K_origin, true);

      // Issue one SIMD matmul instruction.
      S_sram[c / 8].multiply(
        Q_sram[(0 + d) / 8],
        K, true);
    }



        }
      }




    }



    }




    

  if (((C % 128) != 0) &&
      (c + 128 > C)) {
    // Prevent the value from becoming -INF during the FMA before the
    // exponentiation. If the multiplication during FMA returns -INF,
    // subtracting a positive 'm' value will turn it into zero. We don't want
    // that. exp(0) evaluates to 1.00 and corrupts the value of 'l'.
    const float mask_value =
    (0.875 / 1.442695) * -numeric_limits<float>::max();
    
    #pragma clang loop unroll(full)
    for (ushort index = 0; index < 2; ++index) {
      if (morton_offset.x + index >= (C % 128) - ((C % 128) - ((C % 128) % 8))) {
        auto S_elements = S_sram[((C % 128) - ((C % 128) % 8)) / 8].thread_elements();
        (*S_elements)[index] = mask_value;
      }
    }
    #pragma clang loop unroll(full)
    for (ushort c = ((C % 128) - ((C % 128) % 8)) + 8; c < 128; c += 8) {
      auto S_elements = S_sram[c / 8].thread_elements();
      *S_elements = mask_value;
    }
  }




    // m = reduce(m)
    

  // update 'm'
  vec<float, 2> m_new_accumulator;
  #pragma clang loop unroll(full)
  for (ushort c = 0; c < 128; c += 8) {
    auto S_elements = S_sram[c / 8].thread_elements();
    if (c == 0) {
      m_new_accumulator = *S_elements;
    } else {
      m_new_accumulator = max(m_new_accumulator, *S_elements);
    }
  }
  float m_new = max(m_new_accumulator[0], m_new_accumulator[1]);
  m_new = max(m_new, simd_shuffle_xor(m_new, 1));
  m_new = max(m_new, simd_shuffle_xor(m_new, 8));
  m_new *= dot_product_scale;
  



    // correction = exp(m_old) / exp(m_new)
    

  // update 'O'
  float correction = 1;
  if (m_new > m) {
    correction = fast::exp2(m - m_new);
    m = m_new;
  }



    // P = softmax(S * scaleFactor)
    

    

      simdgroup_matrix_storage<float> P_sram[128 / 8];



    {
      

      #pragma clang loop unroll(full)
      for (ushort c = 0; c < 128; c += 8) {
        auto L_elements = m;
        

      auto S = *(S_sram[c / 8].thread_elements());
      auto P = vec<float, 2>(
        fast::exp2(float2(S) * dot_product_scale - float2(L_elements)));
      *(P_sram[c / 8].thread_elements()) = P;



      }



    }




    // l = reduce(l)
    
  
  // update 'l'
  float2 l_new_accumulator;
  #pragma clang loop unroll(full)
  for (ushort c = 0; c < 128; c += 8) {
    auto P_elements = P_sram[c / 8].thread_elements();
    if (c == 0) {
      l_new_accumulator = float2(*P_elements);
    } else {
      l_new_accumulator += float2(*P_elements);
    }
  }
  float l_new = l_new_accumulator[0] + l_new_accumulator[1];
  l_new += simd_shuffle_xor(l_new, 1);
  l_new += simd_shuffle_xor(l_new, 8);
  l = l * correction + l_new;
  



    // O *= correction
    // O += P * V
    // O /= l
    

    
    #pragma clang loop unroll(disable)
    for (
      ushort d_outer = 0;
      d_outer < 64;
      d_outer += 32
    ) {
      

    if ((
          (C % 128 == 0) ||
          (c + 128 <= C)
        ) && (
          (64 % 8 == 0) ||
          (d_outer + 32 <= 64)
        )) {
      

    

    simdgroup_matrix_storage<float> O_sram[32 / 8];



    if (c == 0) {
      
    
    #pragma clang loop unroll(full)
    for (ushort d = 0; d < 32; d += 8) {
      auto O = O_sram + (0 + d) / 8;
      *O = simdgroup_matrix_storage<float>(0);
    }



    } else {
      

      

       uint2 O_src_offset(
         morton_offset.x + d_outer,
         min(parallelization_group_offset + sidx * 8 + morton_offset.y, R - 1));
       auto O_src = simdgroup_matrix_storage<float>
       ::apply_offset(
         O, O_leading_dimension,
         O_src_offset, false);




      #pragma clang loop unroll(full)
      for (ushort d = 0; d < 32; d += 8) {
        ushort2 O_origin(d, 0);
        O_sram[d / 8].load(
          O_src, O_leading_dimension,
          O_origin, false);
      }



      
    
    #pragma clang loop unroll(full)
    for (ushort d = 0; d < 32; d += 8) {
      auto O = O_sram + (0 + d) / 8;
      *(O->thread_elements()) *= correction;
    }



    }
    
      
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        uint2 V_offset(d_outer, c);
        auto src = simdgroup_matrix_storage<half>
        ::apply_offset(
          V, V_leading_dimension,
          V_offset, false);
        auto dst = (threadgroup half*)(threadgroup_block);
        
        ushort D_dimension = min(
          ushort(32),
          ushort(64 - d_outer));
        ushort C_src_dimension = min(
          uint(128),
          uint(C - c));
        ushort C_dst_dimension = max(
          ushort((((C % 128 == 0) ? 128 : C % 128) + 7) / 8 * 8),
          ushort(C_src_dimension));
        ushort2 tile_src(D_dimension, C_src_dimension);
        ushort2 tile_dst(D_dimension, C_dst_dimension);
        
        simdgroup_event event;
        event.async_copy<32, 32>(
          dst, tile_dst,
          src, V_leading_dimension, tile_src, false);
        simdgroup_event::wait(1, &event);
      }

      

      ushort2 V_block_offset(
        morton_offset.x,
        morton_offset.y);
      auto V_src = (threadgroup half*)(threadgroup_block);
      V_src = simdgroup_matrix_storage<half>
      ::apply_offset(
        V_src, 32,
        V_block_offset, false);
      threadgroup_barrier(mem_flags::mem_threadgroup);



      
 

    

        

    #pragma clang loop unroll(full)
    for (ushort c = 0; c < 128; c += 8) {
      
      

    #pragma clang loop unroll(full)
    for (ushort d = 0; d < 32; d += 8) {
      // Load the RHS from memory.
      ushort2 V_origin(d, c);
      simdgroup_matrix_storage<half> V;
      V.load(
        V_src, 32,
        V_origin, false);

      // Issue one SIMD matmul instruction.
      O_sram[(0 + d) / 8].multiply(
        P_sram[c / 8], V, /*accumulate=*/true);
    }



    }



        if (
          (C % 128 == 0) &&
          (c + 128 == C)
        ) {
           
    
    #pragma clang loop unroll(full)
    for (ushort d = 0; d < 32; d += 8) {
      auto O = O_sram + (0 + d) / 8;
      *(O->thread_elements()) *= fast::divide(1, l);
    }



        }



    

      

       uint2 O_src_offset(
         morton_offset.x + d_outer,
         min(parallelization_group_offset + sidx * 8 + morton_offset.y, R - 1));
       auto O_src = simdgroup_matrix_storage<float>
       ::apply_offset(
         O, O_leading_dimension,
         O_src_offset, false);




      if (parallelization_group_offset + sidx * 8 + morton_offset.y < R) {
        #pragma clang loop unroll(full)
        for (ushort d = 0; d < 32; d += 8) {
          ushort2 O_origin(d, 0);
          O_sram[d / 8].store(
            O_src, O_leading_dimension,
            O_origin, false);
        }
      }






    } else {
      

    

    simdgroup_matrix_storage<float> O_sram[32 / 8];



    if (c == 0) {
      
    
    #pragma clang loop unroll(full)
    for (ushort d = 0; d < 32; d += 8) {
      auto O = O_sram + (0 + d) / 8;
      *O = simdgroup_matrix_storage<float>(0);
    }



    } else {
      

      

     threadgroup_barrier(mem_flags::mem_threadgroup);
     if (sidx == 0) {
       uint2 O_offset(d_outer, parallelization_group_offset);
       auto src = simdgroup_matrix_storage<float>
       ::apply_offset(
         O, O_leading_dimension,
         O_offset, false);
       auto dst = (threadgroup float*)(threadgroup_block);
       
       ushort D_dimension = min(
         ushort(32),
         ushort(64 - d_outer));
       ushort R_dimension = min(
         uint(32),
         uint(R - parallelization_group_offset));
       ushort2 tile(D_dimension, R_dimension);
       
       simdgroup_event event;
       event.async_copy<32, 32>(
         dst, tile,
         src, O_leading_dimension, tile, false);
       simdgroup_event::wait(1, &event);
     }



      

       ushort2 O_block_offset(
         morton_offset.x,
         morton_offset.y + sidx * 8);
       auto O_src = (threadgroup float*)(threadgroup_block);
       O_src = simdgroup_matrix_storage<float>
       ::apply_offset(
         O_src, 32,
         O_block_offset, false);
       threadgroup_barrier(mem_flags::mem_threadgroup);



      
      #pragma clang loop unroll(full)
      for (ushort d = 0; d < 32; d += 8) {
        ushort2 O_origin(d, 0);
        O_sram[d / 8].load(
          O_src, 32, 
          O_origin, false);
      }
      


      
    
    #pragma clang loop unroll(full)
    for (ushort d = 0; d < 32; d += 8) {
      auto O = O_sram + (0 + d) / 8;
      *(O->thread_elements()) *= correction;
    }



    }
    
      
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        uint2 V_offset(d_outer, c);
        auto src = simdgroup_matrix_storage<half>
        ::apply_offset(
          V, V_leading_dimension,
          V_offset, false);
        auto dst = (threadgroup half*)(threadgroup_block);
        
        ushort D_dimension = min(
          ushort(32),
          ushort(64 - d_outer));
        ushort C_src_dimension = min(
          uint(128),
          uint(C - c));
        ushort C_dst_dimension = max(
          ushort((((C % 128 == 0) ? 128 : C % 128) + 7) / 8 * 8),
          ushort(C_src_dimension));
        ushort2 tile_src(D_dimension, C_src_dimension);
        ushort2 tile_dst(D_dimension, C_dst_dimension);
        
        simdgroup_event event;
        event.async_copy<32, 32>(
          dst, tile_dst,
          src, V_leading_dimension, tile_src, false);
        simdgroup_event::wait(1, &event);
      }

      

      ushort2 V_block_offset(
        morton_offset.x,
        morton_offset.y);
      auto V_src = (threadgroup half*)(threadgroup_block);
      V_src = simdgroup_matrix_storage<half>
      ::apply_offset(
        V_src, 32,
        V_block_offset, false);
      threadgroup_barrier(mem_flags::mem_threadgroup);



      
 

    

        

    #pragma clang loop unroll(full)
    for (ushort c = 0; c < (((C % 128 == 0) ? 128 : C % 128) + 7) / 8 * 8; c += 8) {
      
      

    #pragma clang loop unroll(full)
    for (ushort d = 0; d < 32; d += 8) {
      // Load the RHS from memory.
      ushort2 V_origin(d, c);
      simdgroup_matrix_storage<half> V;
      V.load(
        V_src, 32,
        V_origin, false);

      // Issue one SIMD matmul instruction.
      O_sram[(0 + d) / 8].multiply(
        P_sram[c / 8], V, /*accumulate=*/true);
    }



    }



        if (c + 128
            < C) {
          

    #pragma clang loop unroll(full)
    for (ushort c = (((C % 128 == 0) ? 128 : C % 128) + 7) / 8 * 8; c < 128; c += 8) {
      
      

    #pragma clang loop unroll(full)
    for (ushort d = 0; d < 32; d += 8) {
      // Load the RHS from memory.
      ushort2 V_origin(d, c);
      simdgroup_matrix_storage<half> V;
      V.load(
        V_src, 32,
        V_origin, false);

      // Issue one SIMD matmul instruction.
      O_sram[(0 + d) / 8].multiply(
        P_sram[c / 8], V, /*accumulate=*/true);
    }



    }



        } else {
          
    
    #pragma clang loop unroll(full)
    for (ushort d = 0; d < 32; d += 8) {
      auto O = O_sram + (0 + d) / 8;
      *(O->thread_elements()) *= fast::divide(1, l);
    }



        }



    

      

       ushort2 O_block_offset(
         morton_offset.x,
         morton_offset.y + sidx * 8);
       auto O_src = (threadgroup float*)(threadgroup_block);
       O_src = simdgroup_matrix_storage<float>
       ::apply_offset(
         O_src, 32,
         O_block_offset, false);
       threadgroup_barrier(mem_flags::mem_threadgroup);




      #pragma clang loop unroll(full)
      for (ushort d = 0; d < 32; d += 8) {
        ushort2 O_origin(d, 0);
        O_sram[d / 8].store(
          O_src, 32,
          O_origin, false);
      }

      

     threadgroup_barrier(mem_flags::mem_threadgroup);
     if (sidx == 0) {
       uint2 O_offset(d_outer, parallelization_group_offset);
       auto src = (threadgroup float*)(threadgroup_block);
       auto dst = simdgroup_matrix_storage<float>
       ::apply_offset(
         O, O_leading_dimension,
         O_offset, false);
       
       ushort D_dimension = min(
         ushort(32),
         ushort(64 - d_outer));
       ushort R_dimension = min(
         uint(32),
         uint(R - parallelization_group_offset));
       ushort2 tile(D_dimension, R_dimension);
       
       simdgroup_event event;
       event.async_copy<32, 32>(
         dst, O_leading_dimension, tile,
         src, tile, false);
       simdgroup_event::wait(1, &event);
     }
     








    }



    }
    




    if (false) {
      ushort d_outer = 64;
      

    if ((
          (C % 128 == 0) ||
          (c + 128 <= C)
        ) && (
          (64 % 8 == 0) ||
          (d_outer + 32 <= 64)
        )) {
      

    

    simdgroup_matrix_storage<float> O_sram[32 / 8];



    if (c == 0) {
      
    
    #pragma clang loop unroll(full)
    for (ushort d = 0; d < 32; d += 8) {
      auto O = O_sram + (0 + d) / 8;
      *O = simdgroup_matrix_storage<float>(0);
    }



    } else {
      

      

       uint2 O_src_offset(
         morton_offset.x + d_outer,
         min(parallelization_group_offset + sidx * 8 + morton_offset.y, R - 1));
       auto O_src = simdgroup_matrix_storage<float>
       ::apply_offset(
         O, O_leading_dimension,
         O_src_offset, false);




      #pragma clang loop unroll(full)
      for (ushort d = 0; d < 32; d += 8) {
        ushort2 O_origin(d, 0);
        O_sram[d / 8].load(
          O_src, O_leading_dimension,
          O_origin, false);
      }



      
    
    #pragma clang loop unroll(full)
    for (ushort d = 0; d < 32; d += 8) {
      auto O = O_sram + (0 + d) / 8;
      *(O->thread_elements()) *= correction;
    }



    }
    
      
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        uint2 V_offset(d_outer, c);
        auto src = simdgroup_matrix_storage<half>
        ::apply_offset(
          V, V_leading_dimension,
          V_offset, false);
        auto dst = (threadgroup half*)(threadgroup_block);
        
        ushort D_dimension = min(
          ushort(32),
          ushort(64 - d_outer));
        ushort C_src_dimension = min(
          uint(128),
          uint(C - c));
        ushort C_dst_dimension = max(
          ushort((((C % 128 == 0) ? 128 : C % 128) + 7) / 8 * 8),
          ushort(C_src_dimension));
        ushort2 tile_src(D_dimension, C_src_dimension);
        ushort2 tile_dst(D_dimension, C_dst_dimension);
        
        simdgroup_event event;
        event.async_copy<32, 32>(
          dst, tile_dst,
          src, V_leading_dimension, tile_src, false);
        simdgroup_event::wait(1, &event);
      }

      

      ushort2 V_block_offset(
        morton_offset.x,
        morton_offset.y);
      auto V_src = (threadgroup half*)(threadgroup_block);
      V_src = simdgroup_matrix_storage<half>
      ::apply_offset(
        V_src, 32,
        V_block_offset, false);
      threadgroup_barrier(mem_flags::mem_threadgroup);



      
 

    

        

    #pragma clang loop unroll(full)
    for (ushort c = 0; c < 128; c += 8) {
      
      

    #pragma clang loop unroll(full)
    for (ushort d = 0; d < 32; d += 8) {
      // Load the RHS from memory.
      ushort2 V_origin(d, c);
      simdgroup_matrix_storage<half> V;
      V.load(
        V_src, 32,
        V_origin, false);

      // Issue one SIMD matmul instruction.
      O_sram[(0 + d) / 8].multiply(
        P_sram[c / 8], V, /*accumulate=*/true);
    }



    }



        if (
          (C % 128 == 0) &&
          (c + 128 == C)
        ) {
           
    
    #pragma clang loop unroll(full)
    for (ushort d = 0; d < 32; d += 8) {
      auto O = O_sram + (0 + d) / 8;
      *(O->thread_elements()) *= fast::divide(1, l);
    }



        }



    

      

       uint2 O_src_offset(
         morton_offset.x + d_outer,
         min(parallelization_group_offset + sidx * 8 + morton_offset.y, R - 1));
       auto O_src = simdgroup_matrix_storage<float>
       ::apply_offset(
         O, O_leading_dimension,
         O_src_offset, false);




      if (parallelization_group_offset + sidx * 8 + morton_offset.y < R) {
        #pragma clang loop unroll(full)
        for (ushort d = 0; d < 32; d += 8) {
          ushort2 O_origin(d, 0);
          O_sram[d / 8].store(
            O_src, O_leading_dimension,
            O_origin, false);
        }
      }






    } else {
      

    

    simdgroup_matrix_storage<float> O_sram[32 / 8];



    if (c == 0) {
      
    
    #pragma clang loop unroll(full)
    for (ushort d = 0; d < 32; d += 8) {
      auto O = O_sram + (0 + d) / 8;
      *O = simdgroup_matrix_storage<float>(0);
    }



    } else {
      

      

     threadgroup_barrier(mem_flags::mem_threadgroup);
     if (sidx == 0) {
       uint2 O_offset(d_outer, parallelization_group_offset);
       auto src = simdgroup_matrix_storage<float>
       ::apply_offset(
         O, O_leading_dimension,
         O_offset, false);
       auto dst = (threadgroup float*)(threadgroup_block);
       
       ushort D_dimension = min(
         ushort(32),
         ushort(64 - d_outer));
       ushort R_dimension = min(
         uint(32),
         uint(R - parallelization_group_offset));
       ushort2 tile(D_dimension, R_dimension);
       
       simdgroup_event event;
       event.async_copy<32, 32>(
         dst, tile,
         src, O_leading_dimension, tile, false);
       simdgroup_event::wait(1, &event);
     }



      

       ushort2 O_block_offset(
         morton_offset.x,
         morton_offset.y + sidx * 8);
       auto O_src = (threadgroup float*)(threadgroup_block);
       O_src = simdgroup_matrix_storage<float>
       ::apply_offset(
         O_src, 32,
         O_block_offset, false);
       threadgroup_barrier(mem_flags::mem_threadgroup);



      
      #pragma clang loop unroll(full)
      for (ushort d = 0; d < 32; d += 8) {
        ushort2 O_origin(d, 0);
        O_sram[d / 8].load(
          O_src, 32, 
          O_origin, false);
      }
      


      
    
    #pragma clang loop unroll(full)
    for (ushort d = 0; d < 32; d += 8) {
      auto O = O_sram + (0 + d) / 8;
      *(O->thread_elements()) *= correction;
    }



    }
    
      
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sidx == 0) {
        uint2 V_offset(d_outer, c);
        auto src = simdgroup_matrix_storage<half>
        ::apply_offset(
          V, V_leading_dimension,
          V_offset, false);
        auto dst = (threadgroup half*)(threadgroup_block);
        
        ushort D_dimension = min(
          ushort(32),
          ushort(64 - d_outer));
        ushort C_src_dimension = min(
          uint(128),
          uint(C - c));
        ushort C_dst_dimension = max(
          ushort((((C % 128 == 0) ? 128 : C % 128) + 7) / 8 * 8),
          ushort(C_src_dimension));
        ushort2 tile_src(D_dimension, C_src_dimension);
        ushort2 tile_dst(D_dimension, C_dst_dimension);
        
        simdgroup_event event;
        event.async_copy<32, 32>(
          dst, tile_dst,
          src, V_leading_dimension, tile_src, false);
        simdgroup_event::wait(1, &event);
      }

      

      ushort2 V_block_offset(
        morton_offset.x,
        morton_offset.y);
      auto V_src = (threadgroup half*)(threadgroup_block);
      V_src = simdgroup_matrix_storage<half>
      ::apply_offset(
        V_src, 32,
        V_block_offset, false);
      threadgroup_barrier(mem_flags::mem_threadgroup);



      
 

    

        

    #pragma clang loop unroll(full)
    for (ushort c = 0; c < (((C % 128 == 0) ? 128 : C % 128) + 7) / 8 * 8; c += 8) {
      
      

    #pragma clang loop unroll(full)
    for (ushort d = 0; d < 32; d += 8) {
      // Load the RHS from memory.
      ushort2 V_origin(d, c);
      simdgroup_matrix_storage<half> V;
      V.load(
        V_src, 32,
        V_origin, false);

      // Issue one SIMD matmul instruction.
      O_sram[(0 + d) / 8].multiply(
        P_sram[c / 8], V, /*accumulate=*/true);
    }



    }



        if (c + 128
            < C) {
          

    #pragma clang loop unroll(full)
    for (ushort c = (((C % 128 == 0) ? 128 : C % 128) + 7) / 8 * 8; c < 128; c += 8) {
      
      

    #pragma clang loop unroll(full)
    for (ushort d = 0; d < 32; d += 8) {
      // Load the RHS from memory.
      ushort2 V_origin(d, c);
      simdgroup_matrix_storage<half> V;
      V.load(
        V_src, 32,
        V_origin, false);

      // Issue one SIMD matmul instruction.
      O_sram[(0 + d) / 8].multiply(
        P_sram[c / 8], V, /*accumulate=*/true);
    }



    }



        } else {
          
    
    #pragma clang loop unroll(full)
    for (ushort d = 0; d < 32; d += 8) {
      auto O = O_sram + (0 + d) / 8;
      *(O->thread_elements()) *= fast::divide(1, l);
    }



        }



    

      

       ushort2 O_block_offset(
         morton_offset.x,
         morton_offset.y + sidx * 8);
       auto O_src = (threadgroup float*)(threadgroup_block);
       O_src = simdgroup_matrix_storage<float>
       ::apply_offset(
         O_src, 32,
         O_block_offset, false);
       threadgroup_barrier(mem_flags::mem_threadgroup);




      #pragma clang loop unroll(full)
      for (ushort d = 0; d < 32; d += 8) {
        ushort2 O_origin(d, 0);
        O_sram[d / 8].store(
          O_src, 32,
          O_origin, false);
      }

      

     threadgroup_barrier(mem_flags::mem_threadgroup);
     if (sidx == 0) {
       uint2 O_offset(d_outer, parallelization_group_offset);
       auto src = (threadgroup float*)(threadgroup_block);
       auto dst = simdgroup_matrix_storage<float>
       ::apply_offset(
         O, O_leading_dimension,
         O_offset, false);
       
       ushort D_dimension = min(
         ushort(32),
         ushort(64 - d_outer));
       ushort R_dimension = min(
         uint(32),
         uint(R - parallelization_group_offset));
       ushort2 tile(D_dimension, R_dimension);
       
       simdgroup_event event;
       event.async_copy<32, 32>(
         dst, O_leading_dimension, tile,
         src, tile, false);
       simdgroup_event::wait(1, &event);
     }
     








    }



    }




  }






    if (parallelization_group_offset + sidx * 8 + morton_offset.y < R) {
      // Premultiplied by log_base_2(e).
      float L_sram = m + fast::log2(l);
    }



}

