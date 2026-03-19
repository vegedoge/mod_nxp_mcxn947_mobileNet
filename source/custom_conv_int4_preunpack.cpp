/*
 * custom_conv_int4_preunpack.cpp
 *
 * Custom INT4 Conv2D operator with pre-unpack optimization.
 *
 * Strategy:
 *   - If layer's INT8 weight size <= PREUNPACK_SCRATCH_KB * 1024:
 *     Bulk unpack INT4→INT8 into static SRAM buffer, then call
 *     arm_convolve_wrapper_s8() which uses SMLAD for ~2x MAC throughput.
 *   - Otherwise:
 *     Fall back to arm_convolve_wrapper_s4() (standard CMSIS-NN INT4 path).
 *
 * Works with --target cmsis packed models (true INT4 scales, rescaled bias).
 */

#include "custom_conv_int4_preunpack.h"
#include "demo_config.h"

// TFLM
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/micro/micro_context.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"

// CMSIS-NN
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

// For profiling breakdown
#include "timer.h"
#include "fsl_debug_console.h"

#include <string.h>

namespace tflite {
namespace {

// Static scratch buffer for INT4→INT8 pre-unpack (lives in BSS/SRAM)
static int8_t s_preunpack_buf[PREUNPACK_SCRATCH_KB * 1024];

struct PreunpackConvOpData {
  // Per-channel quantization
  int32_t* per_channel_multiplier;
  int32_t* per_channel_shift;

  // Offsets
  int32_t input_offset;
  int32_t output_offset;

  // Activation clamp
  int32_t activation_min;
  int32_t activation_max;

  // Padding
  TfLitePaddingValues padding;

  // Scratch buffer indices for CMSIS-NN internal use
  int scratch_buf_index;     // for s8 wrapper
  int scratch_buf_index_s4;  // for s4 fallback

  // Whether this layer fits in the pre-unpack buffer
  bool use_s8_fast;

  // Whether this is a 1×1 convolution (eligible for fused SMLAD path)
  bool is_1x1;

  // Number of INT8 weight bytes (C_out * KH * KW * C_in)
  int weight_bytes_int8;
};

// Bulk unpack INT4 (packed 2-per-byte) → INT8
// Matches GET_INT4_WEIGHT in custom_int4_unpack.h:
//   even index = low nibble, odd index = high nibble

static void unpack_int4_to_int8(const int8_t* packed, int8_t* unpacked,
                                int num_elements) {
  const uint8_t* src = reinterpret_cast<const uint8_t*>(packed);
  const int num_bytes = num_elements / 2;
  for (int i = 0; i < num_bytes; i++) {
    const uint8_t byte = src[i];
    // Low nibble (even index) — sign extend
    int8_t lo = static_cast<int8_t>((byte & 0x0F));
    if (lo & 0x08) lo |= 0xF0;
    // High nibble (odd index) — sign extend
    int8_t hi = static_cast<int8_t>((byte >> 4) & 0x0F);
    if (hi & 0x08) hi |= 0xF0;
    unpacked[2 * i]     = lo;
    unpacked[2 * i + 1] = hi;
  }
  // Handle odd element count (shouldn't happen for conv weights, but be safe)
  if (num_elements & 1) {
    const uint8_t byte = src[num_bytes];
    int8_t lo = static_cast<int8_t>((byte & 0x0F));
    if (lo & 0x08) lo |= 0xF0;
    unpacked[num_elements - 1] = lo;
  }
}

// ── Fused INT4 1×1 Conv2D (optimized, 4-pixel batch) ──
// Reads INT4 weights directly from Flash, unpacks in registers, uses SMLAD.
// No SRAM scratch buffer needed — works for arbitrarily large layers.
// Only valid for 1×1 convolutions (no padding, no im2col).
//
// Optimizations:
//   - SXTAB16 for fused sign-extend + offset addition (1 insn vs 4 ADDs)
//   - SBFX for signed 4-bit extraction (1 insn vs shift+mask+branch)
//   - 4-pixel batching: unpack weights once, reuse across 4 spatial positions
//     Theory: (7 shared + 5×4 per-px) = 27 insns / 16 MACs = 1.69 cyc/MAC
//
// Inner loop macro: load 4 activations, apply offset, MAC with pre-packed weights.
#define FUSED_MAC_ONE_PIXEL(a_ptr, j, offset, wp02, wp13, acc)  \
  do {                                                           \
    uint32_t _raw = *reinterpret_cast<const uint32_t*>((a_ptr) + (j)); \
    uint32_t _lo, _hi;                                           \
    __ASM volatile("sxtab16 %0, %1, %2"                          \
                    : "=r"(_lo) : "r"(offset), "r"(_raw));       \
    __ASM volatile("sxtab16 %0, %1, %2, ror #8"                  \
                    : "=r"(_hi) : "r"(offset), "r"(_raw));       \
    __ASM volatile("smlad %0, %1, %2, %0"                        \
                    : "+r"(acc) : "r"(wp02), "r"(_lo));           \
    __ASM volatile("smlad %0, %1, %2, %0"                        \
                    : "+r"(acc) : "r"(wp13), "r"(_hi));           \
  } while (0)

// Unpack 4 INT4 weights from lower/upper 16 bits of a 32-bit word,
// pack into SMLAD-ready format.
#define UNPACK4_AND_PACK(bits, wp02, wp13)                       \
  do {                                                           \
    int32_t _w0, _w1, _w2, _w3;                                  \
    __ASM volatile("sbfx %0, %1, #0, #4"  : "=r"(_w0) : "r"(bits)); \
    __ASM volatile("sbfx %0, %1, #4, #4"  : "=r"(_w1) : "r"(bits)); \
    __ASM volatile("sbfx %0, %1, #8, #4"  : "=r"(_w2) : "r"(bits)); \
    __ASM volatile("sbfx %0, %1, #12, #4" : "=r"(_w3) : "r"(bits)); \
    __ASM volatile("pkhbt %0, %1, %2, lsl #16"                   \
                    : "=r"(wp02) : "r"(_w0), "r"(_w2));          \
    __ASM volatile("pkhbt %0, %1, %2, lsl #16"                   \
                    : "=r"(wp13) : "r"(_w1), "r"(_w3));          \
  } while (0)

// Requantize + clamp + store macro
#define REQUANT_STORE(acc, oc, px, multiplier, shift, output_offset, \
                      act_min, act_max, output, c_out)               \
  do {                                                                \
    int32_t _a = arm_nn_requantize((acc), (multiplier)[(oc)],         \
                                   (shift)[(oc)]);                    \
    _a += (output_offset);                                            \
    _a = MAX(_a, (act_min));                                          \
    _a = MIN(_a, (act_max));                                          \
    (output)[(px) * (c_out) + (oc)] = static_cast<int8_t>(_a);       \
  } while (0)


static void conv_1x1_fused_s4(
    const int8_t* input,        // [H*W, C_in] INT8
    const int8_t* packed_w,     // [C_out, C_in/2] INT4 packed
    const int32_t* bias,        // [C_out]
    int8_t* output,             // [H*W, C_out] INT8
    int num_pixels,             // H * W
    int c_in,                   // input channels (must be multiple of 4)
    int c_out,                  // output channels
    int32_t input_offset,       // -input_zero_point
    const int32_t* multiplier,  // per-channel requant multiplier
    const int32_t* shift,       // per-channel requant shift
    int32_t output_offset,      // output zero point
    int32_t act_min,
    int32_t act_max) {
  const int8_t* w_base = packed_w;
  const int c_in_packed = c_in / 2;

  // Pack input_offset as two int16 halves for SXTAB16
  const int16_t i16_offset = static_cast<int16_t>(input_offset);
  uint32_t offset_i16x2;
  __ASM volatile("pkhbt %0, %1, %1, lsl #16"
                  : "=r"(offset_i16x2) : "r"((int32_t)i16_offset));

  // ── Main loop: 4 pixels at a time ──
  // Weight unpack shared across 4 pixels: 7 insns (LDRH + SBFX×4 + PKHBT×2)
  // Per pixel: LDR + SXTAB16×2 + SMLAD×2 = 5 insns
  // Total: 7 + 5×4 = 27 insns / 16 MACs = 1.69 cyc/MAC
  // Note: 8-weight unrolling tested but showed no improvement (LDR=LDRH=1 cycle,
  //        extra >>16 shift + larger loop body offset any gains).
  int px = 0;
  for (; px + 3 < num_pixels; px += 4) {
    const int8_t* a0 = input + px * c_in;
    const int8_t* a1 = a0 + c_in;
    const int8_t* a2 = a1 + c_in;
    const int8_t* a3 = a2 + c_in;

    for (int oc = 0; oc < c_out; oc++) {
      const int32_t b = bias ? bias[oc] : 0;
      int32_t acc0 = b, acc1 = b, acc2 = b, acc3 = b;
      const int8_t* w_ptr = w_base + oc * c_in_packed;

      for (int j = 0; j < c_in; j += 4) {
        // ── Unpack 4 INT4 weights (shared across 4 pixels) ──
        uint32_t in16;
        __ASM volatile("ldrh %0, [%1]" : "=r"(in16) : "r"(w_ptr));
        w_ptr += 2;

        uint32_t wp02, wp13;
        UNPACK4_AND_PACK(in16, wp02, wp13);

        // ── MAC all 4 pixels (reuse wp02, wp13) ──
        FUSED_MAC_ONE_PIXEL(a0, j, offset_i16x2, wp02, wp13, acc0);
        FUSED_MAC_ONE_PIXEL(a1, j, offset_i16x2, wp02, wp13, acc1);
        FUSED_MAC_ONE_PIXEL(a2, j, offset_i16x2, wp02, wp13, acc2);
        FUSED_MAC_ONE_PIXEL(a3, j, offset_i16x2, wp02, wp13, acc3);
      }

      REQUANT_STORE(acc0, oc, px,   multiplier, shift, output_offset, act_min, act_max, output, c_out);
      REQUANT_STORE(acc1, oc, px+1, multiplier, shift, output_offset, act_min, act_max, output, c_out);
      REQUANT_STORE(acc2, oc, px+2, multiplier, shift, output_offset, act_min, act_max, output, c_out);
      REQUANT_STORE(acc3, oc, px+3, multiplier, shift, output_offset, act_min, act_max, output, c_out);
    }
  }

  // ── Tail: remaining 1-3 pixels ──
  for (; px < num_pixels; px++) {
    const int8_t* a_row = input + px * c_in;
    for (int oc = 0; oc < c_out; oc++) {
      int32_t acc = bias ? bias[oc] : 0;
      const int8_t* w_ptr = w_base + oc * c_in_packed;

      for (int j = 0; j < c_in; j += 4) {
        uint32_t in16;
        __ASM volatile("ldrh %0, [%1]" : "=r"(in16) : "r"(w_ptr));
        w_ptr += 2;

        uint32_t wp02, wp13;
        UNPACK4_AND_PACK(in16, wp02, wp13);
        FUSED_MAC_ONE_PIXEL(a_row, j, offset_i16x2, wp02, wp13, acc);
      }

      REQUANT_STORE(acc, oc, px, multiplier, shift, output_offset, act_min, act_max, output, c_out);
    }
  }
}

#if FUSED_BATCH_MODE == 1
// ── Fused INT4 1×1 Conv2D (2OC × 2pixel batch) ──
// Alternative batching strategy: process 2 output channels × 2 pixels simultaneously.
// Shares activation loading across 2 OCs (saves 3 insns per shared pixel).
// Theory: (7×2 weight_unpack + 3×2 act_load + 2×4 SMLAD) = 28 insns / 16 MACs = 1.75 cyc/MAC
// Compared to 4px batch: 27 insns / 16 MACs = 1.69 cyc/MAC (slightly worse).
// Purpose: ablation study to validate that pixel-batching > OC-batching for INT4 on Cortex-M33.

static void conv_1x1_fused_s4_2oc2px(
    const int8_t* input,
    const int8_t* packed_w,
    const int32_t* bias,
    int8_t* output,
    int num_pixels, int c_in, int c_out,
    int32_t input_offset,
    const int32_t* multiplier, const int32_t* shift,
    int32_t output_offset, int32_t act_min, int32_t act_max) {
  const int c_in_packed = c_in / 2;

  const int16_t i16_offset = static_cast<int16_t>(input_offset);
  uint32_t offset_i16x2;
  __ASM volatile("pkhbt %0, %1, %1, lsl #16"
                  : "=r"(offset_i16x2) : "r"((int32_t)i16_offset));

  // ── Main loop: 2 pixels × 2 output channels ──
  int px = 0;
  for (; px + 1 < num_pixels; px += 2) {
    const int8_t* a0 = input + px * c_in;
    const int8_t* a1 = a0 + c_in;

    int oc = 0;
    for (; oc + 1 < c_out; oc += 2) {
      const int32_t b0 = bias ? bias[oc] : 0;
      const int32_t b1 = bias ? bias[oc + 1] : 0;
      int32_t acc_p0o0 = b0;  // pixel 0, oc 0
      int32_t acc_p0o1 = b1;  // pixel 0, oc 1
      int32_t acc_p1o0 = b0;  // pixel 1, oc 0
      int32_t acc_p1o1 = b1;  // pixel 1, oc 1

      const int8_t* w_ptr_oc0 = packed_w + oc * c_in_packed;
      const int8_t* w_ptr_oc1 = packed_w + (oc + 1) * c_in_packed;

      for (int j = 0; j < c_in; j += 4) {
        // ── Unpack weights for OC 0 ──
        uint32_t in16_oc0;
        __ASM volatile("ldrh %0, [%1]" : "=r"(in16_oc0) : "r"(w_ptr_oc0));
        w_ptr_oc0 += 2;
        uint32_t wp02_oc0, wp13_oc0;
        UNPACK4_AND_PACK(in16_oc0, wp02_oc0, wp13_oc0);

        // ── Unpack weights for OC 1 ──
        uint32_t in16_oc1;
        __ASM volatile("ldrh %0, [%1]" : "=r"(in16_oc1) : "r"(w_ptr_oc1));
        w_ptr_oc1 += 2;
        uint32_t wp02_oc1, wp13_oc1;
        UNPACK4_AND_PACK(in16_oc1, wp02_oc1, wp13_oc1);

        // ── Pixel 0: load activation once, MAC with both OCs ──
        uint32_t raw0 = *reinterpret_cast<const uint32_t*>(a0 + j);
        uint32_t lo0, hi0;
        __ASM volatile("sxtab16 %0, %1, %2"
                        : "=r"(lo0) : "r"(offset_i16x2), "r"(raw0));
        __ASM volatile("sxtab16 %0, %1, %2, ror #8"
                        : "=r"(hi0) : "r"(offset_i16x2), "r"(raw0));

        __ASM volatile("smlad %0, %1, %2, %0"
                        : "+r"(acc_p0o0) : "r"(wp02_oc0), "r"(lo0));
        __ASM volatile("smlad %0, %1, %2, %0"
                        : "+r"(acc_p0o0) : "r"(wp13_oc0), "r"(hi0));
        __ASM volatile("smlad %0, %1, %2, %0"
                        : "+r"(acc_p0o1) : "r"(wp02_oc1), "r"(lo0));
        __ASM volatile("smlad %0, %1, %2, %0"
                        : "+r"(acc_p0o1) : "r"(wp13_oc1), "r"(hi0));

        // ── Pixel 1: load activation once, MAC with both OCs ──
        uint32_t raw1 = *reinterpret_cast<const uint32_t*>(a1 + j);
        uint32_t lo1, hi1;
        __ASM volatile("sxtab16 %0, %1, %2"
                        : "=r"(lo1) : "r"(offset_i16x2), "r"(raw1));
        __ASM volatile("sxtab16 %0, %1, %2, ror #8"
                        : "=r"(hi1) : "r"(offset_i16x2), "r"(raw1));

        __ASM volatile("smlad %0, %1, %2, %0"
                        : "+r"(acc_p1o0) : "r"(wp02_oc0), "r"(lo1));
        __ASM volatile("smlad %0, %1, %2, %0"
                        : "+r"(acc_p1o0) : "r"(wp13_oc0), "r"(hi1));
        __ASM volatile("smlad %0, %1, %2, %0"
                        : "+r"(acc_p1o1) : "r"(wp02_oc1), "r"(lo1));
        __ASM volatile("smlad %0, %1, %2, %0"
                        : "+r"(acc_p1o1) : "r"(wp13_oc1), "r"(hi1));
      }

      // Requantize all 4 outputs
      acc_p0o0 = arm_nn_requantize(acc_p0o0, multiplier[oc], shift[oc]);
      acc_p0o0 += output_offset;
      output[px * c_out + oc] = static_cast<int8_t>(
          MAX(MIN(acc_p0o0, act_max), act_min));

      acc_p0o1 = arm_nn_requantize(acc_p0o1, multiplier[oc+1], shift[oc+1]);
      acc_p0o1 += output_offset;
      output[px * c_out + oc + 1] = static_cast<int8_t>(
          MAX(MIN(acc_p0o1, act_max), act_min));

      acc_p1o0 = arm_nn_requantize(acc_p1o0, multiplier[oc], shift[oc]);
      acc_p1o0 += output_offset;
      output[(px+1) * c_out + oc] = static_cast<int8_t>(
          MAX(MIN(acc_p1o0, act_max), act_min));

      acc_p1o1 = arm_nn_requantize(acc_p1o1, multiplier[oc+1], shift[oc+1]);
      acc_p1o1 += output_offset;
      output[(px+1) * c_out + oc + 1] = static_cast<int8_t>(
          MAX(MIN(acc_p1o1, act_max), act_min));
    }

    // Handle odd output channel count
    if (oc < c_out) {
      int32_t acc0 = bias ? bias[oc] : 0;
      int32_t acc1 = acc0;
      const int8_t* w_ptr = packed_w + oc * c_in_packed;
      for (int j = 0; j < c_in; j += 4) {
        uint32_t in16;
        __ASM volatile("ldrh %0, [%1]" : "=r"(in16) : "r"(w_ptr));
        w_ptr += 2;
        uint32_t wp02, wp13;
        UNPACK4_AND_PACK(in16, wp02, wp13);
        FUSED_MAC_ONE_PIXEL(a0, j, offset_i16x2, wp02, wp13, acc0);
        FUSED_MAC_ONE_PIXEL(a1, j, offset_i16x2, wp02, wp13, acc1);
      }
      acc0 = arm_nn_requantize(acc0, multiplier[oc], shift[oc]);
      acc0 += output_offset;
      output[px * c_out + oc] = static_cast<int8_t>(MAX(MIN(acc0, act_max), act_min));
      acc1 = arm_nn_requantize(acc1, multiplier[oc], shift[oc]);
      acc1 += output_offset;
      output[(px+1) * c_out + oc] = static_cast<int8_t>(MAX(MIN(acc1, act_max), act_min));
    }
  }

  // Handle last pixel if odd
  if (px < num_pixels) {
    const int8_t* a_row = input + px * c_in;
    for (int oc = 0; oc < c_out; oc++) {
      int32_t acc = bias ? bias[oc] : 0;
      const int8_t* w_ptr = packed_w + oc * c_in_packed;
      for (int j = 0; j < c_in; j += 4) {
        uint32_t in16;
        __ASM volatile("ldrh %0, [%1]" : "=r"(in16) : "r"(w_ptr));
        w_ptr += 2;
        uint32_t wp02, wp13;
        UNPACK4_AND_PACK(in16, wp02, wp13);
        FUSED_MAC_ONE_PIXEL(a_row, j, offset_i16x2, wp02, wp13, acc);
      }
      acc = arm_nn_requantize(acc, multiplier[oc], shift[oc]);
      acc += output_offset;
      output[px * c_out + oc] = static_cast<int8_t>(MAX(MIN(acc, act_max), act_min));
    }
  }
}
#elif FUSED_BATCH_MODE == 2
// ── Fused INT4 1×1 Conv2D (2OC × 4pixel batch) ──
// Maximum batching: 2 output channels × 4 pixels = 8 accumulators.
// Exceeds register file (14 GPRs) — compiler must spill to stack.
// Theory without spill: (7×2 + 3×4 + 2×8) = 42 insns / 32 MACs = 1.31 cyc/MAC
// Expected: spill overhead will negate the theoretical advantage.
// Purpose: ablation study to show register pressure limits on Cortex-M33.

static void conv_1x1_fused_s4_2oc4px(
    const int8_t* input,
    const int8_t* packed_w,
    const int32_t* bias,
    int8_t* output,
    int num_pixels, int c_in, int c_out,
    int32_t input_offset,
    const int32_t* multiplier, const int32_t* shift,
    int32_t output_offset, int32_t act_min, int32_t act_max) {
  const int c_in_packed = c_in / 2;

  const int16_t i16_offset = static_cast<int16_t>(input_offset);
  uint32_t offset_i16x2;
  __ASM volatile("pkhbt %0, %1, %1, lsl #16"
                  : "=r"(offset_i16x2) : "r"((int32_t)i16_offset));

  int px = 0;
  for (; px + 3 < num_pixels; px += 4) {
    const int8_t* a0 = input + px * c_in;
    const int8_t* a1 = a0 + c_in;
    const int8_t* a2 = a1 + c_in;
    const int8_t* a3 = a2 + c_in;

    int oc = 0;
    for (; oc + 1 < c_out; oc += 2) {
      const int32_t b0 = bias ? bias[oc] : 0;
      const int32_t b1 = bias ? bias[oc + 1] : 0;
      // 8 accumulators — will spill to stack
      int32_t acc[8] = {b0, b1, b0, b1, b0, b1, b0, b1};
      // acc[0]=px0_oc0, acc[1]=px0_oc1, acc[2]=px1_oc0, acc[3]=px1_oc1
      // acc[4]=px2_oc0, acc[5]=px2_oc1, acc[6]=px3_oc0, acc[7]=px3_oc1

      const int8_t* w_ptr_oc0 = packed_w + oc * c_in_packed;
      const int8_t* w_ptr_oc1 = packed_w + (oc + 1) * c_in_packed;

      for (int j = 0; j < c_in; j += 4) {
        // Unpack both OCs
        uint32_t in16_0, in16_1;
        __ASM volatile("ldrh %0, [%1]" : "=r"(in16_0) : "r"(w_ptr_oc0));
        w_ptr_oc0 += 2;
        __ASM volatile("ldrh %0, [%1]" : "=r"(in16_1) : "r"(w_ptr_oc1));
        w_ptr_oc1 += 2;

        uint32_t wp02_0, wp13_0, wp02_1, wp13_1;
        UNPACK4_AND_PACK(in16_0, wp02_0, wp13_0);
        UNPACK4_AND_PACK(in16_1, wp02_1, wp13_1);

        // Pixel 0
        FUSED_MAC_ONE_PIXEL(a0, j, offset_i16x2, wp02_0, wp13_0, acc[0]);
        FUSED_MAC_ONE_PIXEL(a0, j, offset_i16x2, wp02_1, wp13_1, acc[1]);
        // Pixel 1
        FUSED_MAC_ONE_PIXEL(a1, j, offset_i16x2, wp02_0, wp13_0, acc[2]);
        FUSED_MAC_ONE_PIXEL(a1, j, offset_i16x2, wp02_1, wp13_1, acc[3]);
        // Pixel 2
        FUSED_MAC_ONE_PIXEL(a2, j, offset_i16x2, wp02_0, wp13_0, acc[4]);
        FUSED_MAC_ONE_PIXEL(a2, j, offset_i16x2, wp02_1, wp13_1, acc[5]);
        // Pixel 3
        FUSED_MAC_ONE_PIXEL(a3, j, offset_i16x2, wp02_0, wp13_0, acc[6]);
        FUSED_MAC_ONE_PIXEL(a3, j, offset_i16x2, wp02_1, wp13_1, acc[7]);
      }

      // Requantize all 8 outputs
      for (int p = 0; p < 4; p++) {
        for (int o = 0; o < 2; o++) {
          int32_t a = acc[p * 2 + o];
          a = arm_nn_requantize(a, multiplier[oc + o], shift[oc + o]);
          a += output_offset;
          a = MAX(a, act_min);
          a = MIN(a, act_max);
          output[(px + p) * c_out + oc + o] = static_cast<int8_t>(a);
        }
      }
    }

    // Odd OC tail
    if (oc < c_out) {
      int32_t a0v = bias ? bias[oc] : 0, a1v = a0v, a2v = a0v, a3v = a0v;
      const int8_t* w_ptr = packed_w + oc * c_in_packed;
      for (int j = 0; j < c_in; j += 4) {
        uint32_t in16;
        __ASM volatile("ldrh %0, [%1]" : "=r"(in16) : "r"(w_ptr));
        w_ptr += 2;
        uint32_t wp02, wp13;
        UNPACK4_AND_PACK(in16, wp02, wp13);
        FUSED_MAC_ONE_PIXEL(a0, j, offset_i16x2, wp02, wp13, a0v);
        FUSED_MAC_ONE_PIXEL(a1, j, offset_i16x2, wp02, wp13, a1v);
        FUSED_MAC_ONE_PIXEL(a2, j, offset_i16x2, wp02, wp13, a2v);
        FUSED_MAC_ONE_PIXEL(a3, j, offset_i16x2, wp02, wp13, a3v);
      }
      int32_t* accs[] = {&a0v, &a1v, &a2v, &a3v};
      for (int p = 0; p < 4; p++) {
        int32_t v = arm_nn_requantize(*accs[p], multiplier[oc], shift[oc]);
        v += output_offset;
        output[(px + p) * c_out + oc] = static_cast<int8_t>(MAX(MIN(v, act_max), act_min));
      }
    }
  }

  // Remaining 1-3 pixels: fall back to single-pixel
  for (; px < num_pixels; px++) {
    const int8_t* a_row = input + px * c_in;
    for (int oc = 0; oc < c_out; oc++) {
      int32_t acc = bias ? bias[oc] : 0;
      const int8_t* w_ptr = packed_w + oc * c_in_packed;
      for (int j = 0; j < c_in; j += 4) {
        uint32_t in16;
        __ASM volatile("ldrh %0, [%1]" : "=r"(in16) : "r"(w_ptr));
        w_ptr += 2;
        uint32_t wp02, wp13;
        UNPACK4_AND_PACK(in16, wp02, wp13);
        FUSED_MAC_ONE_PIXEL(a_row, j, offset_i16x2, wp02, wp13, acc);
      }
      acc = arm_nn_requantize(acc, multiplier[oc], shift[oc]);
      acc += output_offset;
      output[px * c_out + oc] = static_cast<int8_t>(MAX(MIN(acc, act_max), act_min));
    }
  }
}
#endif  // FUSED_BATCH_MODE

#undef FUSED_MAC_ONE_PIXEL
#undef UNPACK4_AND_PACK
#undef REQUANT_STORE

// ── Fused INT4 general Conv2D (for 3×3 etc.) ──
// Direct convolution: no im2col, no SRAM buffer.
// Iterates over kernel spatial positions, reads INT4 weights from Flash.

__attribute__((optimize("O2")))
static void conv_general_fused_s4(
    const int8_t* input,        // [H_in, W_in, C_in]
    const int8_t* packed_w,     // [C_out, KH, KW, C_in / 2] INT4 packed
    const int32_t* bias,
    int8_t* output,             // [H_out, W_out, C_out]
    int input_h, int input_w, int input_ch,
    int output_h, int output_w, int output_ch,
    int filter_h, int filter_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int32_t input_offset,
    const int32_t* multiplier,
    const int32_t* shift,
    int32_t output_offset,
    int32_t act_min, int32_t act_max) {
  const int filter_ch_packed = filter_h * filter_w * input_ch / 2;

  // Pack input_offset for SXTAB16
  const int16_t i16_offset = static_cast<int16_t>(input_offset);
  uint32_t offset_i16x2;
  __ASM volatile("pkhbt %0, %1, %1, lsl #16"
                  : "=r"(offset_i16x2) : "r"((int32_t)i16_offset));

  for (int out_y = 0; out_y < output_h; out_y++) {
    for (int out_x = 0; out_x < output_w; out_x++) {
      const int in_y_origin = out_y * stride_h - pad_h;
      const int in_x_origin = out_x * stride_w - pad_w;

      for (int oc = 0; oc < output_ch; oc++) {
        int32_t acc = bias ? bias[oc] : 0;
        const uint8_t* w_ptr =
            reinterpret_cast<const uint8_t*>(packed_w) + oc * filter_ch_packed;

        for (int fy = 0; fy < filter_h; fy++) {
          const int in_y = in_y_origin + fy;
          if (in_y < 0 || in_y >= input_h) {
            // Skip entire filter row: padded value = zero_point,
            // (zero_point + input_offset) = 0, so contribution is 0.
            w_ptr += filter_w * (input_ch / 2);
            continue;
          }

          for (int fx = 0; fx < filter_w; fx++) {
            const int in_x = in_x_origin + fx;
            if (in_x < 0 || in_x >= input_w) {
              w_ptr += input_ch / 2;
              continue;
            }

            const int8_t* a_ptr =
                input + (in_y * input_w + in_x) * input_ch;

            // Inner MAC loop: same SBFX + SXTAB16 + SMLAD as 1×1
            for (int ic = 0; ic < input_ch; ic += 4) {
              uint32_t in16;
              __ASM volatile("ldrh %0, [%1]" : "=r"(in16) : "r"(w_ptr));
              w_ptr += 2;

              int32_t w0, w1, w2, w3;
              __ASM volatile("sbfx %0, %1, #0, #4"  : "=r"(w0) : "r"(in16));
              __ASM volatile("sbfx %0, %1, #4, #4"  : "=r"(w1) : "r"(in16));
              __ASM volatile("sbfx %0, %1, #8, #4"  : "=r"(w2) : "r"(in16));
              __ASM volatile("sbfx %0, %1, #12, #4" : "=r"(w3) : "r"(in16));

              uint32_t wp02, wp13;
              __ASM volatile("pkhbt %0, %1, %2, lsl #16"
                              : "=r"(wp02) : "r"(w0), "r"(w2));
              __ASM volatile("pkhbt %0, %1, %2, lsl #16"
                              : "=r"(wp13) : "r"(w1), "r"(w3));

              uint32_t lhs_raw =
                  *reinterpret_cast<const uint32_t*>(a_ptr + ic);
              uint32_t lhs_low, lhs_high;
              __ASM volatile("sxtab16 %0, %1, %2"
                              : "=r"(lhs_low)
                              : "r"(offset_i16x2), "r"(lhs_raw));
              __ASM volatile("sxtab16 %0, %1, %2, ror #8"
                              : "=r"(lhs_high)
                              : "r"(offset_i16x2), "r"(lhs_raw));

              __ASM volatile("smlad %0, %1, %2, %0"
                              : "+r"(acc) : "r"(wp02), "r"(lhs_low));
              __ASM volatile("smlad %0, %1, %2, %0"
                              : "+r"(acc) : "r"(wp13), "r"(lhs_high));
            }
          }
        }

        acc = arm_nn_requantize(acc, multiplier[oc], shift[oc]);
        acc += output_offset;
        acc = MAX(acc, act_min);
        acc = MIN(acc, act_max);
        output[(out_y * output_w + out_x) * output_ch + oc] =
            static_cast<int8_t>(acc);
      }
    }
  }
}

void* PreunpackConvInit(TfLiteContext* context, const char* buffer,
                        size_t length) {
  return context->AllocatePersistentBuffer(context,
                                           sizeof(PreunpackConvOpData));
}

TfLiteStatus PreunpackConvPrepare(TfLiteContext* context, TfLiteNode* node) {
  auto* data = static_cast<PreunpackConvOpData*>(node->user_data);
  if (data == nullptr) return kTfLiteError;

  TF_LITE_ENSURE(context, node->inputs->size >= 2);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input =
      micro_context->AllocateTempTfLiteTensor(node->inputs->data[0]);
  TfLiteTensor* filter =
      micro_context->AllocateTempTfLiteTensor(node->inputs->data[1]);
  TfLiteTensor* bias =
      (node->inputs->size == 3)
          ? micro_context->AllocateTempTfLiteTensor(node->inputs->data[2])
          : nullptr;
  TfLiteTensor* output =
      micro_context->AllocateTempTfLiteTensor(node->outputs->data[0]);

  if (input == nullptr || filter == nullptr || output == nullptr) {
    return kTfLiteError;
  }

  const auto* params =
      reinterpret_cast<const TfLiteConvParams*>(node->builtin_data);

  // Dimensions
  const int input_h = input->dims->data[1];
  const int input_w = input->dims->data[2];
  const int input_ch = input->dims->data[3];
  const int filter_h = filter->dims->data[1];
  const int filter_w = filter->dims->data[2];
  const int output_ch = filter->dims->data[0];
  int output_h = output->dims->data[1];
  int output_w = output->dims->data[2];

  // Padding
  data->padding = ComputePaddingHeightWidth(
      params->stride_height, params->stride_width,
      params->dilation_height_factor, params->dilation_width_factor,
      input_h, input_w, filter_h, filter_w,
      params->padding, &output_h, &output_w);

  // Offsets
  data->input_offset = -input->params.zero_point;
  data->output_offset = output->params.zero_point;

  // Activation range
  CalculateActivationRangeQuantized(context, params->activation, output,
                                    &data->activation_min,
                                    &data->activation_max);

  // Per-channel quantization: compute multipliers/shifts from INT4 scales
  // (model has true INT4 scales from pack_int4.py --target cmsis)
  if (data->per_channel_multiplier == nullptr) {
    data->per_channel_multiplier = static_cast<int32_t*>(
        context->AllocatePersistentBuffer(context,
                                          output_ch * sizeof(int32_t)));
  }
  if (data->per_channel_shift == nullptr) {
    data->per_channel_shift = static_cast<int32_t*>(
        context->AllocatePersistentBuffer(context,
                                          output_ch * sizeof(int32_t)));
  }

  const auto* affine =
      reinterpret_cast<TfLiteAffineQuantization*>(filter->quantization.params);
  TF_LITE_ENSURE(context, affine != nullptr && affine->scale != nullptr);

  const double input_scale = static_cast<double>(input->params.scale);
  const double output_scale = static_cast<double>(output->params.scale);
  const float* filter_scales = affine->scale->data;

  for (int i = 0; i < output_ch; i++) {
    // No *16 — model already has true INT4 scales from --target cmsis
    const double eff_scale =
        input_scale * static_cast<double>(filter_scales[i]) / output_scale;
    int shift_tmp;
    QuantizeMultiplier(eff_scale, &data->per_channel_multiplier[i],
                       &shift_tmp);
    data->per_channel_shift[i] = static_cast<int32_t>(shift_tmp);
  }

  // Determine if this layer fits in the pre-unpack buffer
  data->weight_bytes_int8 = output_ch * filter_h * filter_w * input_ch;
  data->use_s8_fast =
      (data->weight_bytes_int8 <= PREUNPACK_SCRATCH_KB * 1024);

  // Check if eligible for fused SMLAD path (1×1 conv, no padding, stride 1)
  data->is_1x1 = (filter_h == 1 && filter_w == 1 &&
                   params->stride_height == 1 && params->stride_width == 1 &&
                   params->dilation_height_factor == 1 &&
                   params->dilation_width_factor == 1 &&
                   (input_ch % 4 == 0));  // need multiple of 4 for SMLAD loop

  // Request scratch buffer for CMSIS-NN s8 wrapper (internal im2col etc.)
  {
    cmsis_nn_conv_params conv_p;
    conv_p.padding.w = data->padding.width;
    conv_p.padding.h = data->padding.height;
    conv_p.stride.w = params->stride_width;
    conv_p.stride.h = params->stride_height;
    conv_p.dilation.w = params->dilation_width_factor;
    conv_p.dilation.h = params->dilation_height_factor;

    cmsis_nn_dims in_dims = {1, input_h, input_w, input_ch};
    cmsis_nn_dims filt_dims = {output_ch, filter_h, filter_w, input_ch};
    cmsis_nn_dims out_dims = {1, output_h, output_w, output_ch};

    int32_t s8_buf_size = arm_convolve_wrapper_s8_get_buffer_size(
        &conv_p, &in_dims, &filt_dims, &out_dims);
    if (s8_buf_size > 0) {
      TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
          context, s8_buf_size, &data->scratch_buf_index));
    } else {
      data->scratch_buf_index = -1;
    }

    // Also request scratch for s4 fallback path
    int32_t s4_buf_size = arm_convolve_wrapper_s4_get_buffer_size(
        &conv_p, &in_dims, &filt_dims, &out_dims);
    if (s4_buf_size > 0) {
      TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
          context, s4_buf_size, &data->scratch_buf_index_s4));
    } else {
      data->scratch_buf_index_s4 = -1;
    }
  }

  // Free temp tensors
  micro_context->DeallocateTempTfLiteTensor(output);
  if (bias) micro_context->DeallocateTempTfLiteTensor(bias);
  micro_context->DeallocateTempTfLiteTensor(filter);
  micro_context->DeallocateTempTfLiteTensor(input);

  return kTfLiteOk;
}

TfLiteStatus PreunpackConvEval(TfLiteContext* context, TfLiteNode* node) {
  const auto* data =
      static_cast<const PreunpackConvOpData*>(node->user_data);
  const auto* params =
      reinterpret_cast<const TfLiteConvParams*>(node->builtin_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, 1);
  const TfLiteEvalTensor* bias =
      (node->inputs->size == 3)
          ? tflite::micro::GetEvalInput(context, node, 2)
          : nullptr;
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);

  const int8_t* input_data = tflite::micro::GetTensorData<int8_t>(input);
  const int8_t* filter_data = tflite::micro::GetTensorData<int8_t>(filter);
  const int32_t* bias_data =
      (bias != nullptr) ? tflite::micro::GetTensorData<int32_t>(bias)
                        : nullptr;
  int8_t* output_data = tflite::micro::GetTensorData<int8_t>(output);

  // Dimensions
  const int batches = input->dims->data[0];
  const int input_h = input->dims->data[1];
  const int input_w = input->dims->data[2];
  const int input_ch = input->dims->data[3];
  const int output_h = output->dims->data[1];
  const int output_w = output->dims->data[2];
  const int output_ch = output->dims->data[3];
  const int filter_h = filter->dims->data[1];
  const int filter_w = filter->dims->data[2];

  // Setup CMSIS-NN structs (shared by both paths)
  cmsis_nn_conv_params conv_params;
  conv_params.input_offset = data->input_offset;
  conv_params.output_offset = data->output_offset;
  conv_params.stride.h = params->stride_height;
  conv_params.stride.w = params->stride_width;
  conv_params.padding.h = data->padding.height;
  conv_params.padding.w = data->padding.width;
  conv_params.dilation.h = params->dilation_height_factor;
  conv_params.dilation.w = params->dilation_width_factor;
  conv_params.activation.min = data->activation_min;
  conv_params.activation.max = data->activation_max;

  cmsis_nn_per_channel_quant_params quant_params;
  quant_params.multiplier = data->per_channel_multiplier;
  quant_params.shift = data->per_channel_shift;

  cmsis_nn_dims in_dims = {batches, input_h, input_w, input_ch};
  cmsis_nn_dims filt_dims = {output_ch, filter_h, filter_w, input_ch};
  cmsis_nn_dims bias_dims = {1, 1, 1, output_ch};
  cmsis_nn_dims out_dims = {batches, output_h, output_w, output_ch};

  // Profiling: log unpack vs kernel time on the 6th inference (serial ready)
  static int s_eval_count = 0;
  s_eval_count++;
  // Inference 6 = sample index 5, logged by profiler. We log on inferences 85-98
  // (= sample 6's 14 Conv2D layers, after serial is connected)
  const bool do_profile = (s_eval_count >= 85 && s_eval_count <= 98);

  if (data->use_s8_fast) {
    // ── Fast path: unpack INT4 → INT8, then call s8 wrapper ──

    auto t0 = TIMER_GetTimeInUS();

    // Bulk unpack into static SRAM buffer
    unpack_int4_to_int8(filter_data, s_preunpack_buf,
                        data->weight_bytes_int8);

    auto t1 = TIMER_GetTimeInUS();

    // Setup context with scratch buffer for s8 kernel
    cmsis_nn_context ctx;
    if (data->scratch_buf_index >= 0) {
      ctx.buf = context->GetScratchBuffer(context, data->scratch_buf_index);
    } else {
      ctx.buf = nullptr;
    }
    ctx.size = 0;  // wrapper doesn't use this field

    arm_convolve_wrapper_s8(&ctx, &conv_params, &quant_params,
                            &in_dims, input_data,
                            &filt_dims, s_preunpack_buf,
                            &bias_dims, bias_data,
                            &out_dims, output_data);

    auto t2 = TIMER_GetTimeInUS();

    if (do_profile) {
      PRINTF("[PreUnpack] %dx%d c%d->%d  unpack=%dus  kernel=%dus  total=%dus  wt=%dB\r\n",
             filter_h, filter_w, input_ch, output_ch,
             (int)(t1-t0), (int)(t2-t1), (int)(t2-t0),
             data->weight_bytes_int8);
    }
  } else if (data->is_1x1) {
    // ── Fused path: INT4 unpack in registers + SMLAD, no SRAM buffer ──
    auto tf0 = TIMER_GetTimeInUS();

#if FUSED_BATCH_MODE == 2
    conv_1x1_fused_s4_2oc4px(
#elif FUSED_BATCH_MODE == 1
    conv_1x1_fused_s4_2oc2px(
#else
    conv_1x1_fused_s4(
#endif
                      input_data, filter_data, bias_data, output_data,
                      output_h * output_w,
                      input_ch, output_ch,
                      data->input_offset,
                      data->per_channel_multiplier,
                      data->per_channel_shift,
                      data->output_offset,
                      data->activation_min, data->activation_max);

    auto tf1 = TIMER_GetTimeInUS();
    if (do_profile) {
      PRINTF("[Fused] %dx%d c%d->%d  kernel=%dus  wt=%dB\r\n",
             filter_h, filter_w, input_ch, output_ch,
             (int)(tf1-tf0), data->weight_bytes_int8);
    }
  } else if (input_ch % 4 == 0) {
    // ── General fused path: direct conv with INT4 unpack + SMLAD ──
    // No SRAM buffer, no im2col. Works for any kernel size.
    // Requires input_ch multiple of 4 for SMLAD inner loop.
    conv_general_fused_s4(input_data, filter_data, bias_data, output_data,
                          input_h, input_w, input_ch,
                          output_h, output_w, output_ch,
                          filter_h, filter_w,
                          params->stride_height, params->stride_width,
                          data->padding.height, data->padding.width,
                          data->input_offset,
                          data->per_channel_multiplier,
                          data->per_channel_shift,
                          data->output_offset,
                          data->activation_min, data->activation_max);
  } else {
    // ── Fallback: s4 wrapper for layers with input_ch not multiple of 4 ──
    cmsis_nn_context ctx;
    if (data->scratch_buf_index_s4 >= 0) {
      ctx.buf =
          context->GetScratchBuffer(context, data->scratch_buf_index_s4);
    } else {
      ctx.buf = nullptr;
    }
    ctx.size = 0;
    arm_convolve_wrapper_s4(&ctx, &conv_params, &quant_params,
                            &in_dims, input_data,
                            &filt_dims, filter_data,
                            &bias_dims, bias_data,
                            &out_dims, output_data);
  }

  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_CONV_2D_INT4_PREUNPACK() {
  return tflite::micro::RegisterOp(PreunpackConvInit, PreunpackConvPrepare,
                                   PreunpackConvEval);
}

}  // namespace tflite
