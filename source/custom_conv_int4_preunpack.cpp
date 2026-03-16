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

// ── Fused INT4 1×1 Conv2D ──
// Reads INT4 weights directly from Flash, unpacks in registers, computes MAC.
// No SRAM scratch buffer needed — works for arbitrarily large layers.
// Only valid for 1×1 convolutions (no padding, no im2col).
//
// Phase 1: C implementation for correctness verification.
// Phase 2: Optimize inner loop with SMLAD intrinsics / inline assembly.
__attribute__((optimize("O2")))
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
  const uint8_t* w_base = reinterpret_cast<const uint8_t*>(packed_w);
  const int c_in_packed = c_in / 2;  // packed bytes per output channel

  for (int px = 0; px < num_pixels; px++) {
    const int8_t* a_row = input + px * c_in;

    for (int oc = 0; oc < c_out; oc++) {
      int32_t acc = bias ? bias[oc] : 0;
      const uint8_t* w_row = w_base + oc * c_in_packed;

      // Process 4 INT4 weights (2 bytes) per iteration, using SMLAD dual-MAC
      int j = 0;
      for (; j + 3 < c_in; j += 4) {
        // Load 2 packed bytes = 4 INT4 weights
        uint32_t packed =
            *reinterpret_cast<const uint16_t*>(w_row + j / 2);

        // Extract and sign-extend 4 nibbles using SBFX
        int32_t w0, w1, w2, w3;
        __ASM volatile("sbfx %0, %1, #0, #4"  : "=r"(w0) : "r"(packed));
        __ASM volatile("sbfx %0, %1, #4, #4"  : "=r"(w1) : "r"(packed));
        __ASM volatile("sbfx %0, %1, #8, #4"  : "=r"(w2) : "r"(packed));
        __ASM volatile("sbfx %0, %1, #12, #4" : "=r"(w3) : "r"(packed));

        // Load 4 activations + input_offset
        int32_t a0 = a_row[j]     + input_offset;
        int32_t a1 = a_row[j + 1] + input_offset;
        int32_t a2 = a_row[j + 2] + input_offset;
        int32_t a3 = a_row[j + 3] + input_offset;

        // Pack pairs for SMLAD: {hi16, lo16}
        uint32_t wp02, wp13, ap02, ap13;
        __ASM volatile("pkhbt %0, %1, %2, lsl #16"
                        : "=r"(wp02) : "r"(w0), "r"(w2));
        __ASM volatile("pkhbt %0, %1, %2, lsl #16"
                        : "=r"(wp13) : "r"(w1), "r"(w3));
        __ASM volatile("pkhbt %0, %1, %2, lsl #16"
                        : "=r"(ap02) : "r"(a0), "r"(a2));
        __ASM volatile("pkhbt %0, %1, %2, lsl #16"
                        : "=r"(ap13) : "r"(a1), "r"(a3));

        // Dual MACs: acc += w0*a0 + w2*a2, then acc += w1*a1 + w3*a3
        __ASM volatile("smlad %0, %1, %2, %0"
                        : "+r"(acc) : "r"(wp02), "r"(ap02));
        __ASM volatile("smlad %0, %1, %2, %0"
                        : "+r"(acc) : "r"(wp13), "r"(ap13));
      }

      // Handle remaining channels (< 4)
      for (; j < c_in; j++) {
        const uint8_t byte = w_row[j / 2];
        int8_t w;
        if (j % 2 == 0) {
          w = static_cast<int8_t>(byte & 0x0F);
          if (w & 0x08) w |= 0xF0;
        } else {
          w = static_cast<int8_t>((byte >> 4) & 0x0F);
          if (w & 0x08) w |= 0xF0;
        }
        acc += w * (static_cast<int32_t>(a_row[j]) + input_offset);
      }

      // Per-channel requantize
      acc = arm_nn_requantize(acc, multiplier[oc], shift[oc]);
      acc += output_offset;
      acc = MAX(acc, act_min);
      acc = MIN(acc, act_max);
      output[px * c_out + oc] = static_cast<int8_t>(acc);
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

  if (data->use_s8_fast) {
    // ── Fast path: unpack INT4 → INT8, then call s8 wrapper ──

    // Bulk unpack into static SRAM buffer
    unpack_int4_to_int8(filter_data, s_preunpack_buf,
                        data->weight_bytes_int8);

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
  } else if (data->is_1x1) {
    // ── Fused path: INT4 unpack in registers + SMLAD, no SRAM buffer ──
    // For 1×1 convolutions that don't fit in pre-unpack scratch.
    conv_1x1_fused_s4(input_data, filter_data, bias_data, output_data,
                      output_h * output_w,  // num_pixels
                      input_ch, output_ch,
                      data->input_offset,
                      data->per_channel_multiplier,
                      data->per_channel_shift,
                      data->output_offset,
                      data->activation_min, data->activation_max);
  } else {
    // ── Fallback: call s4 wrapper directly on packed INT4 weights ──

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
