/*
 * custom_int4_unpack.cpp
 *
 *  Created on: 2025.11.18
 *      Author: yx_wu
 */

#include "custom_int4_unpack.h"
#include "custom_depthwise_conv_int4.h"

// TFLM
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h"
#include "tensorflow/lite/micro/micro_context.h"

// TFLite
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"

#include <cmath>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

namespace tflite {
  namespace {
    static inline void DebugLogNDJSONPrintf(const char* run_id,
                                            const char* hypothesis_id,
                                            const char* location,
                                            const char* message,
                                            const char* data_fmt,
                                            ...) {
      char data_buf[256];
      data_buf[0] = '\0';
      va_list args;
      va_start(args, data_fmt);
      vsnprintf(data_buf, sizeof(data_buf), data_fmt, args);
      va_end(args);
      // NOTE: 打印到串口，由上位机脚本采集写入 debug.log
      printf("{\"sessionId\":\"debug-session\",\"runId\":\"%s\",\"hypothesisId\":\"%s\","
             "\"location\":\"%s\",\"message\":\"%s\",\"data\":%s,\"timestamp\":0}\r\n",
             run_id, hypothesis_id, location, message, data_buf);
    }

    static inline uint32_t FloatBits(float v) {
      uint32_t out = 0;
      memcpy(&out, &v, sizeof(out));
      return out;
    }

    #define MAX_DEPTHWISE_CHANNELS 512 // alpha = 0.25

    // Opdata: to pass precomputed params between Prepare and Eval
    struct DepthwiseOpData {
      TfLitePaddingValues padding;

      // Quantization parameters
      int32_t* per_channel_output_multiplier = nullptr;
      int* per_channel_output_shift = nullptr;

      // static allocation
      // int32_t per_channel_output_multiplier[MAX_DEPTHWISE_CHANNELS];
      // int per_channel_output_shift[MAX_DEPTHWISE_CHANNELS];

      // common params
      int32_t output_offset;
      int32_t input_offset;
      int32_t output_activation_min;
      int32_t output_activation_max;
    };

    // Util: assign OpData
    void* DepthwiseInit_INT4(TfLiteContext* context, const char* buffer, size_t length) {
      // return context->AllocatePersistentBuffer(context, sizeof(DepthwiseOpData));
      // void* raw = context->AllocatePersistentBuffer(context, sizeof(DepthwiseOpData));
      // DepthwiseOpData* data = static_cast<DepthwiseOpData*>(raw);
      // if (data) {
      //   data->per_channel_output_multiplier = nullptr;
      //   data->per_channel_output_shift = nullptr;
      // }
      // return raw;

      return context->AllocatePersistentBuffer(context, sizeof(DepthwiseOpData));
    }

    // Prepare : compute quantization params and padding
    TfLiteStatus DepthwisePrepare_INT4(TfLiteContext* context, TfLiteNode* node) {
      DepthwiseOpData *data = static_cast<DepthwiseOpData *>(node->user_data);

      // check OpData
      if (data == nullptr) {
          printf("Error: OpData is null in Prepare! Init failed?\r\n");
          return kTfLiteError;
      }

      // bias can be optional
      TF_LITE_ENSURE(context, NumInputs(node) == 2 || NumInputs(node) == 3);
      TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

      // const TfLiteTensor *input = GetInput(context, node, 0);
      // const TfLiteTensor *filter = GetInput(context, node, 1);

      // Get micro context(this is where needs to be freed later)
      MicroContext* micro_context = GetMicroContext(context);

      TfLiteTensor* input = micro_context->AllocateTempTfLiteTensor(node->inputs->data[0]);
      TfLiteTensor* filter = micro_context->AllocateTempTfLiteTensor(node->inputs->data[1]);

      // bias is optional
      TfLiteTensor *bias = (NumInputs(node) == 3) ? micro_context->AllocateTempTfLiteTensor(node->inputs->data[2]) : nullptr;
      TfLiteTensor *output = micro_context->AllocateTempTfLiteTensor(node->outputs->data[0]);

      const auto* params = reinterpret_cast<const TfLiteDepthwiseConvParams*>(node->builtin_data);

      // Padding
      int input_width = input->dims->data[2];
      int input_height = input->dims->data[1];
      int filter_width = filter->dims->data[2];
      int filter_height = filter->dims->data[1];
      int output_width = output->dims->data[2];
      int output_height = output->dims->data[1];

      data->padding = ComputePaddingHeightWidth(
          params->stride_height, params->stride_width, 1, 1,
          input_height, input_width, filter_height, filter_width,
          params->padding, &output_height, &output_width
      );

      // Quantization params

      // debug print
      // 1. read params from metadata
      data->input_offset = -input->params.zero_point; 

      // 2. check this value, is it too large?
      printf("DEBUG CHECK: Model ZeroPoint is %d, Calculated Offset is %ld\r\n", 
            input->params.zero_point, data->input_offset);

      // set to 1, only used for the input from outside
      // data->input_offset = 1;
      data->output_offset = output->params.zero_point;

      // clamp
      CalculateActivationRangeQuantized(
          context, params->activation, output,
          &data->output_activation_min,
          &data->output_activation_max
      );

      if (bias != nullptr) {
        // #region agent log
        DebugLogNDJSONPrintf(
            "pre-fix",
            "H7",
            "custom_depthwise_conv_int4.cpp:prepare_params",
            "dw_prepare_params",
            "{\"input_scale_bits\":%u,\"input_zero_point\":%d,"
            "\"output_scale_bits\":%u,\"output_zero_point\":%d,"
            "\"bias_scale_bits\":%u,\"bias_zero_point\":%d}",
            static_cast<unsigned int>(FloatBits(input->params.scale)),
            static_cast<int>(input->params.zero_point),
            static_cast<unsigned int>(FloatBits(output->params.scale)),
            static_cast<int>(output->params.zero_point),
            static_cast<unsigned int>(FloatBits(bias->params.scale)),
            static_cast<int>(bias->params.zero_point));
        // #endregion
      }

      // per-channel quantization
      int num_channels = filter->dims->data[3];

      // check channels
      if (num_channels > MAX_DEPTHWISE_CHANNELS) {
          printf("Error: number of channels (%d) exceeds max supported (%d)\r\n", num_channels, MAX_DEPTHWISE_CHANNELS);
          return kTfLiteError;
      }

      // allocate only if nullptr
      if (data->per_channel_output_multiplier == nullptr) {
          data->per_channel_output_multiplier = static_cast<int32_t *>(
              context->AllocatePersistentBuffer(context, num_channels * sizeof(int32_t))
          );
      }

      if (data->per_channel_output_shift == nullptr) {
          data->per_channel_output_shift = static_cast<int *>(
              context->AllocatePersistentBuffer(context, num_channels * sizeof(int))
          );
      }

      // check allocation
      TF_LITE_ENSURE(context, data->per_channel_output_multiplier != nullptr);
      TF_LITE_ENSURE(context, data->per_channel_output_shift != nullptr);

      // get Filter quantization params (AffineQuantization)
      const auto* affine_quantization = 
          reinterpret_cast<TfLiteAffineQuantization *>(filter->quantization.params);
      
      TF_LITE_ENSURE(context, affine_quantization != nullptr);
      TF_LITE_ENSURE(context, affine_quantization->scale != nullptr);

      // Compute per-channel output multiplier and shift
      // Effective scale = (input_scale * filter_scale) / output_scale
      const double input_scale = input->params.scale;
      const double output_scale = output->params.scale;
      const float* filter_scales = affine_quantization->scale->data;

      for (int i = 0; i < num_channels; ++i) {
        const double filter_scale = static_cast<double>(filter_scales[i]);
        const double effective_scale = (input_scale * filter_scale * 16.0) / output_scale;
        QuantizeMultiplier(effective_scale,
                           &data->per_channel_output_multiplier[i],
                           &data->per_channel_output_shift[i]);
      }

      // free temp tensors
      micro_context->DeallocateTempTfLiteTensor(output);
      if (bias) micro_context->DeallocateTempTfLiteTensor(bias);
      micro_context->DeallocateTempTfLiteTensor(filter);
      micro_context->DeallocateTempTfLiteTensor(input);

      return kTfLiteOk;
    }

    // Custom Eval function to handle packed INT4 weights
    TfLiteStatus DepthwiseEval_INT4(TfLiteContext* context, TfLiteNode* node) {
      const DepthwiseOpData* data = static_cast<const DepthwiseOpData *>(node->user_data);
      const auto* params = reinterpret_cast<const TfLiteDepthwiseConvParams*>(node->builtin_data);
      
      // --- get input tensors, params ---
      const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
      const TfLiteEvalTensor* filer = tflite::micro::GetEvalInput(context, node, 1);

      // bias is optional
      const TfLiteEvalTensor* bias = (NumInputs(node) == 3) ? tflite::micro::GetEvalInput(context, node, 2) : nullptr;
      TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);

      // get data pointers
      const int8_t* input_data = tflite::micro::GetTensorData<int8_t>(input);
      const int8_t* packed_filter_data = tflite::micro::GetTensorData<int8_t>(filer);
      const int32_t *bias_data = (bias != nullptr) ? tflite::micro::GetTensorData<int32_t>(bias) : nullptr;
      int8_t* output_data = tflite::micro::GetTensorData<int8_t>(output);

      // --- extract dimentions ---
      const int batches = input->dims->data[0];
      const int input_height = input->dims->data[1];
      const int input_width = input->dims->data[2];
      const int input_depth = input->dims->data[3];
      const int filter_height = filer->dims->data[1];
      const int filter_width = filer->dims->data[2];
      const int output_height = output->dims->data[1];
      const int output_width = output->dims->data[2];
      const int output_depth = output->dims->data[3];

      const int stride_height = params->stride_height;
      const int stride_width = params->stride_width;
      const int pad_height = data->padding.height;
      const int pad_width = data->padding.width;

      static int dw_debug_once = 0;

      // --- Depthwise Conv ---
      for (int b = 0; b < batches; ++b) {
        for (int out_y = 0; out_y < output_height; ++out_y) {
          for (int out_x = 0; out_x < output_width; ++out_x) {
            for (int ch = 0; ch < output_depth; ++ch) {
              // accumulator init
              int32_t acc = (bias_data) ? bias_data[ch] : 0;
              acc = acc >> 4; // keep bias scale consistent with packed int4 weights

              const bool dw_debug = (b == 0 && out_y == 0 && out_x == 0 && ch == 0);
              if (dw_debug && dw_debug_once == 0) {
                // #region agent log
                DebugLogNDJSONPrintf(
                    "pre-fix",
                    "H10",
                    "custom_depthwise_conv_int4.cpp:bias_shift",
                    "dw_eval_bias_shift",
                    "{\"bias_raw\":%ld,\"bias_shifted\":%ld}",
                    static_cast<long>(bias_data ? bias_data[ch] : 0),
                    static_cast<long>(bias_data ? (bias_data[ch] >> 4) : 0));
                // #endregion
              }

              // dismiss bias
              // acc = 0;

              const int in_y_origin = (out_y * stride_height) - pad_height;
              const int in_x_origin = (out_x * stride_width) - pad_width;

              // filter loop
              for (int fy = 0; fy < filter_height; ++fy) {
                for (int fx = 0; fx < filter_width; ++fx) {
                  const int in_y = in_y_origin + fy;
                  const int in_x = in_x_origin + fx;

                  // check input bounds, zero padding
                  int32_t input_val = 0;
                  if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width) {
                    // compute NHWC flat index: ((b * H + y) * W + x) * D + ch
                    const int input_idx = ((b * input_height + in_y) * input_width + in_x) * input_depth + ch;
                    input_val = input_data[input_idx];
                    
                    // get packed INT4 filter value
                    // suppose NHWC [1, fy, fx, ch]
                    int filter_flat_index = fy * filter_width * output_depth + fx * output_depth + ch;

                    int32_t filter_val = GET_INT4_WEIGHT(packed_filter_data, filter_flat_index);

                    if (b == 0 && out_y == 0 && out_x == 0 && ch == 0 && dw_debug_once == 0) {
                      // #region agent log
                      DebugLogNDJSONPrintf(
                          "pre-fix",
                          "H8",
                          "custom_depthwise_conv_int4.cpp:pack_probe",
                          "dw_eval_first8_weights",
                          "{\"w0\":%d,\"w1\":%d,\"w2\":%d,\"w3\":%d,\"w4\":%d,\"w5\":%d,"
                          "\"w6\":%d,\"w7\":%d,\"pb0\":%u,\"pb1\":%u}",
                          static_cast<int>(GET_INT4_WEIGHT(packed_filter_data, 0)),
                          static_cast<int>(GET_INT4_WEIGHT(packed_filter_data, 1)),
                          static_cast<int>(GET_INT4_WEIGHT(packed_filter_data, 2)),
                          static_cast<int>(GET_INT4_WEIGHT(packed_filter_data, 3)),
                          static_cast<int>(GET_INT4_WEIGHT(packed_filter_data, 4)),
                          static_cast<int>(GET_INT4_WEIGHT(packed_filter_data, 5)),
                          static_cast<int>(GET_INT4_WEIGHT(packed_filter_data, 6)),
                          static_cast<int>(GET_INT4_WEIGHT(packed_filter_data, 7)),
                          static_cast<unsigned int>(((const uint8_t*)packed_filter_data)[0]),
                          static_cast<unsigned int>(((const uint8_t*)packed_filter_data)[1]));
                      // #endregion
                    }
                    
                    // MAC
                    acc += (input_val + data->input_offset) * filter_val;
                  
                  } 
                }
              }
              // --- Requantization ---
              // get 32->8 bit quantization params
              acc = MultiplyByQuantizedMultiplier(
                  acc,
                  data->per_channel_output_multiplier[ch],
                  data->per_channel_output_shift[ch]
              );

              acc += data->output_offset;

              if (dw_debug && dw_debug_once == 0) {
                // #region agent log
                DebugLogNDJSONPrintf(
                    "pre-fix",
                    "H9",
                    "custom_depthwise_conv_int4.cpp:requant",
                    "dw_eval_requant_channel0",
                    "{\"acc_post_mult\":%ld,\"output_offset\":%ld,"
                    "\"out_act_min\":%ld,\"out_act_max\":%ld}",
                    static_cast<long>(acc),
                    data->output_offset,
                    data->output_activation_min,
                    data->output_activation_max);
                // #endregion
              }
              // clamp
              acc = std::max(acc, data->output_activation_min);
              acc = std::min(acc, data->output_activation_max);

              if (dw_debug && dw_debug_once == 0) {
                // #region agent log
                DebugLogNDJSONPrintf(
                    "pre-fix",
                    "H11",
                    "custom_depthwise_conv_int4.cpp:clamp",
                    "dw_eval_clamp_channel0",
                    "{\"acc_clamped\":%ld}",
                    static_cast<long>(acc));
                // #endregion
              }

              // store output
              int output_idx = ((b * output_height + out_y) * output_width + out_x) * output_depth + ch;
              output_data[output_idx] = static_cast<int8_t>(acc);
            }
          }
        }
      }

      if (batches == 1 && output_depth > 0 && dw_debug_once == 0) {
        dw_debug_once = 1;
      }
      
      return kTfLiteOk;
    }
  }

  TFLMRegistration Register_CUSTOM_DEPTHWISE_CONV_INT4() {
    return tflite::micro::RegisterOp(DepthwiseInit_INT4, DepthwisePrepare_INT4, DepthwiseEval_INT4);
  }

}
