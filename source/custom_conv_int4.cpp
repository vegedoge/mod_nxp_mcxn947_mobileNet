/*
 * custom_conv_int4.cpp
 *
 * Custom INT4 Conv2D operator for TFLM.
 * Unpacks INT4 weights at runtime, performs INT32 MAC,
 * and requantizes with per-channel fixed-point multipliers.
 *
 * Works with any Conv2D layer (ResNet-20, MobileNet, etc).
 */
#include "custom_int4_unpack.h"
#include "custom_conv_int4.h"

// TFLM
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/micro/micro_context.h"

// TFLite
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"

#include "demo_config.h"

#include <string.h>

namespace tflite {
  namespace {

    struct ConvOpData {
      TfLitePaddingValues padding;
      int32_t* per_channel_output_multiplier = nullptr;
      int* per_channel_output_shift = nullptr;
      int32_t output_offset;
      int32_t input_offset;
      int32_t output_activation_min;
      int32_t output_activation_max;
    };

    void* ConvInit_INT4(TfLiteContext* context, const char* buffer, size_t length) {
      return context->AllocatePersistentBuffer(context, sizeof(ConvOpData));
    }

    TfLiteStatus ConvPrepare_INT4(TfLiteContext* context, TfLiteNode* node) {
      ConvOpData* data = static_cast<ConvOpData *>(node->user_data);
      if (data == nullptr) {
          return kTfLiteError;
      }

      TF_LITE_ENSURE(context, node->inputs->size == 2 || node->inputs->size == 3);
      TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

      MicroContext* micro_context = GetMicroContext(context);
      TfLiteTensor* input = micro_context->AllocateTempTfLiteTensor(node->inputs->data[0]);
      TfLiteTensor* filter = micro_context->AllocateTempTfLiteTensor(node->inputs->data[1]);
      TfLiteTensor* bias = (node->inputs->size == 3)
                               ? micro_context->AllocateTempTfLiteTensor(node->inputs->data[2])
                               : nullptr;
      TfLiteTensor* output = micro_context->AllocateTempTfLiteTensor(node->outputs->data[0]);

      if (input == nullptr) {
          return kTfLiteError;
      }

      const auto* params = reinterpret_cast<const TfLiteConvParams*>(node->builtin_data);

      // Padding
      int input_width = input->dims->data[2];
      int input_height = input->dims->data[1];
      int filter_width = filter->dims->data[2];
      int filter_height = filter->dims->data[1];
      int output_width = output->dims->data[2];
      int output_height = output->dims->data[1];

      data->padding = ComputePaddingHeightWidth(
        params->stride_height, params->stride_width,
        params->dilation_height_factor, params->dilation_width_factor,
        input_height, input_width, filter_height, filter_width,
        params->padding, &output_height, &output_width
      );

      data->input_offset = -input->params.zero_point;
      data->output_offset = output->params.zero_point;

      CalculateActivationRangeQuantized(
          context, params->activation, output,
          &data->output_activation_min,
          &data->output_activation_max
      );

      // Per-channel quantization
      int num_channels = filter->dims->data[0];

      if (data->per_channel_output_multiplier == nullptr) {
        data->per_channel_output_multiplier =
            static_cast<int32_t *>(context->AllocatePersistentBuffer(
                context, num_channels * sizeof(int32_t)));
      }
      if (data->per_channel_output_shift == nullptr) {
        data->per_channel_output_shift =
            static_cast<int *>(context->AllocatePersistentBuffer(
                context, num_channels * sizeof(int)));
      }

      TF_LITE_ENSURE(context, data->per_channel_output_multiplier != nullptr);
      TF_LITE_ENSURE(context, data->per_channel_output_shift != nullptr);

      const auto* affine_quantization =
          reinterpret_cast<TfLiteAffineQuantization *>(filter->quantization.params);
      TF_LITE_ENSURE(context, affine_quantization != nullptr);
      TF_LITE_ENSURE(context, affine_quantization->scale != nullptr);

      const double input_scale = input->params.scale;
      const double output_scale = output->params.scale;
      const float *filter_scales = affine_quantization->scale->data;

      for (int i = 0; i < num_channels; ++i) {
        const double filter_scale = static_cast<double>(filter_scales[i]);
        // The *16.0 accounts for INT4 unpacking (INT4 range is 1/16 of INT8)
        const double effective_scale = (input_scale * filter_scale * 16.0) / output_scale;
        QuantizeMultiplier(effective_scale,
                           &data->per_channel_output_multiplier[i],
                           &data->per_channel_output_shift[i]);
      }

      // Free temp tensors
      micro_context->DeallocateTempTfLiteTensor(output);
      if (bias) micro_context->DeallocateTempTfLiteTensor(bias);
      micro_context->DeallocateTempTfLiteTensor(filter);
      micro_context->DeallocateTempTfLiteTensor(input);

      return kTfLiteOk;
    }

    TfLiteStatus ConvEval_INT4(TfLiteContext* context, TfLiteNode* node) {
      const ConvOpData *data = static_cast<const ConvOpData *>(node->user_data);
      const auto *params = reinterpret_cast<const TfLiteConvParams *>(node->builtin_data);

      const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
      const TfLiteEvalTensor* filter = tflite::micro::GetEvalInput(context, node, 1);
      const TfLiteEvalTensor* bias = (node->inputs->size == 3)
          ? tflite::micro::GetEvalInput(context, node, 2) : nullptr;
      TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);

      const int8_t* input_data = tflite::micro::GetTensorData<int8_t>(input);
      const int8_t* filter_data = tflite::micro::GetTensorData<int8_t>(filter);
      const int32_t *bias_data = (bias != nullptr)
          ? tflite::micro::GetTensorData<int32_t>(bias) : nullptr;
      int8_t *output_data = tflite::micro::GetTensorData<int8_t>(output);

      const int batches = input->dims->data[0];
      const int input_height = input->dims->data[1];
      const int input_width = input->dims->data[2];
      const int input_depth = input->dims->data[3];

      const int filter_height = filter->dims->data[1];
      const int filter_width = filter->dims->data[2];

      const int output_height = output->dims->data[1];
      const int output_width = output->dims->data[2];
      const int output_depth = output->dims->data[3];

      const int stride_height = params->stride_height;
      const int stride_width = params->stride_width;
      const int pad_height = data->padding.height;
      const int pad_width = data->padding.width;

      for (int b = 0; b < batches; ++b) {
        for (int out_y = 0; out_y < output_height; ++out_y) {
          for (int out_x = 0; out_x < output_width; ++out_x) {
            for (int out_ch = 0; out_ch < output_depth; ++out_ch) {

              // Initialize accumulator with bias >> 4
              int32_t acc = (bias_data) ? (bias_data[out_ch] >> 4) : 0;

              const int in_y_origin = (out_y * stride_height) - pad_height;
              const int in_x_origin = (out_x * stride_width) - pad_width;

              // Convolution MAC loop
              for (int fy = 0; fy < filter_height; ++fy) {
                for (int fx = 0; fx < filter_width; ++fx) {
                  for (int ic = 0; ic < input_depth; ++ic) {
                    const int in_y = in_y_origin + fy;
                    const int in_x = in_x_origin + fx;

                    if ((in_y >= 0) && (in_y < input_height) &&
                        (in_x >= 0) && (in_x < input_width)) {

                      const int input_index =
                          b * (input_height * input_width * input_depth) +
                          in_y * (input_width * input_depth) +
                          in_x * input_depth +
                          ic;

                      const int filter_index =
                          out_ch * (filter_height * filter_width * input_depth) +
                          fy * (filter_width * input_depth) +
                          fx * input_depth +
                          ic;

                      const int8_t input_val = input_data[input_index];
                      const int8_t filter_val = GET_INT4_WEIGHT(filter_data, filter_index);

                      acc += (static_cast<int32_t>(input_val + data->input_offset) *
                              static_cast<int32_t>(filter_val));
                    }
                  }
                }
              }

              // Requantization: fixed-point multiply + shift
              acc = MultiplyByQuantizedMultiplier(
                  acc,
                  data->per_channel_output_multiplier[out_ch],
                  data->per_channel_output_shift[out_ch]
              );
              acc += data->output_offset;

              // Clamp to activation range
              acc = std::max(acc, data->output_activation_min);
              acc = std::min(acc, data->output_activation_max);

              const int output_index =
                  b * (output_height * output_width * output_depth) +
                  out_y * (output_width * output_depth) +
                  out_x * output_depth +
                  out_ch;
              output_data[output_index] = static_cast<int8_t>(acc);
            }
          }
        }
      }

      return kTfLiteOk;
    }

  }   // namespace

  TFLMRegistration Register_CUSTOM_CONV_INT4() {
    return tflite::micro::RegisterOp(ConvInit_INT4, ConvPrepare_INT4, ConvEval_INT4);
  }

} // namespace tflite
