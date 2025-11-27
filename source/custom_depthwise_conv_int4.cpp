/*
 * custom_int4_unpack.cpp
 *
 *  Created on: 2025.11.18
 *      Author: yx_wu
 */

#include "custom_int4_unpack.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_float.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h" 
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/kernels/kernel_util.h"

#include <cmath>

namespace tflite {
  namespace {

    // Opdata: to pass precomputed params between Prepare and Eval
    struct OpData {
      TfLitePaddingValues padding;

      // Quantization parameters
      int32_t* per_channel_output_multiplier;
      int* per_channel_output_shift;

      // common params
      int32_t output_offset;
      int32_t input_offset;
      int32_t output_activation_min;
      int32_t output_activation_max;
    };

    // Util: assign OpData
    void* Init(TfLiteContext* context, const char* buffer, size_t length) {
      return context->AllocatePersistentBuffer(
          context, sizeof(OpData));
    }

    // Prepare : compute quantization params and padding
    TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
      OpData *data = static_cast<OpData *>(node->user_data);

      TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);   // input, filter, bias
      TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

      const TfLiteTensor *input = GetInput(context, node, 0);
      const TfLiteTensor *filter = GetInput(context, node, 1);
      const TfLiteTensor *bias = GetInput(context, node, 2);
      TfLiteTensor *output = GetOutput(context, node, 0);

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
      data->input_offset = -input->params.zero_point;
      data->output_offset = output->params.zero_point;

      // clamp
      CalculateActivationRangeQuantized(
          context, params->activation, output,
          &data->output_activation_min,
          &data->output_activation_max
      );

      // per-channel quantization
      int num_channels = filter->dims->data[3];

      // allocate per-channel multiplier and shift
      data->per_channel_output_multiplier = static_cast<int32_t *>(
          context->AllocatePersistentBuffer(context, num_channels * sizeof(int32_t))
      );
      data->per_channel_output_shift = static_cast<int *>(
          context->AllocatePersistentBuffer(context, num_channels * sizeof(int))
      );

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
        const double effective_scale = (input_scale * filter_scale) / output_scale;
        QuantizeMultiplier(effective_scale,
                           &data->per_channel_output_multiplier[i],
                           &data->per_channel_output_shift[i]);
      }

      return kTfLiteOk;
    }

    // Custom Eval function to handle packed INT4 weights
    TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
      const OpData* data = static_cast<const OpData *>(node->user_data);
      const auto* params = reinterpret_cast<const TfLiteDepthwiseConvParams*>(node->builtin_data);
      
      // --- get input tensors, params ---
      const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
      const TfLiteEvalTensor* filer = tflite::micro::GetEvalInput(context, node, 1);
      const TfLiteEvalTensor* bias = tflite::micro::GetEvalInput(context, node, 2);
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

      // --- Depthwise Conv ---
      for (int b = 0; b < batches; ++b) {
        for (int out_y = 0; out_y < output_height; ++out_y) {
          for (int out_x = 0; out_x < output_width; ++out_x) {
            for (int ch = 0; ch < output_depth; ++ch) {
              // accumulator init
              int32_t acc = (bias_data) ? bias_data[ch] : 0;

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

              // clamp
              acc = std::max(acc, data->output_activation_min);
              acc = std::min(acc, data->output_activation_max);

              // store output
              int output_idx = ((b * output_height + out_y) * output_width + out_x) * output_depth + ch;
              output_data[output_idx] = static_cast<int8_t>(acc);
            }
          }
        }
      }
      
      return kTfLiteOk;
    }
  }

  TFLMRegistration Register_CUSTOM_DEPTHWISE_CONV_INT4() {
    return tflite::micro::RegisterOp(Init, Prepare, Eval);
  }

}
