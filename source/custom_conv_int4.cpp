/*
 * custom_conv_int4.cpp
 *
 *  Created on: 2025年11月27日
 *      Author: yx_wu
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

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

namespace tflite {
  namespace {

    #define MAX_CONV_CHANNELS 512 // alpha = 0.25

    // Opdata: to pass precomputed params between Prepare and Eval
    struct ConvOpData {
      TfLitePaddingValues padding;

      // Quantization parameters (dynamic allocation)
      int32_t* per_channel_output_multiplier = nullptr;
      int* per_channel_output_shift = nullptr;

      // static allocation
      // int32_t per_channel_output_multiplier[MAX_CONV_CHANNELS];
      // int per_channel_output_shift[MAX_CONV_CHANNELS];

      // common params
      int32_t output_offset;
      int32_t input_offset;
      int32_t output_activation_min;
      int32_t output_activation_max;
    };

    static inline void DebugLogNDJSONPrintf(const char* run_id,
                                            const char* hypothesis_id,
                                            const char* location,
                                            const char* message,
                                            const char* data_fmt,
                                            ...) {
#if DEBUG_PRINTS
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
#else
      (void)run_id;
      (void)hypothesis_id;
      (void)location;
      (void)message;
      (void)data_fmt;
#endif
    }

    static inline uint32_t FloatBits(float v) {
      uint32_t out = 0;
      memcpy(&out, &v, sizeof(out));
      return out;
    }

    void* ConvInit_INT4(TfLiteContext* context, const char* buffer, size_t length) {
      // return context->AllocatePersistentBuffer(context, sizeof(ConvOpData));
      // void* raw = context->AllocatePersistentBuffer(context, sizeof(ConvOpData));
      // ConvOpData* data = static_cast<ConvOpData*>(raw);
      // if (data) {
      //   data->per_channel_output_multiplier = nullptr;
      //   data->per_channel_output_shift = nullptr;
      // }
      // return raw;

      return context->AllocatePersistentBuffer(context, sizeof(ConvOpData));
    }
      

    // Prepare
    TfLiteStatus ConvPrepare_INT4(TfLiteContext* context, TfLiteNode* node) {
      ConvOpData* data = static_cast<ConvOpData *>(node->user_data);

      // check OpData
      if (data == nullptr) {
          printf("Error: OpData is null in Prepare! Init failed?\r\n");
          return kTfLiteError;
      }

      // // we now allow 2 or 3 inputs (bias is optional)
      TF_LITE_ENSURE(context, node->inputs->size == 2 || node->inputs->size == 3);
      TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

      // Get micro context(this is where needs to be freed later)
      MicroContext* micro_context = GetMicroContext(context);

      // directly GetTensor to access quantization params
      // const TfLiteTensor *input = context->GetTensor(context, node->inputs->data[0]);
      TfLiteTensor* input = micro_context->AllocateTempTfLiteTensor(node->inputs->data[0]);
      TfLiteTensor* filter = micro_context->AllocateTempTfLiteTensor(node->inputs->data[1]);
      TfLiteTensor* bias = (node->inputs->size == 3)
                               ? micro_context->AllocateTempTfLiteTensor(node->inputs->data[2])
                               : nullptr;
      TfLiteTensor* output = micro_context->AllocateTempTfLiteTensor(node->outputs->data[0]);

      // check nullptr
      if (input == nullptr) {
          printf("Error: input tensor is null\r\n");
          // add something to avoid node from optimization removal
          int size = node->inputs->size;
          printf("Num inputs: %d\n", size);
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

      // offset quantization

      // debug print
      // 1. read params from metadata
      data->input_offset = -input->params.zero_point; 

      // 2. check this value, is it too large?
#if DEBUG_PRINTS
      printf("DEBUG CHECK: Model ZeroPoint is %d, Calculated Offset is %ld\r\n", 
            input->params.zero_point, data->input_offset);
#endif

      // if (input->dims->data[3] == 3) {
      //   printf("DEBUG: Detected First Layer (RGB). Forcing Input Offset to 1.\r\n");
      //   data->input_offset = 1;
      // }

      // set to 1
      // data->input_offset = 1;

      data->output_offset = output->params.zero_point;

      // Actication range
      CalculateActivationRangeQuantized(
          context, params->activation, output,
          &data->output_activation_min,
          &data->output_activation_max
      );

      // #region agent log
      DebugLogNDJSONPrintf(
          "pre-fix",
          "H1",
          "custom_conv_int4.cpp:input_offset",
          "conv_prepare_input_output_params",
          "{\"input_zero_point\":%d,\"input_offset\":%ld,"
          "\"input_scale_bits\":%u,\"output_scale_bits\":%u,"
          "\"output_zero_point\":%d,\"out_act_min\":%ld,\"out_act_max\":%ld,"
          "\"input_type\":%d,\"output_type\":%d}",
          input->params.zero_point,
          data->input_offset,
          static_cast<unsigned int>(FloatBits(input->params.scale)),
          static_cast<unsigned int>(FloatBits(output->params.scale)),
          output->params.zero_point,
          data->output_activation_min,
          data->output_activation_max,
          static_cast<int>(input->type),
          static_cast<int>(output->type));
      // #endregion

      if (bias != nullptr) {
        // #region agent log
        DebugLogNDJSONPrintf(
            "pre-fix",
            "H6",
            "custom_conv_int4.cpp:bias_params",
            "conv_prepare_bias_params",
            "{\"bias_type\":%d,\"bias_scale_bits\":%u,\"bias_zero_point\":%d}",
            static_cast<int>(bias->type),
            static_cast<unsigned int>(FloatBits(bias->params.scale)),
            static_cast<int>(bias->params.zero_point));
        // #endregion
      }

      // Per-channel quantization parameters
      // Conv2D filter shape: [output_channels, filter_height, filter_width, input_channels]
      int num_channels = filter->dims->data[0];

      // only allocate if nullptr
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

      // check allocations
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
        const double effective_scale = (input_scale * filter_scale * 16.0) / output_scale;
        QuantizeMultiplier(effective_scale,
                           &data->per_channel_output_multiplier[i],
                           &data->per_channel_output_shift[i]);
        if (i == 0) {
          const float eff_scale_f = static_cast<float>(effective_scale);
          // #region agent log
      DebugLogNDJSONPrintf(
              "pre-fix",
              "H3",
              "custom_conv_int4.cpp:per_channel_quant",
              "conv_prepare_scale_channel0",
              "{\"filter_scale_bits\":%u,\"effective_scale_bits\":%u,"
              "\"multiplier\":%ld,\"shift\":%d}",
              static_cast<unsigned int>(FloatBits(filter_scales[i])),
              static_cast<unsigned int>(FloatBits(eff_scale_f)),
              static_cast<long>(data->per_channel_output_multiplier[i]),
              data->per_channel_output_shift[i]);
          // #endregion
        }
      }

      // free temp tensors
      micro_context->DeallocateTempTfLiteTensor(output);
      if (bias) micro_context->DeallocateTempTfLiteTensor(bias);
      micro_context->DeallocateTempTfLiteTensor(filter);
      micro_context->DeallocateTempTfLiteTensor(input);

      return kTfLiteOk;
    }

    // Eval
    TfLiteStatus ConvEval_INT4(TfLiteContext* context, TfLiteNode* node) {
      // debug
      static int print_count = 0;
      // debug end 

      const ConvOpData *data = static_cast<const ConvOpData *>(node->user_data);
      const auto *params = reinterpret_cast<const TfLiteConvParams *>(node->builtin_data);

      const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
      const TfLiteEvalTensor* filter = tflite::micro::GetEvalInput(context, node, 1);

      // bias is optional
      const TfLiteEvalTensor* bias = (node->inputs->size == 3) ? tflite::micro::GetEvalInput(context, node, 2) : nullptr;
      
      TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);

      // get data pointers
      const int8_t* input_data = tflite::micro::GetTensorData<int8_t>(input);
      const int8_t* filter_data = tflite::micro::GetTensorData<int8_t>(filter);
      const int32_t *bias_data = (bias != nullptr) ? tflite::micro::GetTensorData<int32_t>(bias) : nullptr;
      int8_t *output_data = tflite::micro::GetTensorData<int8_t>(output);

      // get dimensions
      const int batches = input->dims->data[0];
      const int input_height = input->dims->data[1];
      const int input_width = input->dims->data[2];
      const int input_depth = input->dims->data[3];

      const int filter_height = filter->dims->data[1];
      const int filter_width = filter->dims->data[2];

      const int output_height = output->dims->data[1];
      const int output_width = output->dims->data[2];
      const int output_depth = output->dims->data[3]; // Conv2D output is dim[0]
#if DEBUG_PRINTS
      printf("DEBUG: output_depth is: %d", output_depth);
#endif

      const int stride_height = params->stride_height;
      const int stride_width = params->stride_width;
      const int pad_height = data->padding.height;
      const int pad_width = data->padding.width;

      static int conv_debug_once = 0;

      // core loop
      for (int b = 0; b < batches; ++b) {
        for (int out_y = 0; out_y < output_height; ++out_y) {
          for (int out_x = 0; out_x < output_width; ++out_x) {
            for (int out_channels = 0; out_channels < output_depth; ++out_channels) {

              // init accumulator
              int32_t acc = (bias_data) ? bias_data[out_channels] : 0;
              
              // try to avoid bias
              // acc = 0;
              acc = acc >> 4; // try to reduce bias effect

              // DEBUG PRINT
              bool debug_print = (b==0 && out_y==0 && out_x==0 && out_channels==0);
              if (debug_print)
              {
                // if (out_channels == 0) {
                //   acc = acc << 3;
                // }
#if DEBUG_PRINTS
                printf("DEBUG: Initial Acc(bias) = %ld\r\n", acc);
                printf("DEBUG: Input Offset = %ld\r\n", data->input_offset);
#endif
              }
              
              if (debug_print && conv_debug_once == 0) {
                // #region agent log
                DebugLogNDJSONPrintf(
                    "pre-fix",
                    "H5",
                    "custom_conv_int4.cpp:bias_shift",
                    "conv_eval_bias_shift",
                    "{\"bias_raw\":%ld,\"bias_shifted\":%ld}",
                    static_cast<long>(bias_data ? bias_data[out_channels] : 0),
                    static_cast<long>(acc));
                // #endregion

                // #region agent log
                DebugLogNDJSONPrintf(
                    "pre-fix",
                    "H2",
                    "custom_conv_int4.cpp:pack_probe",
                    "conv_eval_first8_weights",
                    "{\"w0\":%d,\"w1\":%d,\"w2\":%d,\"w3\":%d,\"w4\":%d,\"w5\":%d,\"w6\":%d,\"w7\":%d,"
                    "\"pb0\":%u,\"pb1\":%u,\"pb2\":%u,\"pb3\":%u}",
                    static_cast<int>(GET_INT4_WEIGHT(filter_data, 0)),
                    static_cast<int>(GET_INT4_WEIGHT(filter_data, 1)),
                    static_cast<int>(GET_INT4_WEIGHT(filter_data, 2)),
                    static_cast<int>(GET_INT4_WEIGHT(filter_data, 3)),
                    static_cast<int>(GET_INT4_WEIGHT(filter_data, 4)),
                    static_cast<int>(GET_INT4_WEIGHT(filter_data, 5)),
                    static_cast<int>(GET_INT4_WEIGHT(filter_data, 6)),
                    static_cast<int>(GET_INT4_WEIGHT(filter_data, 7)),
                    static_cast<unsigned int>(((const uint8_t*)filter_data)[0]),
                    static_cast<unsigned int>(((const uint8_t*)filter_data)[1]),
                    static_cast<unsigned int>(((const uint8_t*)filter_data)[2]),
                    static_cast<unsigned int>(((const uint8_t*)filter_data)[3]));
                // #endregion
              }


              const int in_y_origin = (out_y * stride_height) - pad_height;
              const int in_x_origin = (out_x * stride_width) - pad_width;

              // convolution loop
              for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                  for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                    const int in_y = in_y_origin + filter_y;
                    const int in_x = in_x_origin + filter_x;

                    // check bounds
                    if ((in_y >= 0) && (in_y < input_height) &&
                        (in_x >= 0) && (in_x < input_width)) {

                      const int input_index = 
                          b * (input_height * input_width * input_depth) +
                          in_y * (input_width * input_depth) +
                          in_x * (input_depth) +
                          in_channel;

                      const int filter_index = 
                          out_channels * (filter_height * filter_width * input_depth) +
                          filter_y * (filter_width * input_depth) +
                          filter_x * (input_depth) +
                          in_channel;

                      const int8_t input_val = input_data[input_index];
                      const int8_t filter_val = 
                          GET_INT4_WEIGHT(filter_data, filter_index);

                      if (debug_print && print_count == 0) {
                        const uint8_t packed_byte =
                            ((const uint8_t*)(filter_data))[filter_index / 2];
                        // #region agent log
                        DebugLogNDJSONPrintf(
                            "pre-fix",
                            "H2",
                            "custom_conv_int4.cpp:first_mac",
                            "conv_eval_first_mac",
                            "{\"input_val\":%d,\"input_offset\":%ld,"
                            "\"filter_index\":%d,\"packed_byte\":%u,"
                            "\"filter_val\":%d,\"bias\":%ld}",
                            static_cast<int>(input_val),
                            data->input_offset,
                            filter_index,
                            static_cast<unsigned int>(packed_byte),
                            static_cast<int>(filter_val),
                            static_cast<long>(acc));
                        // #endregion
                      }

                      // --- DEBUG LOOP ---
                      // Only print first 10 macs
                      
                      if (debug_print && print_count < 27) {
#if DEBUG_PRINTS
                          printf("  [%d] In=%d, Off=%ld, W_int4=%d, MAC+=%ld\r\n", 
                                print_count, 
                                input_val, 
                                data->input_offset, 
                                filter_val, 
                                (int32_t)(input_val + data->input_offset) * filter_val);
#endif
                          print_count++;
                      }

                      acc += (static_cast<int32_t>(input_val + data->input_offset) *
                              static_cast<int32_t>(filter_val));
                    }
                  }
                }

                if (debug_print)
                {
                  // printf("DEBUG: Before quantization Acc = %ld\r\n", acc);
                  // 【新增调试打印】只打印第一个 Batch, 第一个像素, 第一个通道
                  if (b == 0 && out_y == 0 && out_x == 0 && out_channels == 0) {
                    // 1. 打印累加器的值 (还没缩放前)
                    // 预期：Python算出来大概是正数 (因为 -69 - (-128) = 59)
                    // 如果这里是负数或者0，那就是 MAC 算错了
#if DEBUG_PRINTS
                    printf("DEBUG CHECK:\r\n");
                    printf("  Acc (Before Quant) = %ld\r\n", acc);
                    
                    // 2. 打印量化参数
                    // 如果 Multiplier 是 0，那就是 Scale 没算对
                    printf("  Multiplier = %ld\r\n", data->per_channel_output_multiplier[out_channels]);
                    printf("  Shift      = %d\r\n", data->per_channel_output_shift[out_channels]);
                    
                    // 3. 打印 Output Offset (应该约等于 -128)
                    printf("  Output Offset = %ld\r\n", data->output_offset);
#endif
                  }
                }
              }

              // Requantization
              acc = MultiplyByQuantizedMultiplier(
                  acc,
                  data->per_channel_output_multiplier[out_channels],
                  data->per_channel_output_shift[out_channels]
              );
              acc += data->output_offset;

              if (b == 0 && out_y == 0 && out_x == 0 && out_channels == 0) {
                // #region agent log
                DebugLogNDJSONPrintf(
                    "pre-fix",
                    "H4",
                    "custom_conv_int4.cpp:requant",
                    "conv_eval_requant_channel0",
                    "{\"acc_post_mult\":%ld,\"output_offset\":%ld,"
                    "\"out_act_min\":%ld,\"out_act_max\":%ld}",
                    static_cast<long>(acc),
                    data->output_offset,
                    data->output_activation_min,
                    data->output_activation_max);
                // #endregion
              }
              if (b == 0 && out_y == 0 && out_x == 0 && out_channels == 0) {
#if DEBUG_PRINTS
                  printf("  Final Result before clamping= %ld\r\n", acc);
#endif
              }
              // clamp
              acc = std::max(acc, data->output_activation_min);
              acc = std::min(acc, data->output_activation_max);

              if (b == 0 && out_y == 0 && out_x == 0 && out_channels == 0) {
#if DEBUG_PRINTS
                  printf("  Final Result = %ld\r\n", acc);
#endif
              }

              // store output
              int output_index = 
                  b * (output_height * output_width * output_depth) +
                  out_y * (output_width * output_depth) +
                  out_x * (output_depth) +
                  out_channels;
              output_data[output_index] = static_cast<int8_t>(acc);
            }
          }
        }
      }

      if (input_depth == 8 && output_depth == 16) {
#if DEBUG_PRINTS
        printf("\r\n[MCU DEBUG] Layer 3 (Conv2D) Output First 16 bytes:\r\n");
        for (int i = 0; i < 16; i++) {
            printf("%d, ", (int)output_data[i]);
        }
        printf("\r\n");
#endif
      }

      if (input_depth == 3 && conv_debug_once == 0) {
        conv_debug_once = 1;
      }

      // compare the output with python, only first layer
      if (input_depth == 3) {
#if DEBUG_PRINTS
        printf("\r\n[MCU DEBUG] Layer 1 (Conv2D) Output First 16 bytes:\r\n");
        for (int i = 0; i < 16; i++) {
            // force to int
            printf("%d, ", (int)output_data[i]);
        }
        printf("\r\n");
        
        // 打印完直接死循环卡住，方便我们看日志，防止被后面刷屏
        // 确认第一层对了，再把这行删掉
        // while(1) {}; 
#endif
      }

      return kTfLiteOk;
    }

  }   // namespace

  TFLMRegistration Register_CUSTOM_CONV_INT4() {
    return tflite::micro::RegisterOp(ConvInit_INT4, ConvPrepare_INT4, ConvEval_INT4);
  }

} // namespace tflite




