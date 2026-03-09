/*
 * Copyright 2022 NXP
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/kernels/neutron/neutron.h"

// Header files for model ops
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/micro/kernels/depthwise_conv.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/kernels/softmax.h"
#include "tensorflow/lite/micro/kernels/reshape.h"
#include "demo_config.h"

// Custom INT4 ops
#include "custom_conv_int4.h"
#include "custom_depthwise_conv_int4.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

tflite::MicroOpResolver &MODEL_GetOpsResolver()
{
    static tflite::MicroMutableOpResolver<10> s_microOpResolver;

    // Conv2D: both models need this
#if USE_INT4_CUSTOM_PATH == 2 || USE_INT4_CUSTOM_PATH == 3
    s_microOpResolver.AddConv2D(tflite::Register_CONV_2D_INT4());
#elif USE_INT4_CUSTOM_PATH == 1
    s_microOpResolver.AddConv2D(tflite::Register_CUSTOM_CONV_INT4());
#else
    s_microOpResolver.AddConv2D();
#endif

#if MODEL_SELECT == 0  // ResNet-20: residual connections
    s_microOpResolver.AddAdd();
#elif MODEL_SELECT == 1  // MobileNet-v1: depthwise separable convs
  #if USE_INT4_CUSTOM_PATH == 3
    // Full CMSIS-NN INT4: both Conv2D and DepthwiseConv2D use CMSIS-NN INT4 kernels
    s_microOpResolver.AddDepthwiseConv2D(tflite::Register_DEPTHWISE_CONV_2D_INT4());
  #elif USE_INT4_CUSTOM_PATH == 2
    // CMSIS-NN INT4 Conv2D + builtin INT8 DepthwiseConv2D (mixed precision)
    s_microOpResolver.AddDepthwiseConv2D();
  #elif USE_INT4_CUSTOM_PATH == 1
    s_microOpResolver.AddDepthwiseConv2D(tflite::Register_CUSTOM_DEPTHWISE_CONV_INT4());
  #else
    s_microOpResolver.AddDepthwiseConv2D();
  #endif
#endif

    // Shared ops
    s_microOpResolver.AddFullyConnected();
    s_microOpResolver.AddMean();             // GlobalAveragePooling
    s_microOpResolver.AddReshape();
    s_microOpResolver.AddSoftmax();
    s_microOpResolver.AddDequantize();
    s_microOpResolver.AddCustom(tflite::GetString_NEUTRON_GRAPH(),
        tflite::Register_NEUTRON_GRAPH());

    return s_microOpResolver;
}
