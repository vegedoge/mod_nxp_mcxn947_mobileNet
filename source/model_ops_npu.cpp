/*
 * Copyright 2022 NXP
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/kernels/neutron/neutron.h"

// Header files for MobileNetV1 model ops
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/kernels/pooling.h"
#include "tensorflow/lite/micro/kernels/softmax.h"
#include "tensorflow/lite/micro/kernels/reshape.h"
#include "tensorflow/lite/micro/kernels/depthwise_conv.h"
#include "demo_config.h"

// Custom INT4 ops
#include "custom_conv_int4.h"
#include "custom_depthwise_conv_int4.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

tflite::MicroOpResolver &MODEL_GetOpsResolver()
{
    static tflite::MicroMutableOpResolver<12> s_microOpResolver;

    // Match kernel path with selected model variant.
#if USE_INT4_CUSTOM_PATH
    s_microOpResolver.AddConv2D(tflite::Register_CUSTOM_CONV_INT4());
    s_microOpResolver.AddDepthwiseConv2D(tflite::Register_CUSTOM_DEPTHWISE_CONV_INT4());
#else
    s_microOpResolver.AddConv2D();
    s_microOpResolver.AddDepthwiseConv2D();
#endif

    // Add MobileNetV1 Ops
    // s_microOpResolver.AddConv2D();
    // s_microOpResolver.AddDepthwiseConv2D();
    s_microOpResolver.AddAveragePool2D();
    s_microOpResolver.AddFullyConnected();
    s_microOpResolver.AddMean();        // in micro_ops.h

    // old Ops for cifarNet
    s_microOpResolver.AddReshape();
    s_microOpResolver.AddSlice();
    s_microOpResolver.AddSoftmax();
    s_microOpResolver.AddDequantize();
    s_microOpResolver.AddCustom(tflite::GetString_NEUTRON_GRAPH(),
        tflite::Register_NEUTRON_GRAPH());

    return s_microOpResolver;
}
