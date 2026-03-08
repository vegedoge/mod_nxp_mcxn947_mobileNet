/*
 * Copyright 2021 NXP
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef _DEMO_CONFIG_H_
#define _DEMO_CONFIG_H_

// Model selection:
// 0 -> ResNet-20 (CONV_2D + ADD, no DepthwiseConv2D)
// 1 -> MobileNet-v1 (CONV_2D + DEPTHWISE_CONV_2D, no ADD)
#define MODEL_SELECT 0

#if MODEL_SELECT == 0
  #define EXAMPLE_NAME "CIFAR-10 ResNet-20"
#elif MODEL_SELECT == 1
  #define EXAMPLE_NAME "CIFAR-10 MobileNet-v1"
#endif

#define FRAMEWORK_NAME     "TensorFlow Lite Micro"
#define DETECTION_TRESHOLD 60
#define NUM_RESULTS        1
#define DEMO_VERBOSE       false
#define EOL                "\r\n"
#define DEBUG_PRINTS       0
#define ENABLE_BATCH_TEST  1
#define BATCH_TEST_PRINT_EVERY 1

// Model/kernel path switch:
// 0 -> INT8 baseline (model_data_int8.h + builtin Conv/Depthwise)
// 1 -> INT4 custom kernel (model_data.h + custom INT4 Conv/Depthwise)
// 2 -> INT4 CMSIS-NN (model_data_cmsis.h + Register_CONV_2D_INT4)
#define USE_INT4_CUSTOM_PATH 2

#endif // _DEMO_CONFIG_H_
