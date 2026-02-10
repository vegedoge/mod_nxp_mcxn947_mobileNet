/*
 * Copyright 2021 NXP
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef _DEMO_CONFIG_H_
#define _DEMO_CONFIG_H_

#define EXAMPLE_NAME       "CIFAR-10"
#define FRAMEWORK_NAME     "TensorFlow Lite Micro"
#define DETECTION_TRESHOLD 60
#define NUM_RESULTS        1
#define DEMO_VERBOSE       false
#define EOL                "\r\n"
#define DEBUG_PRINTS       0

// Enable MCU batch test with generated test_batch_data.h
#define ENABLE_BATCH_TEST  0
#define BATCH_TEST_PRINT_EVERY 1

// Model/kernel path switch:
// 0 -> INT8 baseline compare path (model_data_int8.h + builtin Conv/Depthwise)
// 1 -> INT4 custom path (model_data.h + custom INT4 Conv/Depthwise)
#define USE_INT4_CUSTOM_PATH 1

#endif // _DEMO_CONFIG_H_
