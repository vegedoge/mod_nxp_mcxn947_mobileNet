/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
   Copyright 2021-2023 NXP

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/* File modified by NXP. Changes are described in file
   /middleware/eiq/tensorflow-lite/readme.txt in section "Release notes" */

#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "fsl_debug_console.h"
#include "model.h"
#include "demo_config.h"
#include "output_postproc.h"
#include "timer.h"

// Model data selection: MODEL_SELECT x USE_INT4_CUSTOM_PATH
#if MODEL_SELECT == 0  // ResNet-20
  #if USE_INT4_CUSTOM_PATH == 4 || USE_INT4_CUSTOM_PATH == 2
    #include "model_data_cmsis.h"
  #elif USE_INT4_CUSTOM_PATH == 1
    #include "model_data.h"
  #else
    #include "model_data_int8.h"
  #endif
#elif MODEL_SELECT == 1  // MobileNet-v1
  #if USE_INT4_CUSTOM_PATH == 3
    #include "model_data_mobilenet_hybrid.h"
  #elif USE_INT4_CUSTOM_PATH == 4 || USE_INT4_CUSTOM_PATH == 2
    #include "model_data_mobilenet_cmsis.h"
  #elif USE_INT4_CUSTOM_PATH == 1
    #include "model_data_mobilenet.h"
  #else
    #include "model_data_mobilenet_int8.h"
  #endif
#endif

// post-processed input
// #include "image_data.h"
#include "image_data_direct.h"

#if ENABLE_BATCH_TEST
#include "test_batch_data.h"
#endif


// modules for profiling
// the original tflm profiler can not stick to the NXP timer functions
// so we implement our own profiler based on tflm profiler interface
#include "custom_tflm_profiler.h"

// static tflite::MicroProfiler s_profiler;
static CustomProfiler s_custom_profiler;

static const tflite::Model* s_model = nullptr;
static tflite::MicroInterpreter* s_interpreter = nullptr;

extern tflite::MicroOpResolver &MODEL_GetOpsResolver();

// Forward declaration for helper defined later in this file.
uint8_t* GetTensorData(TfLiteTensor* tensor, tensor_dims_t* dims, tensor_type_t* type);

// An area of memory to use for input, output, and intermediate arrays.
// (Can be adjusted based on the model needs.)
#ifdef TENSORARENA_NONCACHE
static uint8_t s_tensorArena[kTensorArenaSize] __ALIGNED(16) __attribute__((section("NonCacheable")));
#else
static uint8_t s_tensorArena[kTensorArenaSize] __ALIGNED(16);
#endif

static uint32_t s_tensorArenaSizeUsed = 0;
status_t MODEL_Init(void)
{
    TFLM_Timer_Init();
    // Map the model into a usable data structure. This doesn't involve any
    // copying or parsing, it's a very lightweight operation.
    s_model = tflite::GetModel(model_data);
    if (s_model->version() != TFLITE_SCHEMA_VERSION)
    {
        PRINTF("Model provided is schema version %d not equal "
               "to supported version %d!\r\n",
               s_model->version(), TFLITE_SCHEMA_VERSION);
        return kStatus_Fail;
    }

    // Pull in only the operation implementations we need.
    // This relies on a complete list of all the ops needed by this graph.
    // NOLINTNEXTLINE(runtime-global-variables)
    tflite::MicroOpResolver &micro_op_resolver = MODEL_GetOpsResolver();

    // Build an interpreter to run the model with.
    // static tflite::MicroInterpreter static_interpreter(
    //     s_model, micro_op_resolver, s_tensorArena, kTensorArenaSize);

    // here we added out own profiler
    static tflite::MicroInterpreter static_interpreter(
        s_model, micro_op_resolver, s_tensorArena, kTensorArenaSize, nullptr, &s_custom_profiler);
    s_interpreter = &static_interpreter;
 
    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = s_interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk)
    {
        PRINTF("AllocateTensors() failed!\r\n");
        return kStatus_Fail;
    }

    s_tensorArenaSizeUsed = s_interpreter->arena_used_bytes();
#if (defined(CPU_MIMXRT798SGAWAR_hifi4) || defined(CPU_MIMXRT798SGFOA_hifi4))
    PRINTF("Hifi4 DSP Frequency: %d MHz\r\n", CLOCK_GetFreq(kCLOCK_Hifi4CpuClk)/1000000);
#elif  (defined(CPU_MIMXRT798SGAWAR_hifi1) || defined(CPU_MIMXRT798SGFOA_hifi1))
    PRINTF("Hifi1 DSP Frequency: %d MHz\r\n", CLOCK_GetFreq(kCLOCK_Hifi1CpuClk)/1000000);
#elif (defined(CPU_MIMXRT685SFAWBR_dsp) || defined(CPU_MIMXRT685SFFOB_dsp) || defined(CPU_MIMXRT685SFVKB_dsp) || defined(CPU_MIMXRT685SVFVKB_dsp) ||(defined(CPU_MIMXRT595SFAWC_dsp) || defined(CPU_MIMXRT595SFFOC_dsp)))

    PRINTF("DSP Frequency: %d MHz\r\n", CLOCK_GetFreq(kCLOCK_DspCpuClk)/1000000);
#else 
    PRINTF("Core/NPU Frequency: %d MHz\r\n", CLOCK_GetFreq(kCLOCK_CoreSysClk)/1000000);
#endif
    PRINTF("TensorArena Addr: 0x%x - 0x%x\r\n", s_tensorArena, s_tensorArena + kTensorArenaSize);
    PRINTF("TensorArena Size: Total 0x%x (%d B); Used 0x%x (%d B)\r\n" , kTensorArenaSize, kTensorArenaSize, s_tensorArenaSizeUsed, s_tensorArenaSizeUsed);
    PRINTF("Model Addr: 0x%x - 0x%x\r\n" , model_data, model_data + sizeof(model_data));
    PRINTF("Model Size: 0x%x (%d B)\r\n" , sizeof(model_data), sizeof(model_data));
    PRINTF("Total Size Used: %d B (Model (%d B) + TensorArena (%d B))\r\n" , (sizeof(model_data) + s_tensorArenaSizeUsed), sizeof(model_data), s_tensorArenaSizeUsed);

    return kStatus_Success;
}

status_t MODEL_RunInference(void)
{
    // test for all 1s
    TfLiteTensor* input_tensor = s_interpreter->input(0);
    int8_t* input_data = tflite::GetTensorData<int8_t>(input_tensor);

    // size_t input_bytes = input_tensor->bytes;
    // memset(input_data, 1, input_bytes);

    // PRINTF("DEBUG CHECK: Input[0-4]: %d, %d, %d, %d, %d\r\n", 
    //        input_data[0], input_data[1], input_data[2], input_data[3], input_data[4]);

    memcpy(input_data, image_data_direct, input_tensor->bytes);
    // memcpy(input_data, image_data, input_tensor->bytes);

#if DEBUG_PRINTS
    // To compare input with python
    PRINTF("MCU INPUT TENSOR (First 16 bytes):\r\n");
    for (int i = 0; i < 16; i++) {
        PRINTF("%d, ", (int)input_data[i]);
    }
    PRINTF("\r\n");
#endif

    if (s_interpreter->Invoke() != kTfLiteOk)
    {
        PRINTF("Invoke failed!\r\n");
        return kStatus_Fail;
    }

#if DEBUG_PRINTS
    PRINTF("--- Operator Profiling Results ---\r\n");
    s_custom_profiler.LogResults();
    PRINTF("--- Profiling Ends ---\r\n");
#endif

    return kStatus_Success;
}

status_t MODEL_RunBatchTest(void)
{
#if ENABLE_BATCH_TEST
    TfLiteTensor* input_tensor = s_interpreter->input(0);
    TfLiteTensor* output_tensor = s_interpreter->output(0);

    if (input_tensor->type != kTfLiteInt8) {
        PRINTF("BatchTest: input type is not INT8\r\n");
        return kStatus_Fail;
    }

    const size_t input_bytes = input_tensor->bytes;

    int8_t* input_data = tflite::GetTensorData<int8_t>(input_tensor);
    int8_t* output_data = tflite::GetTensorData<int8_t>(output_tensor);

    tensor_dims_t output_dims;
    tensor_type_t output_type;
    GetTensorData(output_tensor, &output_dims, &output_type);

    const int total = TEST_BATCH_SIZE;
    const int8_t* inputs = g_test_batch_inputs;

#if DEBUG_PRINTS
    PRINTF("BatchTest: running %d samples\r\n", total);
#endif

    for (int i = 0; i < total; ++i) {
        const int8_t* src = inputs + (i * input_bytes);
        memcpy(input_data, src, input_bytes);

        auto start_time = TIMER_GetTimeInUS();
        if (s_interpreter->Invoke() != kTfLiteOk) {
            PRINTF("BatchTest: invoke failed on sample %d\r\n", i);
            return kStatus_Fail;
        }
        auto end_time = TIMER_GetTimeInUS();
        int inference_time = static_cast<int>(end_time - start_time);

        if (BATCH_TEST_PRINT_EVERY > 0 && (i % BATCH_TEST_PRINT_EVERY) == 0) {
#if DEBUG_PRINTS
            PRINTF("BatchTest[%d]\r\n", i);
#endif
            MODEL_ProcessOutput(reinterpret_cast<const uint8_t*>(output_data),
                                &output_dims, output_type, inference_time);
        }

#if DEBUG_PRINTS
        // Log per-layer profiling after sample 5 (serial is connected by then)
        if (i == 5) {
            PRINTF("--- Operator Profiling Results ---\r\n");
            s_custom_profiler.LogResults();
            PRINTF("--- Profiling Ends ---\r\n");
        }
#endif

    }

    return kStatus_Success;
#else
    PRINTF("BatchTest: disabled (ENABLE_BATCH_TEST=0)\r\n");
    return kStatus_Fail;
#endif
}

status_t MODEL_GetOutputQuantParams(float* scale, int* zeroPoint)
{
    if (scale == nullptr || zeroPoint == nullptr || s_interpreter == nullptr)
    {
        return kStatus_Fail;
    }

    TfLiteTensor* outputTensor = s_interpreter->output(0);
    if (outputTensor == nullptr)
    {
        return kStatus_Fail;
    }

    *scale = outputTensor->params.scale;
    *zeroPoint = outputTensor->params.zero_point;
    return kStatus_Success;
}

uint8_t* GetTensorData(TfLiteTensor* tensor, tensor_dims_t* dims, tensor_type_t* type)
{
    switch (tensor->type)
    {
        case kTfLiteFloat32:
            *type = kTensorType_FLOAT32;
            break;
        case kTfLiteUInt8:
            *type = kTensorType_UINT8;
            break;
        case kTfLiteInt8:
            *type = kTensorType_INT8;
            break;
        default:
            assert("Unknown input tensor data type!\r\n");
    };

    dims->size = tensor->dims->size;
    assert(dims->size <= MAX_TENSOR_DIMS);
    for (int i = 0; i < tensor->dims->size; i++)
    {
        dims->data[i] = tensor->dims->data[i];
    }

    return tensor->data.uint8;
}

uint8_t* MODEL_GetInputTensorData(tensor_dims_t* dims, tensor_type_t* type)
{
    TfLiteTensor* inputTensor = s_interpreter->input(0);

    return GetTensorData(inputTensor, dims, type);
}

uint8_t* MODEL_GetOutputTensorData(tensor_dims_t* dims, tensor_type_t* type)
{
    TfLiteTensor* outputTensor = s_interpreter->output(0);

    return GetTensorData(outputTensor, dims, type);
}

// Convert unsigned 8-bit image data to model input format in-place.
void MODEL_ConvertInput(uint8_t* data, tensor_dims_t* dims, tensor_type_t type)
{
    int size = dims->data[2] * dims->data[1] * dims->data[3];
    switch (type)
    {
        case kTensorType_UINT8:
            break;
        case kTensorType_INT8:
            for (int i = size - 1; i >= 0; i--)
            {
                reinterpret_cast<int8_t*>(data)[i] =
                    static_cast<int>(data[i]) - 128;
            }
            break;
        case kTensorType_FLOAT32:
            for (int i = size - 1; i >= 0; i--)
            {
                reinterpret_cast<float*>(data)[i] =
                    (static_cast<int>(data[i]) - MODEL_INPUT_MEAN) / MODEL_INPUT_STD;
            }
            break;
        default:
            assert("Unknown input tensor data type!\r\n");
    }
}

const char* MODEL_GetModelName(void)
{
    return MODEL_NAME;
}
