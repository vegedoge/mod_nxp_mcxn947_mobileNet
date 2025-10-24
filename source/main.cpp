/*
 * Copyright 2020-2022 NXP
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "board_init.h"
#include "demo_config.h"
#include "demo_info.h"
#include "fsl_debug_console.h"
#include "image.h"
#include "image_utils.h"
#include "model.h"
#include "output_postproc.h"
#include "timer.h"

int main(void)
{
    BOARD_Init();
    TIMER_Init();

    DEMO_PrintInfo();

    if (MODEL_Init() != kStatus_Success)
    {
        PRINTF("Failed initializing model" EOL);
        for (;;) {}
    }

    tensor_dims_t inputDims;
    tensor_type_t inputType;
    uint8_t* inputData = MODEL_GetInputTensorData(&inputDims, &inputType);

    tensor_dims_t outputDims;
    tensor_type_t outputType;
    uint8_t* outputData = MODEL_GetOutputTensorData(&outputDims, &outputType);

    // ============ Print model info ============
    PRINTF("--- TFLite model info ---\r\n");
    PRINTF("Input B: %d, H: %d, W: %d, C: %d\n", 
        inputDims.data[0], inputDims.data[1], inputDims.data[2], inputDims.data[3]);
    PRINTF("Input type: %d (0=kTfLiteFloat32, 1=kTfLiteUint8, 2=kTfLiteInt8)\n", inputType); // 9 is kTfLiteInt8

    PRINTF("Output B: %d, Classes: %d\n", 
        outputDims.data[0], outputDims.data[1]);
    PRINTF("Output type: %d (0=kTfLiteFloat32, 1=kTfLiteUint8, 2=kTfLiteInt8)\n", outputType);
    PRINTF("--------------------------\r\n");
    // ========================================

    while (1)
    {
        /* Expected tensor dimensions: [batches, height, width, channels] */
        if (IMAGE_GetImage(inputData, inputDims.data[2], inputDims.data[1], inputDims.data[3]) != kStatus_Success)
        {
            PRINTF("Failed retrieving input image" EOL);
            for (;;) {}
        }

        MODEL_ConvertInput(inputData, &inputDims, inputType);

        auto startTime = TIMER_GetTimeInUS();
        MODEL_RunInference();
        auto endTime = TIMER_GetTimeInUS();

        MODEL_ProcessOutput(outputData, &outputDims, outputType, endTime - startTime);
    }
}
