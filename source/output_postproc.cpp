/*
 * Copyright 2020-2022 NXP
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "fsl_debug_console.h"
#include "output_postproc.h"
#include "get_top_n.h"
#include "demo_config.h"
#include "labels.h"
#include <math.h>
#ifdef EIQ_GUI_PRINTF
#include "chgui.h"
#endif

status_t MODEL_ProcessOutput(const uint8_t* data, const tensor_dims_t* dims,
                             tensor_type_t type, int inferenceTime)
{
    const float threshold = (float)DETECTION_TRESHOLD / 100;
    const char* label = "No label detected";
    float confidence = 0.0f;
    int top_index = -1;

    // For INT8 outputs, align confidence with Python baseline:
    // 1) dequantize via output tensor scale/zero_point
    // 2) run softmax on dequantized logits
    if (type == kTensorType_INT8) {
        const int num_classes = dims->data[dims->size - 1];
        const int8_t* raw_logits = reinterpret_cast<const int8_t*>(data);
        float out_scale = 1.0f;
        int out_zero_point = 0;
        if (MODEL_GetOutputQuantParams(&out_scale, &out_zero_point) != kStatus_Success) {
            PRINTF("Failed to get output quant params, fallback to GetTopN" EOL);
        } else {
            float max_logit = -1e30f;
            for (int i = 0; i < num_classes; ++i) {
                const float deq = (static_cast<float>(raw_logits[i]) - static_cast<float>(out_zero_point)) * out_scale;
                if (deq > max_logit) {
                    max_logit = deq;
                    top_index = i;
                }
            }

            float sum_exp = 0.0f;
            for (int i = 0; i < num_classes; ++i) {
                const float deq = (static_cast<float>(raw_logits[i]) - static_cast<float>(out_zero_point)) * out_scale;
                sum_exp += expf(deq - max_logit);
            }

            if (sum_exp > 0.0f && top_index >= 0) {
                const float top_deq =
                    (static_cast<float>(raw_logits[top_index]) - static_cast<float>(out_zero_point)) * out_scale;
                confidence = expf(top_deq - max_logit) / sum_exp;
                if (confidence > threshold) {
                    label = labels[top_index];
                }
            }
        }
    } else {
        result_t topResults[NUM_RESULTS];
        MODEL_GetTopN(data, dims->data[dims->size - 1], type, NUM_RESULTS, threshold, topResults);
        if (topResults[0].index >= 0) {
            auto result = topResults[0];
            confidence = result.score;
            top_index = result.index;
            if (confidence > threshold) {
                label = labels[top_index];
            }
        }
    }

    int score = (int)(confidence * 100);
    PRINTF("----------------------------------------" EOL);
    PRINTF("     Inference time: %d us" EOL, inferenceTime);
    PRINTF("     Detected: %s (%d%%)" EOL, label, score);
    if (top_index >= 0) {
        PRINTF("     Top-1 index: %d" EOL, top_index);
    }
    PRINTF("----------------------------------------" EOL);

    // Raw output to compare
    const int8_t* raw_logits = reinterpret_cast<const int8_t*>(data);
    int num_classes = dims->data[dims->size - 1];

    PRINTF("Raw model output logits:" EOL);
    for (int i = 0; i < num_classes; i++)
    {
        int val = static_cast<int>(raw_logits[i]);
        
        // fix negative values
        if (val < 0)
        {
            PRINTF("Class %2d: -%d" EOL, i, -val);
        }
        else
        {
            PRINTF("Class %2d:  %d" EOL, i, val);
        }
    }
    PRINTF("----------------------------------------" EOL);

#ifdef EIQ_GUI_PRINTF
    GUI_PrintfToBuffer(GUI_X_POS, GUI_Y_POS, "Detected: %.20s (%d%%)", label, score);
#endif

    return kStatus_Success;
}
