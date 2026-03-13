/*
 * custom_conv_int4_preunpack.h
 *
 * Custom INT4 Conv2D operator with pre-unpack optimization.
 * Unpacks INT4 weights to INT8 in a scratch buffer, then calls
 * arm_convolve_wrapper_s8() for SMLAD-accelerated inference.
 * Falls back to arm_convolve_wrapper_s4() for layers that exceed
 * the scratch buffer size.
 */

#ifndef CUSTOM_CONV_INT4_PREUNPACK_H_
#define CUSTOM_CONV_INT4_PREUNPACK_H_

#include "tensorflow/lite/micro/micro_common.h"

namespace tflite {
TFLMRegistration Register_CONV_2D_INT4_PREUNPACK();
}

#endif /* CUSTOM_CONV_INT4_PREUNPACK_H_ */
