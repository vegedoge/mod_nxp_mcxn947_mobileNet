/*
 * custom_int4_unpack.h
 *
 *  Created on: 2025.11.18
 *      Author: yx_wu
 */

#ifndef CUSTOM_INT4_UNPACK_H_
#define CUSTOM_INT4_UNPACK_H_

#include <cstdint>

// Macro to get INT4 weights
// We ragard TFLM INT8 weights as UINT8, and switch back to INT8
// small-endian on Cortex-M33
#define GET_INT4_WEIGHT(packed_weights, channel_index) \
  ({ \
    /* get the INT8 byte index containing the INT4 weight*/ \
    const int byte_index = (channel_index) / 2; \
    /* get original unit8 view */ \
    const uint8_t packed_byte = ((const uint8_t*)(packed_weights))[byte_index]; \
    /* high or low */ \
    int8_t weight; \
    if ((channel_index) % 2 == 0) { \
      /* low 4 bits, 0, 2, 4 -> lower*/ \
      weight = (packed_byte & 0x0F); \
    } else { \
      /* high 4 bits, 1, 3, 5 -> higher */ \
      weight = ((packed_byte >> 4) & 0x0F); \
    } \
    /* sign extend to INT8 */ \
    /* e.g.: 0b1111 (15), -1 in INT4, still -1 in INT8 */ \
    /* if  the sign bit is 1, fill in 1s to the left(complementary code) */ \
    if (weight & 0x08) { \
      weight |= 0xF0; \
    } \
    weight; \
})

#endif /* CUSTOM_INT4_UNPACK_H_ */
