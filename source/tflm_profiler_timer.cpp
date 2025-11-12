/*
 * tflm_profiler_timer.cpp
 *
 * Description: This file connects the TFLite Micro profiler's clock functions
 * to the NXP demo's hardware TIMER_GetTimeInUS() function.
 * 
 * Created on: 2025年11月10日
 * Author: yx_wu
 */

#include "tensorflow/lite/micro/micro_time.h"
#include "timer.h"

// TFLM profiler weakly link to this function.
// So we provide our own strong implementation here,
// to let the compiler select it.

extern "C" void TFLM_Timer_Init() {
  // to fake the compiler
}

namespace tflite {
  namespace micro {
    uint32_t GetCurrentTimeTicks() {
      // NXP hardware timer
      return static_cast<uint32_t>(TIMER_GetTimeInUS());
    }

    uint32_t ticks_per_second() {
      return 1000000;  // TIMER_GetTimeInUS() returns microseconds
    }

  }
}





