/*
 * custom_tflm_profiler.h
 *
 *  Created on: 2025年11月15日
 *      Author: yx_wu
 */

#ifndef CUSTOM_TFLM_PROFILER_H_
#define CUSTOM_TFLM_PROFILER_H_

#include "fsl_debug_console.h"
#include "tensorflow/lite/micro/micro_time.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/micro_profiler_interface.h"
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

// max operators for profiling
#define MAX_PROFILER_EVENTS 64
class CustomProfiler : public tflite::MicroProfiler {
public:
    CustomProfiler() : event_count_(0) {}

    // TFLM will call this beginEvent
    virtual uint32_t BeginEvent(const char* tag) override {
        if (event_count_ >= MAX_PROFILER_EVENTS) {
            return 0;  // No more space for events
        }

        uint32_t event_handle = event_count_;

        // record time and tag
        events_[event_count_].tag = tag;
        events_[event_count_].start_ticks = tflite::micro::GetCurrentTimeTicks();

        event_count_++;

        return event_handle;
    }

    virtual void EndEvent(uint32_t event_handle) {
        if (event_handle >= MAX_PROFILER_EVENTS) {
            return; // invalid handle
        }

        // record end time
        events_[event_handle].end_ticks = tflite::micro::GetCurrentTimeTicks();
    }

    void LogResults() {
        PRINTF("--- Custom Profiling Results ---\r\n");
        uint32_t total_ticks = 0;
        for (int i = 0; i < event_count_; ++i) {
            if (events_[i].end_ticks == 0) {
                events_[i].end_ticks = events_[i].start_ticks;  // if end not set, set to start
            }

            uint32_t duration = events_[i].end_ticks - events_[i].start_ticks;
            total_ticks += duration;
            PRINTF("%s took %u ticks (%u ms)\r\n", 
                   events_[i].tag ? events_[i].tag : "Unknown", // check for null tag
                   duration, 
                   duration / 1000); // 1000 us = 1 ms
        }
        PRINTF("Total ticks: %u (%u ms)\r\n", total_ticks, total_ticks / 1000);
        PRINTF("--- Custom Profiling Ends ---\r\n");

        event_count_ = 0;  // reset for next profiling session
    }

private:
    struct ProfileEvent {
        const char* tag = nullptr;
        uint32_t start_ticks = 0;
        uint32_t end_ticks = 0;
    };

    ProfileEvent events_[MAX_PROFILER_EVENTS];
    int event_count_ = 0;

    TF_LITE_REMOVE_VIRTUAL_DELETE
};



#endif /* CUSTOM_TFLM_PROFILER_H_ */
