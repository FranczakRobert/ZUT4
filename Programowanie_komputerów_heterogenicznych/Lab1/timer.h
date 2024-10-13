#ifndef TIMER_LIB
#define TIMER_LIB
#include <sys/time.h>

enum time_type {
    CPU_TIME = 0,
    KERNEL_TIME,
    KERNEL_TRANSFER_TIME
};

struct time_data {
    struct timeval start;
    double stop;
};

struct timeval start_timer();
void stop_timer(struct time_data* time_data,enum time_type TIME ,const char* str);
double count_percentage(double stop_timer_KERNEL_TRANS, double stop_timer_KERNEL);

#endif // TIMER_LIB