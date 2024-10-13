#include "timer.h"
#include <stdio.h>
#include <sys/time.h>

struct time_data time_data_arr[3];

struct timeval start_timer() {
    struct timeval begin;
    gettimeofday(&begin, 0);
    return begin;
}

void stop_timer(struct time_data* time_data, enum time_type TIME_TYPE, const char* str) {
    struct timeval end;
    gettimeofday(&end, 0);
    long seconds = end.tv_sec - time_data[TIME_TYPE].start.tv_sec;
    long microseconds = end.tv_usec - time_data[TIME_TYPE].start.tv_usec;
    printf("\t[%s]\n", str);
    printf("\tTime in s [%lf] \n\n", seconds + microseconds * 1e-6);
    time_data[TIME_TYPE].stop = seconds + microseconds * 1e-6;
}

double count_percentage(double stop_timer_KERNEL_TRANS, double stop_timer_KERNEL) {
    double transfer_time = stop_timer_KERNEL_TRANS - stop_timer_KERNEL;
    return (transfer_time / stop_timer_KERNEL_TRANS) * 100;
}
