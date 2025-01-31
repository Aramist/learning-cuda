#ifndef RAND_LIB_H
#define RAND_LIB_H
#include <limits.h>
#include <stdlib.h>
#include <time.h>

void setup_rand();
float rand_float();

void rand_matrix(float *data, size_t m, size_t n);

#endif
