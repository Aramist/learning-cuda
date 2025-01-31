#include "rand_lib.h"

void setup_rand(){
  srand(time(NULL));
}

float rand_float(){
  return (float)rand() / (float)RAND_MAX;
}


void rand_matrix(float *data, size_t m, size_t n) {
  for(int flat_idx = 0; flat_idx < m * n; flat_idx++)
    data[flat_idx] = rand_float();
}
