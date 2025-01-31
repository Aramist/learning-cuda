#include <stdlib.h>
#include <ctype.h>
#define NPY_LIB_IMPL
#include "npy_lib.h"


int main(void){
  const char *filename = "test.npy";
  float data[8] = {1.0, 2., 3., 4., 5., 6., 7., 8.};
  size_t shape[2] = {2, 4};

  save_float_arr(filename, data, shape, 2);
}
