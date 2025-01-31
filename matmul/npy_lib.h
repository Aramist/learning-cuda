#ifdef NPY_LIB_IMPL
#undef NPY_LIB_IMPL

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define SAVE_NPY_SUCCESS 1
#define SAVE_NPY_FAILURE 0

char *make_float_header(size_t shape[], int ndim) {
  /* Creates a string containing a Python dict literal with the
   * following keys:
   * descr (str): "float32"
   * fortran_order (bool): False
   * shape (tuple[int...]): shape of the array
   *
   * It is terminated by a newline
   */

  char buffer[4096] = {0};
  strcat(
      buffer,
      "{'descr': 'f4', 'fortran_order': False, 'shape': ("
  );

  // Construct the shape string
  for(int idx = 0; idx < ndim; idx++) {
    char *digits;
    asprintf(&digits, "%lu", shape[idx]);
    strcat(buffer, (const char *) digits);
    if (idx < ndim - 1) {
      // Add dim size with comma
      strcat(buffer, ", ");
    }
    free(digits);
  }

  strcat(buffer, ")}");
  
  return strdup(buffer);  // free this later
}

int save_float_arr(const char *filename, float *arr, size_t shape[], size_t ndim) {
  char *header;
  size_t header_len, num_align, n_elem;
  FILE *fp = fopen(filename, "w");
  if (fp == NULL) {
    fprintf(stderr, "Failed to create file %s", filename);
    return SAVE_NPY_FAILURE;
  }

  // Compute array size
  n_elem = 1;
  for(int i=0; i<ndim; i++){
    n_elem *= shape[i];
  }
  
  // Write magic string: \x93NUMPY
  fputs("\x93NUMPY", fp);
  // Put version bytes 2 and 0
  fputc(2, fp);
  fputc(0, fp);

  // Compute header so we can get its length
  header = make_float_header(shape, ndim);
  header_len = strlen(header);
  
  // 6 + 2 + 4 + header + align should divide 64
  num_align = ((header_len + 6 + 2 + 4) % 64) ? 64 - ((header_len + 6 + 2 + 4) % 64) : 0;
  header_len += num_align;

  // Write the header length
  for(int i = 0; i < 4; i++) {
    // Get the lsB
    uint8_t lsb = (header_len >> (i * 8)) & 0xFF;
    // Write to file
    fputc(lsb, fp);
  }


  // Write the header
  fputs(header, fp);
  // Write the spaces for alignment purposes
  for(int i = 0; i < num_align - 1; i++)
    fputc(0x20, fp);
  // Apparently the newline comes after the spaces
  fputc('\n', fp);

  // Now write the data
  fwrite((void *)arr, sizeof(float), n_elem, fp);

  free(header);
  fclose(fp);
  return SAVE_NPY_SUCCESS;
}

#endif
