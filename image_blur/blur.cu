#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_HDR
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define CUDA_CHKERR(err) if (err != cudaSuccess) { \
    printf("%s in %s at %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    exit(EXIT_FAILURE); \
  }


void blurImg(float *image_h, size_t width, size_t height, size_t channels, size_t ksize);
__global__ void kernel_blurImg(const float* M_in, float *M_out, size_t width, size_t height, size_t channels, size_t ksize);
const char *getSavePath(const char *filename);


int main(int argc, char *argv[]) {
  char *filename;
  const char *savePath;
  int imgHeight, imgWidth, numChannels, kernelSize;
  stbi_uc *imgBytes;
  float *imgFloat;

  if (argc < 3) {
    printf("Usage: ./blur img_path kernel_size\n");
    return EXIT_FAILURE;
  }

  filename = argv[1];
  if (sscanf(argv[2], "%d", &kernelSize) != 1) {
    fprintf(stderr, "Invalid kernel_size argument. Usage: ./blur img_path kernel_size\n");
    return EXIT_FAILURE;
  }

  if (kernelSize < 0 || kernelSize % 2 == 0) {
    fprintf(stderr, "Invalid kernel size. Must be an odd, positive number.\n");
    return EXIT_FAILURE;
  }

  // Attempt to read the image
  imgBytes = stbi_load((const char *)filename, &imgWidth, &imgHeight, &numChannels, 0);
  if (imgBytes == NULL) {
    fprintf(stderr, "Failed to load image!\n");
    return EXIT_FAILURE;
  }

  fprintf(stdout, "Img stats: %dx%d, %d channels.\n", imgWidth, imgHeight, numChannels);

  // Convert to float buffer
  imgFloat = (float *) malloc(numChannels * imgWidth * imgHeight * sizeof(float));
  // The original data order is HWC, colors stored as RGB
  //for (int i = 0; i < imgHeight; i++) {
    //for (int j = 0; j < imgWidth; j++) {
      //for (int c = 0; c < numChannels; c++){
        //int idx = i * (imgWidth * numChannels) + j * (numChannels) + c;
        //imgFloat[idx] = imgBytes[idx] / (float)255;
      //} } }
  for (int idx = 0; idx < imgHeight * imgWidth * numChannels; idx++){
    imgFloat[idx] = imgBytes[idx] / (float)255;
  }

  // Now actually do the blur
  blurImg(imgFloat, imgWidth, imgHeight, numChannels, kernelSize);

  // Convert back to ubyte
  for (int idx = 0; idx < imgHeight * imgWidth * numChannels; idx++){
    imgBytes[idx] = (stbi_uc)(imgFloat[idx] * 255);
  }
  // Don't need the float img anymore
  free(imgFloat);

  // Save to the new file
  // get new filename
  savePath = getSavePath(filename);
  // write image
  stbi_write_jpg(savePath, imgWidth, imgHeight, numChannels, imgBytes, 100);
  
  // Free byte image and filename
  free((void *) savePath);
  STBI_FREE(imgBytes);
  
  return EXIT_SUCCESS;
}


void blurImg(float *image_h, size_t width, size_t height, size_t channels, size_t ksize){
  float *image_d, *result_d;
  cudaError_t cudaErr;
  int gHeight, gWidth, bHeight, bWidth;
  size_t bufferSize = width * height * channels * sizeof(float);

  bWidth = 32;
  bHeight = 32;
  gWidth = (int) ceil(width / (float) bWidth);
  gHeight = (int) ceil(height / (float) bHeight);
  dim3 gDim(channels, gWidth, gHeight);
  dim3 bDim(bWidth, bHeight, 1);

  cudaErr = cudaMalloc((void **) &image_d, bufferSize);
  CUDA_CHKERR(cudaErr)
  cudaErr = cudaMalloc((void **) &result_d, bufferSize);
  CUDA_CHKERR(cudaErr)
  cudaErr = cudaMemcpy( (void*) image_d, (void*) image_h, bufferSize, cudaMemcpyHostToDevice);
  CUDA_CHKERR(cudaErr)

  kernel_blurImg<<<gDim, bDim>>>(image_d, result_d, width, height, channels, ksize);

  cudaErr = cudaMemcpy( (void *) image_h, (void *) result_d, bufferSize, cudaMemcpyDeviceToHost);
  CUDA_CHKERR(cudaErr)

  // Done with these device variables
  cudaFree(image_d);
  cudaFree(result_d);
}

__global__ void kernel_blurImg(const float *M_in, float *M_out, size_t width, size_t height, size_t channels, size_t ksize) {
  // Ensure the target pixel is valid
  int ci, cj, ck;  // output (center) pixel and channel
  float *cPix;
  int kradius = ksize / 2;
  int nAcc = 0;
  float acc = 0.0;

  // Grid dim(blockIdx): (height: z, width: y, channels: x)
  // Block dim(threadIdx):(height: y, width: x)
  
  ck = blockIdx.x;
  cj = blockIdx.y * blockDim.x + threadIdx.x;
  ci = blockIdx.y * blockDim.y + threadIdx.y;

  if (ci >= height || cj >= width) return;

  for (int i = ci - kradius; i<=ci + kradius; i++) {
    if (i < 0 || i >= height) continue;
    for (int j = cj - kradius; j <= cj + kradius; j++) {
      // Ensure the pixel is not out of bounds
      if (j < 0 || j >= width) continue;
      // Compute the location of this pixel in the image buffer
      int imgByteIdx = i * (width * channels) + j * (channels) + ck;
      acc += M_in[imgByteIdx];
      nAcc++;
    }
  }

  // Find the center pixel
  cPix = M_out + ci * (width * channels) + cj * (channels) + ck;
  // nAcc cannot be zero because ci,cj,ck is guaranteed in range
  *cPix = acc / nAcc;
}


const char *getSavePath(const char *filename) {
  // Find the location of the last period in the filename to chop off the extension
  int periodLocation, length;
  char *newFilename;
  length = strlen(filename);
  periodLocation = length;

  while (periodLocation > 0 && filename[--periodLocation] != '.') continue;
  if (periodLocation < 0) { return strdup("blurred.jpg"); }  // fallback

  newFilename = (char *) malloc(periodLocation + 13);  // Appending 12 characters: "_blurred.jpg\x0"
  strncpy(newFilename, filename, periodLocation);
  strncpy(newFilename + periodLocation, "_blurred.jpg", 12);
  return (const char *)newFilename;
}
