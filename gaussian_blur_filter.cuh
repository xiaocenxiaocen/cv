#ifndef GAUSSIAN_BLUR_FILTER_H
#define GAUSSIAN_BLUR_FILTER_H

typedef unsigned char uchar;

extern "C" void myGaussianBlur_gpu(uchar * __restrict__ src, uchar * __restrict__ dst, int w, int h, int kw, int kh, double sigmaX, double sigmaY, int channels);

#endif
