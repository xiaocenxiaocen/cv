/* @file: gaussian_kernel.cu
 * @author: Zhang Xiao
 * @date: 2017.01.13
 */
#include <cstdio>
#include <math.h>
#include <omp.h>
#include <assert.h>
#include <iostream>
using namespace std;

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#define ALIGNEDMENT_8u (32 / sizeof(uchar))
template <int BLOCK_Y, int BLOCK_X, int RADIUS_X, int cn, int alignedment_8u> __global__ static void 
rowFilterKernel_8u32f(uchar * __restrict__ src, int strideSrc, float * __restrict__ dst, int strideDst, int height)
{
	__shared__ float sdata[BLOCK_Y][BLOCK_X + 2 * cn * RADIUS_X];

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int ltidx = tx + cn * RADIUS_X;
	int ltidy = ty;

	int gtidx = blockIdx.x * blockDim.x + tx + alignedment_8u;
	int gtidy = blockIdx.y * blockDim.y + ty;

	int inputIdx = gtidy * strideSrc + gtidx;
	int outputIdx = gtidy * strideDst + gtidx - alignedment_8u;

	sdata[ltidy][ltidx] = float(src[inputIdx]);
	for(int i = tx; i < cn * RADIUS_X; i += BLOCK_X) {
		int leftBoundIdx = inputIdx - cn * RADIUS_X + i - tx;
		int rightBoundIdx = inputIdx + BLOCK_X + i - tx;
		sdata[ltidy][i                          ] = float(src[leftBoundIdx]);
		sdata[ltidy][i + BLOCK_X + cn * RADIUS_X] = float(src[rightBoundIdx]);
	}
	__syncthreads();

	float val = coeffX[0] * sdata[ltidy][ltidx]
	#pragma unroll
	for(int k = 1; k < ksize; k++) {
		val += coeffX[k] * (sdata[ltidy][ltidx + cn * k] + sdata[ltidy][ltidx - cn * k]);
	}

	dst[outputIdx] = val;
}

template <int BLOCK_Y, int BLOCK_X, int RADIUS_Y, int cn> __global__ static void
columnFilterKernel_32f8u(float * __restrict__ src, int strideSrc, uchar * __restrict__ dst, int strideDst, int height)
{
	__shared__ float sdata[BLOCK_Y + 2 * RADIUS_Y][BLOCK_X];
	
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	int ltidx = tx;
	int ltidy = ty + RADIUS_Y;

	int gtidx = blockIdx.x * blockDim.x + tx;
	int gtidy = blockIdx.y * blockDim.y + ty;

	int inputIdx = gtidy * strideSrc + gtidx;
	int outputIdx = gtidy * strideDst + gtidx;

	// make boarder, defaut replicated boarder
	sdata[ltidy][ltidx] = src[inputIdx];
	for(int j = ty; j < RADIUS_Y; j += BLOCK_Y) {
		int upperBoundIdx = gtidy + j - ty >= RADIUS_Y ? inputIdx - (RADIUS_Y - j + ty) * strideSrc : gtidx;
		int bottomBoundIdx = gtidy + j - ty + BLOCK_Y < height - RADIUS_Y ? inputIdx + (BLOCK_Y + j - ty) : (height - 1) * strideSrc + gtidx;
		sdata[j                     ][ltidx] = src[upperBoundIdx];
		sdata[j + BLOCK_Y + RADIUS_Y][ltidx] = src[bottomBoundIdx];
	}
	__syncthreads();

	float val = coeffY[0] * sdata[ltidy][ltidx];
	#pragma unroll
	for(int k = 1; k < ksize; k++) {
		val += coeffY[k] * (sdata[ltidy + k][ltidx] + sdata[ltidy - k][ltidx]);
	}

	dst[outputIdx] = val;
}

__global__ static void rowPaddingKernel()
{
}

void myGaussianBlur_gpu()
{
	
}
