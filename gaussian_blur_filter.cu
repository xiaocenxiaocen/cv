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

#include "gaussian_blur_filter.cuh"

#define MAX_KSIZE 1024
__constant__ float coeffX[MAX_KSIZE];
__constant__ float coeffY[MAX_KSIZE];

#define ALIGNEDMENT_8u (32 / sizeof(uchar))
template <int BLOCK_Y, int BLOCK_X, int RADIUS_X, int cn> __global__ static void 
rowFilterKernel_8u32f(uchar * __restrict__ src, int strideSrc, float * __restrict__ dst, int strideDst, int height, int alignedment_8u)
{
	__shared__ float sdata[BLOCK_Y][BLOCK_X + 2 * cn * RADIUS_X];

	int ksize = RADIUS_X + 1;

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

	float val = coeffX[0] * sdata[ltidy][ltidx];
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

	int ksize = RADIUS_Y + 1;	

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
		int bottomBoundIdx = gtidy + j - ty + BLOCK_Y < height - RADIUS_Y ? inputIdx + (BLOCK_Y + j - ty) * strideSrc : (height - 1) * strideSrc + gtidx;
		sdata[j                     ][ltidx] = src[upperBoundIdx];
		sdata[j + BLOCK_Y + RADIUS_Y][ltidx] = src[bottomBoundIdx];
	}
	__syncthreads();

	float val = coeffY[0] * sdata[ltidy][ltidx];
	#pragma unroll
	for(int k = 1; k < ksize; k++) {
		val += coeffY[k] * (sdata[ltidy + k][ltidx] + sdata[ltidy - k][ltidx]);
	}

	val = val >= 0 ? val : 0;
	val = val <= 0xff ? val : 0xff;

	dst[outputIdx] = val;
}

template <int BLOCK_Y, int BLOCK_X, int RADIUS_X, int cn> __global__ static void 
makeBorderReplicate_8u(uchar * __restrict__ src, int w, int strideSrc, int alignedment_8u)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int gtidy = blockIdx.y * blockDim.y + ty;

	if(blockIdx.x == 0) {
		int inputIdx = gtidy * strideSrc + alignedment_8u;
		uchar s[cn];
		for(int c = 0; c < cn; c++) {
			s[c] = src[inputIdx + c]; 
		}
		for(int i = tx; i < cn * RADIUS_X; i += BLOCK_X) {
			int c = i % cn;
			src[inputIdx - cn * RADIUS_X + i] = s[c];
		}
	}
	else if(blockIdx.x == gridDim.x - 1) {
		int inputIdx = gtidy * strideSrc + w - cn + alignedment_8u;
		int outputIdx = gtidy * strideSrc + w + alignedment_8u;
		uchar s[cn];
		for(int c = 0; c < cn; c++) {
			s[c] = src[inputIdx + c]; 
		}
		for(int i = tx; i < cn * RADIUS_X; i += BLOCK_X) {
			int c = i % cn;
			src[outputIdx + i] = s[c];
		}
	}
}

void myGaussianBlur_gpu(uchar * __restrict__ src, uchar * __restrict__ dst, int w, int h, int kw, int kh, double sigmaX, double sigmaY, int channels)
{
	// get gaussian kernel
	assert(kw % 2 == 1 && kh % 2 == 1);
	int radiusX = kw / 2;
	int radiusY = kh / 2;
	float h_kx[radiusX + 1];
	float h_ky[radiusY + 1];
	assert(sigmaX > 0 && sigmaY > 0);
	double weightX = 1.0 / (sqrt(2.0 * M_PI) * sigmaX);
	double weightY = 1.0 / (sqrt(2.0 * M_PI) * sigmaY);
	double invSqrSigmaX = 1.0 / (2.0 * sigmaX * sigmaX);
	double invSqrSigmaY = 1.0 / (2.0 * sigmaY * sigmaY);
	double sumY = 0.0;
	for(int y = 0; y < radiusY + 1; y++) {
		h_ky[y] = weightY * exp( - y * y * invSqrSigmaY );
		if(y > 0) sumY += 2.0 * h_ky[y];
		else sumY += h_ky[y];
	}
	for(int y = 0; y < radiusY + 1; y++) h_ky[y] /= sumY;
	double sumX = 0.0; 
	for(int x = 0; x < radiusX + 1; x++) {
		h_kx[x] = weightX * exp( - x * x * invSqrSigmaX );
		if(x > 0) sumX += 2.0 * h_kx[x];
		else sumX += h_kx[x];
	}
	for(int x = 0; x < radiusX + 1; x++) h_kx[x] /= sumX;

	cudaMemcpyToSymbol(coeffX, (void*)h_kx, sizeof(float) * (radiusX + 1));
	cudaMemcpyToSymbol(coeffY, (void*)h_ky, sizeof(float) * (radiusY + 1));

	// padding
	w *= channels;
	int block_y = 4;
	int block_x = 32;
	int wPad = w % block_x == 0 ? w : block_x * (w / block_x + 1);
	int hPad = h % block_y == 0 ? h : block_y * (h / block_y + 1);
	int alignedment_8u = max(static_cast<int>(ALIGNEDMENT_8u), channels * radiusX);
	alignedment_8u = alignedment_8u % ALIGNEDMENT_8u == 0 ? alignedment_8u : ALIGNEDMENT_8u * (alignedment_8u / ALIGNEDMENT_8u + 1);
	int ww = wPad;
	wPad += 2 * alignedment_8u;

	const int threadNum = 4;
	
	uchar * h_src = new uchar[wPad * hPad];
	
	uchar * d_src;
	uchar * d_dst;
	float * d_buffer;
	cudaMalloc((void**)&d_src, sizeof(uchar) * wPad * hPad);
	cudaMalloc((void**)&d_dst, sizeof(uchar) * ww * hPad);
	cudaMalloc((void**)&d_buffer, sizeof(float) * ww * hPad);
	cudaMemset(d_src, 0, sizeof(uchar) * wPad * hPad);
	#pragma omp parallel for num_threads(threadNum) shared(src, h_src) schedule(static, h / threadNum)
	for(int y = 0; y < h; y++) {
		memcpy(h_src + y * wPad + alignedment_8u, src + y * w, sizeof(uchar) * w);
	}
	for(int y = h; y < hPad; y++) {
		memcpy(h_src + y * wPad + alignedment_8u, src + (h - 1) * w, sizeof(uchar) * w);
	}
	cudaMemcpy(d_src, h_src, sizeof(uchar) * wPad * hPad, cudaMemcpyHostToDevice);	
	delete[] h_src;

//	for(int y = 0; y < h; y++) {
//		cudaMemcpy(d_src + y * wPad + alignedment_8u, src + y * w, sizeof(uchar) * w, cudaMemcpyHostToDevice);
//	}
//	for(int y = h; y < hPad; y++) {
//		cudaMemcpy(d_src + y * wPad + alignedment_8u, src + (h - 1) * w, sizeof(uchar) * w, cudaMemcpyHostToDevice);
//	}

	float gpuTime;

	cudaEvent_t start;
	cudaEvent_t finish;

	cudaEventCreate(&start);
	cudaEventCreate(&finish);

	cudaEventRecord(start,0);

	int gridX = ww / block_x;
	int gridY = hPad / block_y;
	dim3 grids(gridX, gridY, 1);
	dim3 threads(block_x, block_y, 1);
	if(channels == 3 && block_y == 4 && block_x == 32 && radiusX == 5 && radiusY == 5) {
		makeBorderReplicate_8u<4, 32, 5, 3><<<grids, threads>>>(d_src, w, wPad, alignedment_8u);
		rowFilterKernel_8u32f<4, 32, 5, 3><<<grids, threads>>>(d_src, wPad, d_buffer, ww, hPad, alignedment_8u);
		columnFilterKernel_32f8u<4, 32, 5, 3><<<grids, threads>>>(d_buffer, ww, d_dst, ww, hPad);
	}
	
	cudaDeviceSynchronize();

	cudaEventSynchronize(finish);

	cudaEventRecord(finish,0);

	cudaEventSynchronize(finish);
	
	cudaEventElapsedTime(&gpuTime,start,finish);

	cudaEventDestroy(start);
	cudaEventDestroy(finish);

	gpuTime *= 0.001;
	cout << "\nwall time of gpu gaussian blur: " << gpuTime << "\n";

	uchar * h_dst = new uchar[ww * hPad];
	cudaMemcpy(h_dst, d_dst, sizeof(uchar) * ww * hPad, cudaMemcpyDeviceToHost);
	#pragma omp parallel for num_threads(threadNum) shared(dst, h_dst) schedule(static, h / threadNum)
	for(int y = 0; y < h; y++) {
		memcpy(dst + y * w, h_dst + y * ww, sizeof(uchar) * w);
//		cudaMemcpy(dst + y * w, d_dst + y * ww, sizeof(uchar) * w, cudaMemcpyDeviceToHost);
	}
	delete[] h_dst;

	cudaFree(d_src); d_src = NULL;
	cudaFree(d_dst); d_dst = NULL;
	cudaFree(d_buffer); d_buffer = NULL;	
}
