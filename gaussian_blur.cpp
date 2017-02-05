#include <cstdio>
#include <iostream>
#include <omp.h>
#include <time.h>
#include <cstdlib>
#include <cstring>
#include <assert.h>
#include <math.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/flann/miniflann.hpp"

#include <mmintrin.h> 	// MMX
#include <xmmintrin.h>	// SSE
#include <emmintrin.h>	// SSE2
#include <immintrin.h>	// AVX

#include "gaussian_blur_filter.cuh"

using namespace cv; // all the new API is put into "cv" namespace. Export its content
using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979
#endif

template<typename T> T * ArrayAlloc(const size_t n0)
{
	T * ptr __attribute__((aligned(16))) = (T*)malloc(sizeof(T) * n0);
	memset(ptr, 0, sizeof(T) * n0);
	return ptr;
}

#define ArrayFree3D(ptr) \
{ \
	free(**(ptr)); \
	free(*(ptr)); \
	free(ptr); \
}

#define ArrayFree2D(ptr) \
{ \
	free(*(ptr)); \
	free(ptr); \
}

/* @brief: template function for 2d array allocation
 * @param: n0, n1, size of dimensions
 * @retval: buffer, buffer for array
 */
template<typename T> void ArrayAlloc2D(T ***buffer, const int n0, const int n1)
{
	*buffer = (T**)malloc(sizeof(T*) * n0);
	(*buffer)[0] = (T*)malloc(sizeof(T) * n0 * n1);
	for(int i0 = 1; i0 < n0; i0++) (*buffer)[i0] = (*buffer)[0] + i0 * n1;
	return;
}

/* @brief: template function for 3d array allocation
 * @param: n0, n1, n2 size of dimensions
 * @retval: buffer, buffer for array
 */
template<typename T> void ArrayAlloc3D(T ****buffer, const int n0, const int n1, const int n2)
{
	*buffer = (T***)malloc(sizeof(T**) * n0);
	(*buffer)[0] = (T**)malloc(sizeof(T*) * n0 * n1);
	for(int i0 = 1; i0 < n0; i0++) (*buffer)[i0] = (*buffer)[0] + i0 * n1;
	(*buffer)[0][0] = (T*)malloc(sizeof(T) * n0 * n1 * n2);
	for(int i1 = 1; i1 < n1; i1++) (*buffer)[0][i1] = (*buffer)[0][0] + i1 * n2;
	for(int i0 = 1; i0 < n0; i0++) {
		for(int i1 = 0; i1 < n1; i1++) {
			(*buffer)[i0][i1] = (*buffer)[0][0] + i0 * n1 * n2 + i1 * n2;
		}
	}
	return;
}

void myGaussianBlur_(const Mat& src, Mat& dst, Size kerSize, double sigmaX, double sigmaY = 1.0)
{
	// get gaussian kernel
	int kx = kerSize.width;
	int ky = kerSize.height;
	assert(kx % 2 == 1 && ky % 2 == 1);
	int radiusX = kx / 2;
	int radiusY = ky / 2;
	kx = radiusX + 1;
	ky = radiusY + 1;
	float kerY[ky];
	float kerX[kx];
	assert(sigmaX > 0 && sigmaY > 0);
	double weightX = 1.0 / (sqrt(2.0 * M_PI) * sigmaX);
	double weightY = 1.0 / (sqrt(2.0 * M_PI) * sigmaY);
	double invSqrSigmaX = 1.0 / (2.0 * sigmaX * sigmaX);
	double invSqrSigmaY = 1.0 / (2.0 * sigmaY * sigmaY);
	double sumY = 0.0;
	for(int y = 0; y < ky; y++) {
		kerY[y] = weightY * exp( - y * y * invSqrSigmaY );
		if(y > 0) sumY += 2.0 * kerY[y];
		else sumY += kerY[y];
	}
	for(int y = 0; y < ky; y++) kerY[y] /= sumY;
	double sumX = 0.0; 
	for(int x = 0; x < kx; x++) {
		kerX[x] = weightX * exp( - x * x * invSqrSigmaX );
		if(x > 0) sumX += 2.0 * kerX[x];
		else sumX += kerX[x];
	}
	for(int x = 0; x < kx; x++) kerX[x] /= sumX;
	
	// allocate buffer	
	Size srcsize = src.size();
	Size dstsize = dst.size();
	int w = srcsize.width;
	int h = srcsize.height;
	int ww = w + 2 * radiusX;
	int hh = h + 2 * radiusY;
	assert(dstsize == srcsize);
	Mat filBuff(hh, ww, CV_8UC3);
	Mat dstBuff(hh, w, CV_32FC3);
	const int threadNum = 4;
	for(int y = radiusY; y < hh - radiusY; y++) {
		Vec3b * bufPtr = filBuff.ptr<Vec3b>(y) + radiusX;
		const Vec3b * srcPtr = src.ptr<Vec3b>(y - radiusY);
		memcpy(bufPtr, srcPtr, sizeof(Vec3b) * w);
	}
	
	// left & right
	for(int y = radiusY; y < hh - radiusY; y++) {
		Vec3b * bufPtr = filBuff.ptr<Vec3b>(y);
		for(int x = 0; x < radiusX; x++) {
			bufPtr[x] = bufPtr[radiusX];
			bufPtr[x + ww - radiusX] = bufPtr[ww - radiusX - 1];
		}
	}

	// top & bottom
	Vec3b * topPtr = filBuff.ptr<Vec3b>(0);
	Vec3b * botPtr = filBuff.ptr<Vec3b>(hh - radiusY);
	Vec3b * topLin = filBuff.ptr<Vec3b>(radiusY);
	Vec3b * botLin = filBuff.ptr<Vec3b>(hh - radiusY - 1);
	for(int y = 0; y < radiusY; y++, topPtr += ww, botPtr += ww) {
		memcpy(topPtr, topLin, sizeof(Vec3b) * ww);
		memcpy(botPtr, botLin, sizeof(Vec3b) * ww);
	}

#ifdef _SSE2
	int cn = src.channels();
	assert(w * cn % 16 == 0);
	__m128i z = _mm_setzero_si128();
	// apply gaussian filter
	#ifdef _OPENMP
	#pragma omp parallel num_threads(threadNum) shared(filBuff, dstBuff, dst, kerX, kerY)
	#endif
	{
	// row filter
	#ifdef _OPENMP
	#pragma omp for schedule(static, hh / threadNum)
	#endif
	for(int y = 0; y < hh; y++) {
		Vec3b * srcPtr = filBuff.ptr<Vec3b>(y) + radiusX;
		Vec3f * dstPtr = dstBuff.ptr<Vec3f>(y);
		uchar * srcRaw = reinterpret_cast<uchar*>(&srcPtr[0][0]);
		float * dstRaw = reinterpret_cast<float*>(&dstPtr[0][0]);
		int x = 0;
		for( ; x < w * cn; x += 16, srcRaw += 16) {
			__m128 f = _mm_load_ss(kerX);
			f = _mm_shuffle_ps(f, f, 0);
			__m128i x0 = _mm_loadu_si128((__m128i*)(srcRaw));
			__m128i x1, x2, x3, x4, y0;
			x1 = _mm_unpackhi_epi8(x0, z);
			x2 = _mm_unpacklo_epi8(x0, z);
			x3 = _mm_unpackhi_epi16(x2, z);
			x4 = _mm_unpacklo_epi16(x2, z);
			x2 = _mm_unpacklo_epi16(x1, z);
			x1 = _mm_unpackhi_epi16(x1, z);
			__m128 s1, s2, s3, s4;
			s1 = _mm_mul_ps(f, _mm_cvtepi32_ps(x1));
			s2 = _mm_mul_ps(f, _mm_cvtepi32_ps(x2));
			s3 = _mm_mul_ps(f, _mm_cvtepi32_ps(x3));
			s4 = _mm_mul_ps(f, _mm_cvtepi32_ps(x4));
			for(int k = 1; k < kx; k++) {
				f = _mm_load_ss(kerX + k);
				f = _mm_shuffle_ps(f, f, 0);
				uchar * shi = srcRaw + k * cn;
				uchar * slo = srcRaw - k * cn;
				x0 = _mm_loadu_si128((__m128i*)(shi));
				y0 = _mm_loadu_si128((__m128i*)(slo));
				x1 = _mm_unpackhi_epi8(x0, z);
				x2 = _mm_unpacklo_epi8(x0, z);
				x3 = _mm_unpackhi_epi8(y0, z);
				x4 = _mm_unpacklo_epi8(y0, z);
				x1 = _mm_add_epi16(x1, x3);
				x2 = _mm_add_epi16(x2, x4);
			
				x3 = _mm_unpackhi_epi16(x2, z);
				x4 = _mm_unpacklo_epi16(x2, z);
				x2 = _mm_unpacklo_epi16(x1, z);
				x1 = _mm_unpackhi_epi16(x1, z);
				s1 = _mm_add_ps(s1, _mm_mul_ps(f, _mm_cvtepi32_ps(x1)));
				s2 = _mm_add_ps(s2, _mm_mul_ps(f, _mm_cvtepi32_ps(x2)));
				s3 = _mm_add_ps(s3, _mm_mul_ps(f, _mm_cvtepi32_ps(x3)));
				s4 = _mm_add_ps(s4, _mm_mul_ps(f, _mm_cvtepi32_ps(x4)));	
			}
			_mm_storeu_ps(dstRaw + x, s4);
			_mm_storeu_ps(dstRaw + x + 4, s3);
			_mm_storeu_ps(dstRaw + x + 8, s2);
			_mm_storeu_ps(dstRaw + x + 12, s1);
		}
	}
	
	// column filter
	#ifdef _OPENMP
	#pragma omp for schedule(static, h / threadNum)
	#endif
	for(int y = 0; y < h; y++) {
		Vec3f * srcPtr = dstBuff.ptr<Vec3f>(y + radiusY);
		Vec3b * dstPtr = dst.ptr<Vec3b>(y);
		float * srcRaw = reinterpret_cast<float*>(&srcPtr[0][0]);
		uchar * dstRaw = reinterpret_cast<uchar*>(&dstPtr[0][0]);
		int x = 0;
		for( ; x < w * cn; x += 16, srcRaw += 16) {
			__m128 f = _mm_load_ss(kerY);
			f = _mm_shuffle_ps(f, f, 0);
			__m128 s1, s2, s3, s4;
			__m128 s0;
			s1 = _mm_loadu_ps(srcRaw);
			s2 = _mm_loadu_ps(srcRaw + 4);
			s3 = _mm_loadu_ps(srcRaw + 8);
			s4 = _mm_loadu_ps(srcRaw + 12);
			s1 = _mm_mul_ps(s1, f);
			s2 = _mm_mul_ps(s2, f);
			s3 = _mm_mul_ps(s3, f);
			s4 = _mm_mul_ps(s4, f);
			for(int k = 1; k < ky; k++) {
				f = _mm_load_ss(kerY + k);
				f = _mm_shuffle_ps(f, f, 0);
				s0 = _mm_add_ps(_mm_loadu_ps(srcRaw + k * w * cn), _mm_loadu_ps(srcRaw - k * w * cn));
				s1 = _mm_add_ps(s1, _mm_mul_ps(f, s0));
				s0 = _mm_add_ps(_mm_loadu_ps(srcRaw + 4 + k * w * cn), _mm_loadu_ps(srcRaw + 4 - k * w * cn));
				s2 = _mm_add_ps(s2, _mm_mul_ps(f, s0));
				s0 = _mm_add_ps(_mm_loadu_ps(srcRaw + 8 + k * w * cn), _mm_loadu_ps(srcRaw + 8 - k * w * cn));
				s3 = _mm_add_ps(s3, _mm_mul_ps(f, s0));
				s0 = _mm_add_ps(_mm_loadu_ps(srcRaw + 12 + k * w * cn), _mm_loadu_ps(srcRaw + 12 - k * w * cn));
				s4 = _mm_add_ps(s4, _mm_mul_ps(f, s0));
			}
			__m128i x1 = _mm_cvttps_epi32(s1);
			__m128i x2 = _mm_cvttps_epi32(s2);
			__m128i x3 = _mm_cvttps_epi32(s3);
			__m128i x4 = _mm_cvttps_epi32(s4);
			x1 = _mm_packs_epi32(x1, x2);
			x2 = _mm_packs_epi32(x3, x4);
			x1 = _mm_packus_epi16(x1, x2);
			_mm_storeu_si128((__m128i*)(dstRaw + x), x1);
		}
	}
	}
#else
	// apply gaussian filter
	#ifdef _OPENMP
	#pragma omp parallel num_threads(threadNum) shared(filBuff, dstBuff, dst, kerX, kerY)
	#endif
	{
	// row filter
	#ifdef _OPENMP
	#pragma omp for schedule(static, hh / threadNum)
	#endif
	for(int y = 0; y < hh; y++) {
		Vec3b * srcPtr = filBuff.ptr<Vec3b>(y) + radiusX;
		Vec3f * dstPtr = dstBuff.ptr<Vec3f>(y);
		for(int x = 0; x < w; x++) {
			double val0 = kerX[0] * srcPtr[x][0];
			double val1 = kerX[0] * srcPtr[x][1];
			double val2 = kerX[0] * srcPtr[x][2];
			for(int xx = 1; xx < kx; xx++) {
				#define Term(i, j) (kerX[xx + (i)] * (srcPtr[x + xx + (i)][(j)] + srcPtr[x - xx - (i)][(j)]))
				val0 += Term(0, 0);
				val1 += Term(0, 1);
				val2 += Term(0, 2);
			}
			dstPtr[x] = Vec3f(val0, val1, val2);
		}
	}
	
	// column filter
	#ifdef _OPENMP
	#pragma omp for schedule(static, h / threadNum)
	#endif
	for(int y = 0; y < h; y++) {
		Vec3f * srcPtr = dstBuff.ptr<Vec3f>(y + radiusY);
		Vec3b * dstPtr = dst.ptr<Vec3b>(y);
		for(int x = 0; x < w; x++) {
			Vec3d vec = kerY[0] * srcPtr[x];		
			for(int yy = 1; yy < ky; yy++) {
				vec += kerY[yy] * (*(srcPtr + yy * w + x) + *(srcPtr - yy * w + x));
			}
			vec[0] = vec[0] < 0 ? 0 : vec[0]; vec[0] = vec[0] > 0xff ? 0xff : vec[0];
			vec[1] = vec[1] < 0 ? 0 : vec[1]; vec[1] = vec[1] > 0xff ? 0xff : vec[1];
			vec[2] = vec[2] < 0 ? 0 : vec[2]; vec[2] = vec[2] > 0xff ? 0xff : vec[2];
			dstPtr[x] = Vec3b(vec[0], vec[1], vec[2]);
		}
	}
	}
#endif

}

int main(int argc, char * argv[])
{
	if(argc != 4) {
		fprintf(stdout, "Usage: inputfile outputfile1 outputfile2\n");
		return -1;
	}

	const char * imagename = argv[1];
	const char * outputimage1 = argv[2];
	const char * outputimage2 = argv[3];
	Mat img = imread(imagename);
	if(img.empty()) {
		fprintf(stderr, "ERROR: cannot load image %s\n", imagename);
		return -1;
	}
	
	if( !img.data )
		return -1;

	cout << img.size() << "\n";

	cout << img.type() << "\n";

	Mat dst(img.size(), img.type());

	clock_t beg, end;

	beg = clock();
	double t;
	t = omp_get_wtime();
	GaussianBlur(img, dst, Size(11, 11), 1, 1, BORDER_REPLICATE);
	t = omp_get_wtime() - t;
	end = clock();
	cout << "clock cycles of opencv::GaussianBlur(): " << static_cast<float>(end - beg) << "\n";
	cout << "elapsed time of opencv::GaussianBlur(): " << t << "s\n";
	
	imwrite("opecv.jpg", dst);

	beg = clock();
	t = omp_get_wtime();
	Mat dst1(img.size(), img.type());
	myGaussianBlur_(img, dst1, Size(11, 11), 1, 1);
	t = omp_get_wtime() - t;
	end = clock();
	cout << "clock cycles of myGaussianBlur(): " << static_cast<float>(end - beg) << "\n";
	cout << "elapsed time of myGaussianBlur(): " << t << "s\n";
	imwrite(outputimage1, dst1);

	imwrite("error.jpg", dst1 - dst);

	// warmup
	uchar * h_src = reinterpret_cast<uchar*>(&((img.ptr<Vec3b>(0))[0][0]));
	uchar * h_dst = reinterpret_cast<uchar*>(&((dst1.ptr<Vec3b>(0))[0][0]));
	int w = img.cols;
	int h = img.rows;
	int cn = img.channels();
	myGaussianBlur_gpu(h_src, h_dst, w, h, 11, 11, 1, 1, cn);

	beg = clock();
	t = omp_get_wtime();
	Mat dst2(img.size(), img.type());
	h_src = reinterpret_cast<uchar*>(&((img.ptr<Vec3b>(0))[0][0]));
	h_dst = reinterpret_cast<uchar*>(&((dst2.ptr<Vec3b>(0))[0][0]));
	myGaussianBlur_gpu(h_src, h_dst, w, h, 11, 11, 1, 1, cn);
	t = omp_get_wtime() - t;
	end = clock();
	cout << "clock cycles of myGaussianBlur_gpu(): " << static_cast<float>(end - beg) << "\n";
	cout << "elapsed time of myGaussianBlur_gpu(): " << t << "s\n";
	imwrite(outputimage2, dst2);

	imwrite("error2.jpg", dst2 - dst);

	cout << dst.size() << "\n";

	return 0;	
}
