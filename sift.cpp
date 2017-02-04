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

void myGaussianBlur(const Mat& src, Mat& dst, Size kerSize, double sigmaX, double sigmaY = 1.0)
{
	// get gaussian kernel
	int kx = kerSize.width;
	int ky = kerSize.height;
	assert(kx % 2 == 1 && ky % 2 == 1);
	double * kerX = ArrayAlloc<double>(kx);
	double * kerY = ArrayAlloc<double>(ky);
	int radiusX = kx / 2;
	int radiusY = ky / 2;
	assert(sigmaX > 0 && sigmaY > 0);
	double weightX = 1.0 / (sqrt(2.0 * M_PI) * sigmaX);
	double weightY = 1.0 / (sqrt(2.0 * M_PI) * sigmaY);
	double invSqrSigmaX = 1.0 / (2.0 * sigmaX * sigmaX);
	double invSqrSigmaY = 1.0 / (2.0 * sigmaY * sigmaY);
	for(int y = 0; y < ky; y++) {
		kerY[y] = weightY * exp( - (y - radiusY) * (y - radiusY) * invSqrSigmaY );
	}
	for(int x = 0; x < kx; x++) {
		kerX[x] = weightX * exp( - (x - radiusX) * (x - radiusX) * invSqrSigmaX );
	}
	
	// allocate buffer	
	Size srcsize = src.size();
	Size dstsize = dst.size();
	int w = srcsize.width;
	int h = srcsize.height;
	int wb = w + 2 * radiusX;
	int hb = h + 2 * radiusY;
	assert(dstsize == srcsize);
	Mat filBuff(hb, wb, CV_8UC3);
	Mat dstBuff(hb, w, CV_64FC3);
	const int threadNum = 8;
	double t0 = 0.0;
	double t1 = 0.0;
	for(int y = radiusY; y < hb - radiusY; y++) {
		Vec3b * bufPtr = filBuff.ptr<Vec3b>(y); 
		for(int x = radiusX; x < wb - radiusX; x++) {
			bufPtr[x] = src.at<Vec3b>(y - radiusY, x - radiusX);
		}
	}
	
	// left & right
	for(int y = radiusY; y < hb - radiusY; y++) {
		Vec3b * bufPtr = filBuff.ptr<Vec3b>(y);
		for(int x = 0; x < radiusX; x++) {
			bufPtr[x] = bufPtr[radiusX];
			bufPtr[x + wb - radiusX] = bufPtr[wb - radiusX - 1];
		}
	}

	// top & bottom
	Vec3b * topPtr = filBuff.ptr<Vec3b>(0);
	Vec3b * botPtr = filBuff.ptr<Vec3b>(hb - radiusY);
	Vec3b * topLin = filBuff.ptr<Vec3b>(radiusY);
	Vec3b * botLin = filBuff.ptr<Vec3b>(hb - radiusY - 1);
	for(int y = 0; y < radiusY; y++, topPtr += wb, botPtr += wb) {
		memcpy(topPtr, topLin, sizeof(Vec3b) * wb);
		memcpy(botPtr, botLin, sizeof(Vec3b) * wb);
	}

	t0 = omp_get_wtime();
	// apply gaussian filter
//	#ifdef _OPENMP
//	#pragma omp parallel for num_threads(threadNum) schedule(dynamic) shared(kerBuff, filBuff, dst)
//	#endif
//	for(int y = 0; y < h; y++) {
//		for(int x = 0; x < w; x++) {
//			double val0(0.0);
//			double val1(0.0);
//			double val2(0.0);
//			double * kerPtr = kerBuff[0];
//			for(int yy = -radiusY; yy <= radiusY; yy++, kerPtr += kx) {
//				Vec3b * srcPtr = filBuff.ptr<Vec3b>(y + radiusY + yy);
//				for(int xx = -radiusX; xx <= radiusX; xx++) {
//					val0 += kerPtr[xx + radiusX] * srcPtr[x + xx + radiusX][0];
//					val1 += kerPtr[xx + radiusX] * srcPtr[x + xx + radiusX][1];
//					val2 += kerPtr[xx + radiusX] * srcPtr[x + xx + radiusX][2];
//				}
//			}
//			val0 = val0 < 0 ? 0 : val0; val0 = val0 > 0xff ? 0xff : val0;
//			val1 = val1 < 0 ? 0 : val1; val1 = val1 > 0xff ? 0xff : val1;
//			val2 = val2 < 0 ? 0 : val2; val2 = val2 > 0xff ? 0xff : val2;
//			dst.at<Vec3b>(y, x) = Vec3b(static_cast<uchar>(val0), static_cast<uchar>(val1), static_cast<uchar>(val2));
//		}
//	}	

	int chunk_size = 256;
	#ifdef _OPENMP
	#pragma omp parallel num_threads(threadNum) shared(kerX, kerY, filBuff, dstBuff, dst)
	#endif
	{
	#ifdef _OPENMP
	#pragma omp for schedule(dynamic)
	#endif
	for(int x = 0; x < w; x += chunk_size) {
		for(int y = 0; y < hb; y++) {
			Vec3b * srcPtr = filBuff.ptr<Vec3b>(y);
			Vec3d * dstPtr = dstBuff.ptr<Vec3d>(y);
			vector<Vec3d> row(chunk_size, 0);
			for(int xx = 0; xx < kx; xx++) {
				for(int ii = 0; ii < chunk_size && x + ii < w; ii++) {
		//		double val0(0.0);
		//		double val1(0.0);
		//		double val2(0.0);
		//		double val[4] = {0.0};
		//		__m128d mmx0 = _mm_set1_pd(0.0);
		//		__m128d mmx1 = _mm_set1_pd(0.0);
		//			val0 += kerX[xx] * srcPtr[x + xx][0];
		//			val1 += kerX[xx] * srcPtr[x + xx][1];
		//			val2 += kerX[xx] * srcPtr[x + xx][2];
					row[ii] += kerX[xx] * srcPtr[x + xx + ii];
			//		__m128d mmx2 = _mm_set1_pd(kerX[xx]);
			//		__m128d mmx3 = _mm_set_pd(srcPtr[x + ii + xx][1], srcPtr[x + ii + xx][0]);
			//		__m128d mmx4 = _mm_set1_pd(srcPtr[x + ii + xx][2]);
			//		mmx4 = _mm_mul_pd(mmx4, mmx2);
			//		mmx3 = _mm_mul_pd(mmx3, mmx2);
			//		mmx0 = _mm_add_pd(mmx0, mmx3);
			//		mmx1 = _mm_add_pd(mmx1, mmx4);
				}
			//	_mm_storeu_pd(val, mmx0);
			//	_mm_storeu_pd(val + 2, mmx1);
			//	dstPtr[x + ii] = Vec3d(val[0], val[1], val[2]);
//				dstPtr[x] = Vec3d(val0, val1, val2);
			}
		//	int len = min(chunk_size, w - x);
		//	memcpy(&dstPtr[x], &row[0], sizeof(Vec3d) * len);
			for(int ii = 0; ii < chunk_size && x + ii < w; ii++) {
				dstPtr[x + ii] = row[ii];
			}
		}
	}

	#ifdef _OPENMP
	#pragma omp for schedule(dynamic)
	#endif
	for(int y = 0; y < h; y++) {
		vector<Vec3d> row(w, Vec3d(0.0, 0.0, 0.0));
		Vec3d * srcPtr = dstBuff.ptr<Vec3d>(y);
		for(int yy = 0; yy < ky; yy++, srcPtr += w) {
			double val = kerY[yy];
		//	__m128d mmx0 = _mm_set1_pd(kerY[yy]);		
			for(int x = 0; x < w; x++) {
		//		double val[2] = {0.0};
		//		__m128d mmx1 = _mm_loadu_pd((double*)(&srcPtr[x][0]));
		//		__m128d mmx2 = _mm_set1_pd(srcPtr[x][2]);
		//		__m128d mmx3 = _mm_loadu_pd((double*)(&row[x][0]));
		//		__m128d mmx4 = _mm_set1_pd(row[x][2]);
		//		mmx1 = _mm_mul_pd(mmx1, mmx0);
		//		mmx2 = _mm_mul_pd(mmx2, mmx0);
		//		mmx3 = _mm_add_pd(mmx3, mmx1);
		//		mmx4 = _mm_add_pd(mmx4, mmx2);
		//		_mm_storeu_pd((double*)(&row[x][0]), mmx3);
		//		_mm_storeu_pd(val, mmx4);
		//		row[x][2] = val[0];
				row[x] += srcPtr[x] * val;
			}
		}
		for(int x = 0; x < w; x++) {
			double val0 = row[x][0];
			double val1 = row[x][1];
			double val2 = row[x][2];
			val0 = val0 < 0 ? 0 : val0; val0 = val0 > 0xff ? 0xff : val0;
			val1 = val1 < 0 ? 0 : val1; val1 = val1 > 0xff ? 0xff : val1;
			val2 = val2 < 0 ? 0 : val2; val2 = val2 > 0xff ? 0xff : val2;
			dst.at<Vec3b>(y, x) = Vec3b(static_cast<uchar>(val0), static_cast<uchar>(val1), static_cast<uchar>(val2));
//			dst.at<Vec3b>(y, x) = Vec3b(row[x]);
		}
	}
	}
	t1 += omp_get_wtime() - t0;
	cout << t1 << "\n";
	
	free(kerX);
	free(kerY);

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
	const int threadNum = 8;
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

void myGaussianBlur_SIFT(const Mat& src, Mat& dst, Size ksize, double sigmaX, double sigmaY)
{
	// get gaussian kernel
	int kx = ksize.width;
	int ky = ksize.height;
	kx = static_cast<int>(6.0 * sigmaX) + 1; kx = kx % 2 == 0 ? kx + 1 : kx;
	ky = static_cast<int>(6.0 * sigmaY) + 1; ky = ky % 2 == 0 ? ky + 1 : ky;
	assert(kx % 2 == 1 && ky % 2 == 1);
	int radiusX = kx / 2;
	int radiusY = ky / 2;
	kx = radiusX + 1;
	ky = radiusY + 1;
	double kerY[ky];
	double kerX[kx];
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
	int h = src.rows;
	int w = src.cols;
	int hh = h + 2 * radiusY;
	int ww = w + 2 * radiusX;
	Mat filBuff(hh, ww, CV_8UC1);
	Mat dstBuff(hh, w, CV_64FC1);
	const int threadNum = 8;
	double t0 = 0.0;
	double t1 = 0.0;
	for(int y = radiusY; y < hh - radiusY; y++) {
		uchar * bufPtr = filBuff.ptr<uchar>(y) + radiusX;
		const uchar * srcPtr = src.ptr<uchar>(y - radiusY);
		memcpy(bufPtr, srcPtr, sizeof(uchar) * w);
	}
	
	// left & right
	for(int y = radiusY; y < hh - radiusY; y++) {
		uchar * bufPtr = filBuff.ptr<uchar>(y);
		for(int x = 0; x < radiusX; x++) {
			bufPtr[x] = bufPtr[radiusX];
			bufPtr[x + ww - radiusX] = bufPtr[ww - radiusX - 1];
		}
	}

	// top & bottom
	uchar * topPtr = filBuff.ptr<uchar>(0);
	uchar * botPtr = filBuff.ptr<uchar>(hh - radiusY);
	uchar * topLin = filBuff.ptr<uchar>(radiusY);
	uchar * botLin = filBuff.ptr<uchar>(hh - radiusY - 1);
	for(int y = 0; y < radiusY; y++, topPtr += ww, botPtr += ww) {
		memcpy(topPtr, topLin, sizeof(uchar) * ww);
		memcpy(botPtr, botLin, sizeof(uchar) * ww);
	}

	t0 = omp_get_wtime();
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
		uchar * srcPtr = filBuff.ptr<uchar>(y) + radiusX;
		double * dstPtr = dstBuff.ptr<double>(y);
		for(int x = 0; x < w; x++) {
			dstPtr[x] = kerX[0] * srcPtr[x];
			for(int xx = 1; xx < kx; xx++) {
				dstPtr[x] += kerX[xx] * (srcPtr[x + xx] + srcPtr[x - xx]);
			}
		}
	}
	
	// column filter
	#ifdef _OPENMP
	#pragma omp for schedule(static, h / threadNum)
	#endif
	for(int y = 0; y < h; y++) {
		double * srcPtr = dstBuff.ptr<double>(y + radiusY);
		uchar * dstPtr = dst.ptr<uchar>(y);
		for(int x = 0; x < w; x++) {
			double val = kerY[0] * srcPtr[x];		
			for(int yy = 1; yy < ky; yy++) {
				val += kerY[yy] * (*(srcPtr + yy * w + x) + *(srcPtr - yy * w + x));
			}
			val = val < 0 ? 0 : val; val = val > 0xff ? 0xff : val;
			dstPtr[x] = static_cast<uchar>(val);
		}
	}
	}
	t1 += omp_get_wtime() - t0;
//	cout << t1 << "\n";

}

void resize(const Mat& src, Mat& dst)
{
	int hSrc = src.rows;
	int wSrc = src.cols;
	int hDst = dst.rows;
	int wDst = dst.cols;
	double dhDst = (hSrc - 1.0) / (hDst - 1.0);
	double dwDst = (wSrc - 1.0) / (wDst - 1.0);

	const int threadNum = 8;
	
	#ifdef _OPENMP
	#pragma omp parallel for num_threads(threadNum) shared(dst, src, hSrc, wSrc, hDst, wDst, dhDst, dwDst) schedule(static, hDst / threadNum)
	#endif
	for(int y = 0; y < hDst; y++) {
		Vec3b * dstPtr = dst.ptr<Vec3b>(y);
		for(int x = 0; x < wDst; x++) {
			double locX = x * dwDst;
			double locY = y * dhDst;
			int xSrc = static_cast<int>(locX);
			int ySrc = static_cast<int>(locY);
			int x1 = xSrc + 1; x1 = x1 >= wSrc ? wSrc - 1 : x1;
			int y1 = ySrc + 1; y1 = y1 >= hSrc ? hSrc - 1 : y1;
			double fx = locX - xSrc;
			double fy = locY - ySrc;

			double w00 = (1.0 - fx) * (1.0 - fx);
			double w01 = (1.0 - fx) * (      fy);
			double w10 = (      fx) * (1.0 - fy);
			double w11 = (      fx) * (      fy);			

			Vec3d vec00 = Vec3d(src.at<Vec3b>(ySrc, xSrc));
			Vec3d vec01 = Vec3d(src.at<Vec3b>(y1, xSrc));
			Vec3d vec10 = Vec3d(src.at<Vec3b>(ySrc, x1));
			Vec3d vec11 = Vec3d(src.at<Vec3b>(y1, x1));
			Vec3d vecDst = vec00 * w00 + vec01 * w01 + vec10 * w10 + vec11 * w11;
			dstPtr[x] = Vec3b(vecDst);
		}
	}	
}

void myResize_SIFT(const Mat& src, Mat& dst)
{
	int hSrc = src.rows;
	int wSrc = src.cols;
	int hDst = dst.rows;
	int wDst = dst.cols;
	double dhDst = (hSrc - 1.0) / (hDst - 1.0);
	double dwDst = (wSrc - 1.0) / (wDst - 1.0);

	const int threadNum = 8;
	
	#ifdef _OPENMP
	#pragma omp parallel for num_threads(threadNum) shared(dst, src, hSrc, wSrc, hDst, wDst, dhDst, dwDst) schedule(static, hDst / threadNum)
	#endif
	for(int y = 0; y < hDst; y++) {
		uchar * dstPtr = dst.ptr<uchar>(y);
		for(int x = 0; x < wDst; x++) {
			double locX = x * dwDst;
			double locY = y * dhDst;
			int xSrc = static_cast<int>(locX);
			int ySrc = static_cast<int>(locY);
			int x1 = xSrc + 1; x1 = x1 >= wSrc ? wSrc - 1 : x1;
			int y1 = ySrc + 1; y1 = y1 >= hSrc ? hSrc - 1 : y1;
			double fx = locX - xSrc;
			double fy = locY - ySrc;

			double w00 = (1.0 - fx) * (1.0 - fx);
			double w01 = (1.0 - fx) * (      fy);
			double w10 = (      fx) * (1.0 - fy);
			double w11 = (      fx) * (      fy);			

			double val00 = src.at<uchar>(ySrc, xSrc);
			double val01 = src.at<uchar>(y1, xSrc);
			double val10 = src.at<uchar>(ySrc, x1);
			double val11 = src.at<uchar>(y1, x1);
			double val = val00 * w00 + val01 * w01 + val10 * w10 + val11 * w11;
			dstPtr[x] = saturate_cast<uchar>(val); 
		}
	}	
}

void myGradient_SIFT(const Mat& mat1, const Mat& mat2, Mat& sub, double step)
{
	int rows = sub.rows;
	int cols = sub.cols;
	
	const int threadNum = 8;
	#ifdef _OPENMP
	#pragma omp parallel for num_threads(threadNum) shared(rows, cols, mat1, mat2, sub, step) schedule(static, rows / threadNum)
	#endif
	for(int y = 0; y < rows; y++) {
		const uchar * ptr1 = mat1.ptr<uchar>(y);
		const uchar * ptr2 = mat2.ptr<uchar>(y);
		uchar * dst = sub.ptr<uchar>(y);
		for(int x = 0; x < cols; x++) {
			double val = (ptr2[x] - ptr1[x]) / step;
			dst[x] = saturate_cast<uchar>(val);
		}
	}
}

void mySIFT(const Mat& src, int minH, int minW, int s, double sigma0, vector<Vec2i>& keyPoints)
{
	Mat srcSIFT(src.size(), CV_8U);
	cvtColor(src, srcSIFT, CV_BGR2GRAY);

	vector<Size> octaves;
	int h = src.rows;
	int w = src.cols;
	octaves.push_back(Size(2 * w, 2 * h));
	int hh = h;
	int ww = w;
	while(hh >= minH && ww >= minW) {
		octaves.push_back(Size(ww, hh));
		ww = ww / 2; hh = hh / 2;
	}
	fprintf(stdout, "INFO: octaves' size ");
	for(unsigned int i = 0; i < octaves.size(); i++) {
		fprintf(stdout, "( %d, %d ), ", octaves[i].height, octaves[i].width);

	}
	fprintf(stdout, "\n");	

	int nocts = octaves.size();
	int ndogs = nocts * ( s + 2 );
	
	vector<Mat> octaveImg(ndogs, Mat());
	for(int im = 0; im < ndogs; im++) {
		int iSize = im / (s + 2);
		octaveImg[im] = Mat(octaves[iSize], CV_8U);
	}
	
	double k = pow(2.0, 1.0 / s);
	int m = 1;
	double scale = sqrt(k * k - 1.0);
	Mat imgUpper;
	Size size = Size(9, 9);
	double t0 = omp_get_wtime();
	// process octave 0
	{
		double sigma = sigma0 * m;
		double deltaSigma = sigma * scale;
		fprintf(stderr, "INFO: o = 0, s = 0, delta sigma = %f\n", deltaSigma);
		Mat img0(octaves[0], CV_8U);
		Mat img1(octaves[0], CV_8U);
		myResize_SIFT(srcSIFT, img0);
		myGaussianBlur_SIFT(img0, img1, size, deltaSigma, deltaSigma);
//		GaussianBlur(img0, img1, size, deltaSigma, deltaSigma);
		octaveImg[0] = img1 - img0;
	//	myGradient_SIFT(img1, img0, octaveImg[0], (k - 1.0) * sigma);
//		imwrite("m1.jpg", img0);
//		imwrite("m2.jpg", img1);
		Mat * img0ptr = &img1;
		Mat * img1ptr = &img0;
		sigma *= k;
		for(int is = 1; is < s + 2; is++, sigma *= k) {
			deltaSigma = sigma * scale;
			fprintf(stderr, "INFO: o = 0, s = %d, delta sigma = %f\n", is + 1, deltaSigma);
			myGaussianBlur_SIFT(*img0ptr, *img1ptr, size, deltaSigma, deltaSigma);
			octaveImg[is] = *img1ptr - *img0ptr;
	//		myGradient_SIFT(*img1ptr, *img0ptr, octaveImg[is], (k - 1.0) * sigma);
			{
				Mat * swapPtr = img0ptr;
				img0ptr = img1ptr;
				img1ptr = swapPtr;
			}
			if(is == s - 1) imgUpper = img1ptr->clone();
		}
	}

	m *= 2;

	// process octave 1 --> nocts
	for(int o = 1; o < nocts; o++, m *= 2) {
		double sigma = sigma0 * m;
		double deltaSigma = sigma * scale;
		fprintf(stderr, "INFO: o = %d, s = 0, delta sigma = %f\n", o, deltaSigma);
		Mat img0(octaves[o], CV_8U);
		Mat img1(octaves[o], CV_8U);
		myResize_SIFT(imgUpper, img0);
		myGaussianBlur_SIFT(img0, img1, size, deltaSigma, deltaSigma);
		octaveImg[o * (s + 2)] = img1 - img0;
	//	myGradient_SIFT(img1, img0, octaveImg[o * (s + 2)], (k - 1.0) * sigma);
		Mat * img0ptr = &img1;
		Mat * img1ptr = &img0;
		sigma *= k;
		for(int is = 1; is < s + 2; is++, sigma *= k) {
			deltaSigma = sigma * scale;
			fprintf(stderr, "INFO: o = %d, s = %d, delta sigma = %f\n", o, is + 1, deltaSigma);
			myGaussianBlur_SIFT(*img0ptr, *img1ptr, size, deltaSigma, deltaSigma);
			octaveImg[o * (s + 2) + is] = *img1ptr - *img0ptr;
		//	myGradient_SIFT(*img1ptr, *img0ptr, octaveImg[o * (s + 2) + is], (k - 1.0) * sigma);
			{
				Mat * swapPtr = img0ptr;
				img0ptr = img1ptr;
				img1ptr = swapPtr;
			}
			if(is == s - 1) imgUpper = img1ptr->clone();
		}
	}

	t0 = omp_get_wtime() - t0;
	fprintf(stdout, "INFO: Time of DOG is %f s\n", t0);

	for(int i = 0; i < ndogs; i++) {
		char filename[256];
		int o = i / (s + 2);
		int is = i % (s + 2) + 1;
		sprintf(filename, "octave_img_o_%02d_s_%02d.jpg", o, is);
		imwrite(filename, octaveImg[i]);
	}

	for(int o = 0; o < nocts; o++) {
		int hh = octaves[o].height;
		int ww = octaves[o].width;
		double scale = (1.0 * h) / hh;
		for(int y = 1; y < hh - 1; y++) {
			for(int x = 1; x < ww - 1; x++) {
				uchar curMax = 0;
				uchar prevMax = 0;
				uchar nextMax = 0;
				uchar curMin = 255;
				uchar prevMin = 255;
				uchar nextMin = 255;
				Mat& mat0 = octaveImg[o * (s + 2) + 0];
				Mat& mat1 = octaveImg[o * (s + 2) + 1];
				for(int yy = -1; yy <=1; yy++) {
					for(int xx = -1; xx <=1; xx++) {
						curMax = std::max(mat1.at<uchar>(y + yy, x + xx), curMax);
						prevMax = std::max(mat0.at<uchar>(y + yy, x + xx), prevMax);
						curMin = std::min(mat1.at<uchar>(y + yy, x + xx), curMin);
						prevMin = std::min(mat0.at<uchar>(y + yy, x + xx), prevMin);
					}
				}
				for(int is = 1; is < s + 1; is++) {
					Mat& curImg = octaveImg[o * (s + 2) + is];
					Mat& nextImg = octaveImg[o * (s + 2) + is + 1];
					uchar loc = curImg.at<uchar>(y, x);
					nextMax = 0;
					nextMin = 255;
					for(int yy = -1; yy <=1; yy++) {
						for(int xx = -1; xx <=1; xx++) {
							nextMax = std::max(nextImg.at<uchar>(y + yy, x + xx), nextMax);
							nextMin = std::min(nextImg.at<uchar>(y + yy, x + xx), nextMin);
						}
					}
					if((curMin == curMax) || (prevMin == prevMax) || (nextMin == nextMax)) continue;
					if((loc <= prevMin && loc <= curMin && loc <= nextMin)
					|| (loc >= prevMax && loc >= curMax && loc >= nextMax)) {
						double hess[3];
						double grad[2];
						
						keyPoints.push_back(Vec2i(scale * x, scale * y));	
					}
					prevMin = curMin;
					curMin = nextMin;
					prevMax = curMax;
					curMax = nextMax;
				}
			}		
		}
	}
//	fprintf(stdout, "INFO: key points ");
//	for(unsigned int i = 0; i < keyPoints.size(); i++) {
//		fprintf(stdout, "( %d, %d ), ", keyPoints[i][0], keyPoints[i][1]);
//	}
//	fprintf(stdout, "\n");
	cout << keyPoints.size() << "\n";
}

int main(int argc, char * argv[])
{
	if(argc != 3) {
		fprintf(stdout, "Usage: inputfile outputfile\n");
		return -1;
	}

	const char * imagename = argv[1];
	const char * outputimage = argv[2];
	Mat img = imread(imagename);
	if(img.empty()) {
		fprintf(stderr, "ERROR: cannot load image %s\n", imagename);
		return -1;
	}
	
	if( !img.data )
		return -1;

	resize(img, img, Size(5120, 5120));

	cout << img.size() << "\n";

	cout << img.type() << "\n";

	Mat dst(img.size(), img.type());

	clock_t beg, end;

	beg = clock();
	double t;
	t = omp_get_wtime();
	GaussianBlur(img, dst, Size(101, 101), 1, 1, BORDER_REPLICATE);
//	dst = dst - img;
	t = omp_get_wtime() - t;
	end = clock();
	cout << "clock cycles of opencv::GaussianBlur(): " << static_cast<float>(end - beg) << "\n";
	cout << "elapsed time of opencv::GaussianBlur(): " << t << "s\n";

	beg = clock();
	t = omp_get_wtime();
	Mat dst1(img.size(), img.type());
	myGaussianBlur_(img, dst1, Size(101, 101), 1, 1);
//	dst1 -= img;
	t = omp_get_wtime() - t;
	end = clock();
	cout << "clock cycles of myGaussianBlur(): " << static_cast<float>(end - beg) << "\n";
	cout << "elapsed time of myGaussianBlur(): " << t << "s\n";
	imwrite("mygaussian.jpg", dst1);

	imwrite(outputimage, dst);

	imwrite("error.jpg", dst1 - dst);

	cout << dst.size() << "\n";
	
//	int h = img.rows;
//	int w = img.cols;
//	
//	Mat dst2(0.5 * h, 0.5 * w, img.type());
//	resize(img, dst2);
//	imwrite("myresize.jpg", dst2);
//
//	vector<Vec2i> keyPoints;
//	mySIFT(img, 16, 16, 3, 2.0, keyPoints);
//
//	Scalar color(0, 255, 0);
//	for(auto i = 0; i < keyPoints.size(); i++) {
//		circle(img, keyPoints[i], 1.5, color); 
//	}
//
//		
//	imshow("lena", img);
//	waitKey();
	
	return 0;	
}
