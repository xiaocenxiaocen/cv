# Location of the CUDA Toolkit
CUDA_PATH = /home/xiaocen/Software/cuda/cuda-8.0
CC = gcc -O3 -g -Wall -std=c99
CXX = g++ -O3 -g -Wall -std=c++0x -Wno-deprecated

NVCC = nvcc -ccbin g++ -Xcompiler -fopenmp

#NVCC = nvcc -ccbin icc -Xcompiler -openmp

CUDA_INCLUDE = $(CUDA_PATH)/include
CUDA_COMMON_INCLUDE = /home/xiaocen/Software/cuda/samples/NVIDIA_CUDA-8.0_Samples/common/inc

OPENCV_PATH = /home/xiaocen/Software/opencv
OPENCV_INCLUDE = $(OPENCV_PATH)/include

INCLUDES = -I$(CUDA_COMMON_INCLUDE) -I$(CUDA_INCLUDE) -I$(OPENCV_INCLUDE)

GENCODE_FLAGS = -m64 -gencode arch=compute_50,code=sm_50
CUDA_FLAGS = --ptxas-options=-v
CFLAGS = $(CUDA_FLAGS)

CXXFLAGS = -fopenmp -D_OPENMP -D_SSE2 $(INCLUDES)

LIBRARIES = -L$(OPENCV_PATH)/lib -L$(CUDA_PATH)/lib64

LDFLAGS = -lm -lpthread -lcudart -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_ml -lopencv_nonfree -lopencv_objdetect -lopencv_ocl -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_ts -lopencv_video -lopencv_videostab

all: target

target: get_gpu_info gauss_blur

get_gpu_info: get_gpu_info.o
	$(NVCC) $(LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

get_gpu_info.o: get_gpu_info.cpp
	$(NVCC) $(INCLUDES) $(CFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

gaussian_blur_filter.o: gaussian_blur_filter.cu
	$(NVCC) $(INCLUDES) $(CFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

gauss_blur: gaussian_blur_filter.o gaussian_blur.o
	$(CXX) -o $@ -fopenmp $+ $(LDFLAGS) $(LIBRARIES)

.cpp.o:
	$(CXX) -c $(CXXFLAGS) $<

.c.o:
	$(CC) -c $(CXXFLAGS) $<


.PHONY: clean
clean:
	-rm *.o
	-rm get_gpu_info
	-rm gauss_blur
