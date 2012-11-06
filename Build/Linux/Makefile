all:	GPUBenchmark

Main.o:	Main.cpp
	nvcc -c Main.cpp -Xcompiler "-fPIC -O2 -g"
    
CTools.o:	Common/CTools.cpp Common/CTools.h
	nvcc -c Common/CTools.cpp -D CUDA50 -Xcompiler "-fPIC -O2 -g"

CUDATools.o:	Common/CUDATools.cpp Common/CUDATools.h
	nvcc -c Common/CUDATools.cpp -D CUDA50 -Xcompiler "-fPIC -O2 -g"

CUDATools_cu.o:	Common/CUDATools.cu Common/CUDATools.h
	nvcc -c -dc -o CUDATools_cu.o Common/CUDATools.cu -use_fast_math -D CUDA50 -Xcompiler "-fPIC -O2 -g" -gencode=arch=compute_35,code=\"sm_35,compute_35\"

GPUBenchmark.o: GPUBenchmark.cpp GPUBenchmark.h
	nvcc -c GPUBenchmark.cpp -D CUDA50 -Xcompiler "-fPIC -O2 -g"

GPUBenchmark_cu.o: GPUBenchmark.cu GPUBenchmark.h
	nvcc -c -dc -o GPUBenchmark_cu.o GPUBenchmark.cu -use_fast_math -D CUDA50 -Xcompiler "-fPIC -O2 -g" -gencode=arch=compute_35,code=\"sm_35,compute_35\"

GPUBenchmark: Main.o GPUBenchmark.o GPUBenchmark_cu.o CTools.o CUDATools.o CUDATools_cu.o
	nvcc -arch=sm_35 -rdc=true -o $@ Main.o GPUBenchmark.o GPUBenchmark_cu.o CTools.o CUDATools.o CUDATools_cu.o -lcuda -lcudart -lcudadevrt -lcurand

clean:
	rm *.o GPUBenchmark
	chmod 644 *.h *.cpp *.cu Makefile