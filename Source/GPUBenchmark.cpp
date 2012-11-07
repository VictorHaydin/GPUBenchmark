#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include "Common/CTools.h"
#include "Common/CUDATools.h"
#include "GPUBenchmark.h"

#define MAX_TIMING_COUNT	5

double GPUBenchmark::execOuterTiming(benchmarkOuterFunc func, int p1, int p2, int p3, int p4, int p5)
{
	TimingCounter counter;
	double minTime = 1E10;
	for (int i = 0; i < MAX_TIMING_COUNT; i++)
	{
		TimingClearAndStart(counter);
		(this->*func)(p1, p2, p3, p4, p5);
		cudaSafeCall(cudaDeviceSynchronize());
		TimingFinish(counter);
		minTime = min(minTime, TimingSeconds(counter));
	}
	return minTime;
}

double GPUBenchmark::execInnerTiming(benchmarkInnerFunc func, int p1, int p2, int p3, int p4, int p5)
{
	TimingCounter counter;
	double minTime = 1E10;
	for (int i = 0; i < MAX_TIMING_COUNT; i++)
	{
		double time = (this->*func)(p1, p2, p3, p4, p5);
		minTime = min(minTime, time);
	}
	cudaSafeCall(cudaDeviceSynchronize());
	return minTime;
}

void GPUBenchmark::deviceMemAllocRelease(int size, int repeatCount, int, int, int)
{
	deviceMem<byte> dmem;
	for (int i = 0; i < repeatCount; i++)
	{
		dmem.allocate(size);
		dmem.release();
	}
}

void GPUBenchmark::mappedMemAllocRelease(int size, int repeatCount, int, int, int)
{
	mappedMem<byte> mmem;
	for (int i = 0; i < repeatCount; i++)
	{
		mmem.allocate(size);
		mmem.release();
	}
}

void GPUBenchmark::hostMemWriteCombinedAllocRelease(int size, int repeatCount, int, int, int)
{
	hostMem<byte> hmem;
	for (int i = 0; i < repeatCount; i++)
	{
		hmem.allocate(size, cudaHostAllocWriteCombined);
		hmem.release();
	}
}

void GPUBenchmark::hostMemRegisterUnregister(int size, int repeatCount, int, int, int)
{
	hostMem<byte> hmem(size);
	mappedMem<byte> mmem;
	for (int i = 0; i < repeatCount; i++)
	{
		mmem.registerHost(hmem.hptr, size);
		mmem.release();
	}
}
/*
double GPUBenchmark::hostMemTransfer(int size, int streamCount, int mode, int direction, int)
{
	streamSize = size / streamCount;
	hostMem<deviceMem<byte> > dmem(streamCount);
	hostMem<deviceMem<byte> > dmem2(streamCount);
	hostMem<hostMem<byte> > hmem(streamCount);
	hostMem<cudaStream_t> streams(streamCount);
	for (int i = 0; i < streamCount; i++)
	{
		dmem[i].allocate(streamSize);
		if (direction == 2)
			dmem2[i].allocate(streamSize);
		switch (mode)
		{
		case 0:
			hmem[i].allocate(streamSize);
			break;
		case 1:
			hmem[i].allocate(streamSize, cudaHostAllocDefault);
			break;
		case 2:
			hmem[i].allocate(streamSize, cudaHostAllocWriteCombined);
			break;
		default:
			assert(false);
		}
		hmem[i].copyTo(dmem[i]); // device memory initialization
		cudaSafeCall(cudaStreamCreate(&streams[i]));
	}
	TimingCounter counter;
	TimingClearAndStart(counter);
	for (int i = 0; i < streamCount; i++)
		switch (direction)
		{
		case 0:
			hmem[i].copyToAsync(dmem[i], streams[i]);
			break;
		case 1:
			dmem[i].copyToAsync(hmem[i], streams[i]);
			break;
		case 2:
			dmem[i].copyToAsync(dmem2[i], streams[i]);
			break;
		default:
			assert(false);
		}
	cudaSafeCall(cudaDeviceSynchronize());
	TimingFinish(counter);

	for (int i = 0; i < streamCount; i++)
		cudaSafeCall(cudaStreamDestroy(streams[i]));

	return TimingSeconds(counter);
}
*/
double GPUBenchmark::memTransfer(int size, int mode, int direction, int, int)
{
	const int ITERATIONS = 10;
	deviceMem<byte> dmem(size);
	deviceMem<byte> dmem2(size);
	hostMem<byte> hmem(size);
	dmem.allocate(size);
	if (direction == 2)
		dmem2.allocate(size);
	switch (mode)
	{
	case 0:
		hmem.allocate(size);
		break;
	case 1:
		hmem.allocate(size, cudaHostAllocDefault);
		break;
	case 2:
		hmem.allocate(size, cudaHostAllocWriteCombined);
		break;
	default:
		assert(false);
	}
	hmem.copyTo(dmem); // device memory initialization
	TimingCounter counter;
	TimingClearAndStart(counter);
	for (int i = 0; i < ITERATIONS; i++)
		switch (direction)
		{
		case 0:
			hmem.copyTo(dmem);
			break;
		case 1:
			dmem.copyTo(hmem);
			break;
		case 2:
			dmem.copyTo(dmem2);
			break;
		default:
			assert(false);
		}
	cudaSafeCall(cudaDeviceSynchronize());
	TimingFinish(counter);

	return TimingSeconds(counter) / ITERATIONS;
}

double GPUBenchmark::kernelExecuteTinyTask(int blockCount, int threadCount, int, int, int)
{
	const int ITERATIONS = 1; //1000;
	TimingCounter counter;
	TimingClearAndStart(counter);
	for(int i = 0; i < ITERATIONS; i++)
	{
		cuda_doTinyTask(blockCount, threadCount);
		cudaSafeCall(cudaDeviceSynchronize());
	}
	TimingFinish(counter);

	return TimingSeconds(counter) / ITERATIONS;
}

double GPUBenchmark::kernelScheduleTinyTask(int blockCount, int threadCount, int, int, int)
{
	const int ITERATIONS = 1; //1000;
	TimingCounter counter;
	TimingClearAndStart(counter);
	for(int i = 0; i < ITERATIONS; i++)
		cuda_doTinyTask(blockCount, threadCount);
	TimingFinish(counter);

	return TimingSeconds(counter) / ITERATIONS;
}

template<typename T>
void GPUBenchmark::kernelAdd(int count, int blockCount, int threadCount, int, int)
{
	cuda_doAdd<T>(count, blockCount, threadCount);
	cudaSafeCall(cudaDeviceSynchronize());
}

template<typename T>
void GPUBenchmark::kernelAdd2(int count, int blockCount, int threadCount, int, int)
{
	cuda_doAdd2<T>(count, blockCount, threadCount);
	cudaSafeCall(cudaDeviceSynchronize());
}

template<typename T>
void GPUBenchmark::kernelAddMulMix(int count, int blockCount, int threadCount, int, int)
{
	cuda_doAddMulMix<T>(count, blockCount, threadCount);
	cudaSafeCall(cudaDeviceSynchronize());
}

template<typename T>
void GPUBenchmark::kernelMul(int count, int blockCount, int threadCount, int, int)
{
	cuda_doMul<T>(count, blockCount, threadCount);
	cudaSafeCall(cudaDeviceSynchronize());
}

template<typename T>
void GPUBenchmark::kernelDiv(int count, int blockCount, int threadCount, int, int)
{
	cuda_doDiv<T>(count, blockCount, threadCount);
	cudaSafeCall(cudaDeviceSynchronize());
}

template<typename T>
void GPUBenchmark::kernelSin(int count, int blockCount, int threadCount, int, int)
{
	cuda_doSin<T>(count, blockCount, threadCount);
	cudaSafeCall(cudaDeviceSynchronize());
}

#ifdef CUDA50
double GPUBenchmark::kernelDynamicExecuteTinyTask(int blockCount, int threadCount, int, int, int)
{
	return (cuda_doDynamicTinyTask(blockCount, threadCount, 1) / deviceClockRate);
}

double GPUBenchmark::kernelDynamicScheduleTinyTask(int blockCount, int threadCount, int, int, int)
{
	return (cuda_doDynamicTinyTask(blockCount, threadCount, 0) / deviceClockRate);
}
#endif

template<typename T> 
double GPUBenchmark::deviceMemAccess(int count, int repeatCount, int blockCount, int threadsPerBlock, int memAccess)
{
	TimingCounter counter;
	deviceMem<T> dData;
	mappedMem<T> mData;
	T *dptr;
	switch (memAccess)
	{
	case maPinnedAlignedRead:
	case maPinnedAlignedWrite:
	case maPinnedNotAlignedRead:
	case maPinnedNotAlignedWrite:
		mData.allocate(count);
		memset(mData.hptr, 0, count * sizeof(T));
		dptr = mData.dptr;
		break;
	default:
		dData.allocate(count);
		dData.clear();
		dptr = dData.dptr;
	}
	cudaSafeCall(cudaDeviceSynchronize());

	TimingClearAndStart(counter);
	switch (memAccess)
	{
	case maAlignedRead:
	case maPinnedAlignedRead:
		cuda_alignedRead<T>(dptr, count, repeatCount, blockCount, threadsPerBlock);
		break;
	case maAlignedWrite:
	case maPinnedAlignedWrite:
		cuda_alignedWrite<T>(dptr, count, repeatCount, blockCount, threadsPerBlock);
		break;
	case maNotAlignedRead:
	case maPinnedNotAlignedRead:
		cuda_notAlignedRead<T>(dptr, count, repeatCount, blockCount, threadsPerBlock);
		break;
	case maNotAlignedWrite:
	case maPinnedNotAlignedWrite:
		cuda_notAlignedWrite<T>(dptr, count, repeatCount, blockCount, threadsPerBlock);
		break;
	}
	cudaSafeCall(cudaDeviceSynchronize());
	TimingFinish(counter);

	return TimingSeconds(counter) / repeatCount;
}

template<typename T>
double GPUBenchmark::reductionSum(int count, int repeatCount, int blockCount, int threadsPerBlock, int)
{
	TimingCounter counter;
	deviceMem<T> dData;
	deviceMem<T> dSum;
	hostMem<T> hData;
	T sum;
	dData.allocate(count);
	dSum.allocate(1);
	hData.allocate(count);
	for (int i = 0; i < count; i++)
		hData[i] = (T)i;
	hData.copyTo(dData);

	TimingClearAndStart(counter);
	cuda_reductionSum<T>(dData.dptr, dSum.dptr, count, repeatCount, blockCount, threadsPerBlock);
	cudaSafeCall(cudaDeviceSynchronize());
	TimingFinish(counter);

	dSum.copyTo(&sum);
	double correctSum = ((double)(count - 1) / 2.0 * count);
	if (fabs(1.0 - double(sum) / correctSum) > 1e-3)
		printf("Reduction FAILED: Sum: %f, CorrectSum: %f\n", sum, correctSum);
	//assert(fabs(1.0 - double(sum) / ((double)(count - 1) / 2.0 * count)) < 1e-3);

	return TimingSeconds(counter) / repeatCount;
}

void GPUBenchmark::run()
{
	cudaDeviceProp deviceProps;
	int device, deviceCount;
	cudaSafeCall(cudaGetDeviceCount(&deviceCount));
	if (deviceCount > 1)
		cudaSafeCall(cudaSetDevice(1));
	cudaSafeCall(cudaGetDevice(&device));
	cudaSafeCall(cudaGetDeviceProperties(&deviceProps, device));
	deviceClockRate = deviceProps.clockRate * 1000;
	
	printf("\n");
	printf("GPU properties\n");
	printf("Name:                           %30s\n", deviceProps.name);
	printf("Clock rate:                     %30s\n", IntToStrF(deviceClockRate).c_str());
	printf("Memory clock rate:              %30s\n", IntToStrF((__int64)deviceProps.memoryClockRate * 1000).c_str());
	printf("Multiprocessors:                %30d\n", deviceProps.multiProcessorCount);
	printf("Maximum resident threads per multiprocessor:        %10d\n", deviceProps.maxThreadsPerMultiProcessor);

	printf("Version (compute capability):   %30s\n", (IntToStrF(deviceProps.major) + "." + IntToStrF(deviceProps.minor)).c_str());
	printf("Total global memory:            %30s\n", IntToStrF(deviceProps.totalGlobalMem).c_str());
	printf("Shared memory per Block:        %30s\n", IntToStrF(deviceProps.sharedMemPerBlock).c_str());
	printf("Registers per Block:            %30s\n", IntToStrF(deviceProps.regsPerBlock).c_str());
	printf("Warp size:                      %30s\n", IntToStrF(deviceProps.warpSize).c_str());
	printf("Mem pitch:                      %30s\n", IntToStrF(deviceProps.memPitch).c_str());
	printf("Max threads per block:          %30s\n", IntToStrF(deviceProps.maxThreadsPerBlock).c_str());
	printf("Max threads dimentions:    %35s\n", 
		(IntToStrF(deviceProps.maxThreadsDim[0]) + " x " + 
		IntToStrF(deviceProps.maxThreadsDim[1]) + " x " + 
		IntToStrF(deviceProps.maxThreadsDim[2])).c_str());
	printf("Max grid size:             %35s\n", 
		(IntToStrF(deviceProps.maxGridSize[0]) + " x " + 
		IntToStrF(deviceProps.maxGridSize[1]) + " x " + 
		IntToStrF(deviceProps.maxGridSize[2])).c_str());
	printf("Total const memory:             %30s\n", IntToStrF(deviceProps.totalConstMem).c_str());
	printf("Texture alignment:              %30s\n", IntToStrF(deviceProps.textureAlignment).c_str());
	printf("Texture pitch alignment:        %30s\n", IntToStrF(deviceProps.texturePitchAlignment).c_str());
	printf("Device overlap:                 %30s\n", BoolToStrYesNo(deviceProps.deviceOverlap > 0).c_str());
	printf("Kernel exec timeout enabled:    %30s\n", BoolToStrYesNo(deviceProps.kernelExecTimeoutEnabled > 0).c_str());
	printf("Integrated:                     %30s\n", BoolToStrYesNo(deviceProps.integrated > 0).c_str());
	printf("Can map host memory:            %30s\n", BoolToStrYesNo(deviceProps.canMapHostMemory > 0).c_str());
	printf("Compute mode:                   %30s\n", IntToStrF(deviceProps.computeMode).c_str());
	printf("Concurrent kernels:             %30s\n", BoolToStrYesNo(deviceProps.concurrentKernels > 0).c_str());
	printf("ECC enabled:                    %30s\n", BoolToStrYesNo(deviceProps.ECCEnabled > 0).c_str());
	printf("PCI bus ID:                     %30s\n", IntToStrF(deviceProps.pciBusID).c_str());
	printf("PCI device ID:                  %30s\n", IntToStrF(deviceProps.pciDeviceID).c_str());
	printf("PCI domain ID:                  %30s\n", IntToStrF(deviceProps.pciDomainID).c_str());
	printf("TCC driver:                     %30s\n", IntToStrF(deviceProps.tccDriver).c_str());
	printf("Async engine count:             %30s\n", IntToStrF(deviceProps.asyncEngineCount).c_str());
	printf("Unified addressing:             %30s\n", BoolToStrYesNo(deviceProps.unifiedAddressing > 0).c_str());
	printf("Memory bus width:               %30s\n", IntToStrF(deviceProps.memoryBusWidth).c_str());
	printf("L2 cache size:                  %30s\n", IntToStrF(deviceProps.l2CacheSize).c_str());
	printf("\n");

	printf("***** Host kernel schedule latencies, microseconds\n");
	printf("1x1:                                              %12.3f microsec\n", execInnerTiming(&GPUBenchmark::kernelScheduleTinyTask, 1, 1) * 1000000.0);
	printf("1x32:                                             %12.3f microsec\n", execInnerTiming(&GPUBenchmark::kernelScheduleTinyTask, 1, 32) * 1000000.0);
	printf("8x64:                                             %12.3f microsec\n", execInnerTiming(&GPUBenchmark::kernelScheduleTinyTask, 8, 64) * 1000000.0);
	printf("16384x128:                                        %12.3f microsec\n", execInnerTiming(&GPUBenchmark::kernelScheduleTinyTask, 16384, 128) * 1000000.0);
	printf("***** Host kernel execution latencies, microseconds\n");
	printf("1x1:                                              %12.3f microsec\n", execInnerTiming(&GPUBenchmark::kernelExecuteTinyTask, 1, 1) * 1000000.0);
	printf("1x32:                                             %12.3f microsec\n", execInnerTiming(&GPUBenchmark::kernelExecuteTinyTask, 1, 32) * 1000000.0);
	printf("8x64:                                             %12.3f microsec\n", execInnerTiming(&GPUBenchmark::kernelExecuteTinyTask, 8, 64) * 1000000.0);
	printf("16384x128:                                        %12.3f microsec\n", execInnerTiming(&GPUBenchmark::kernelExecuteTinyTask, 16384, 128) * 1000000.0);
	printf("\n");

#ifdef CUDA50
	printf("***** Device kernel schedule latencies\n");
	printf("1x1:                                              %12.3f microsec\n", execInnerTiming(&GPUBenchmark::kernelDynamicScheduleTinyTask, 1, 1) * 1000000.0);
	printf("1x32:                                             %12.3f microsec\n", execInnerTiming(&GPUBenchmark::kernelDynamicScheduleTinyTask, 1, 32) * 1000000.0);
	printf("8x64:                                             %12.3f microsec\n", execInnerTiming(&GPUBenchmark::kernelDynamicScheduleTinyTask, 8, 64) * 1000000.0);
	printf("16384x128:                                        %12.3f microsec\n", execInnerTiming(&GPUBenchmark::kernelDynamicScheduleTinyTask, 16384, 128) * 1000000.0);
	printf("***** Device kernel execution latencies\n");
	printf("1x1:                                              %12.3f microsec\n", execInnerTiming(&GPUBenchmark::kernelDynamicExecuteTinyTask, 1, 1) * 1000000.0);
	printf("1x32:                                             %12.3f microsec\n", execInnerTiming(&GPUBenchmark::kernelDynamicExecuteTinyTask, 1, 32) * 1000000.0);
	printf("8x64:                                             %12.3f microsec\n", execInnerTiming(&GPUBenchmark::kernelDynamicExecuteTinyTask, 8, 64) * 1000000.0);
	printf("16384x128:                                        %12.3f microsec\n", execInnerTiming(&GPUBenchmark::kernelDynamicExecuteTinyTask, 16384, 128) * 1000000.0);
	printf("\n");
#endif

	printf("***** Reduction time (SUM), microseconds\n");
	printf("1x32,    256 elements, int:                       %12.3f microsec\n", execInnerTiming(&GPUBenchmark::reductionSum<int>, 256, 10000, 1, 32) * 1000000.0);
	printf("1x32,    256 elements, float:                     %12.3f microsec\n", execInnerTiming(&GPUBenchmark::reductionSum<float>, 256, 10000, 1, 32) * 1000000.0);
	printf("1x32,    256 elements, double:                    %12.3f microsec\n", execInnerTiming(&GPUBenchmark::reductionSum<double>, 256, 10000, 1, 32) * 1000000.0);
	printf("128x64,  1M elements, int64:                      %12.3f microsec\n", execInnerTiming(&GPUBenchmark::reductionSum<__int64>, 1048576, 10, 128, 64) * 1000000.0);
	printf("128x64,  1M elements, float:                      %12.3f microsec\n", execInnerTiming(&GPUBenchmark::reductionSum<float>, 1048576, 10, 128, 64) * 1000000.0);
	printf("128x64,  1M elements, double:                     %12.3f microsec\n", execInnerTiming(&GPUBenchmark::reductionSum<double>, 1048576, 10, 128, 64) * 1000000.0);
	printf("128x256, 1M elements, int64:                      %12.3f microsec\n", execInnerTiming(&GPUBenchmark::reductionSum<__int64>, 1048576, 10, 128, 256) * 1000000.0);
	printf("128x256, 1M elements, float:                      %12.3f microsec\n", execInnerTiming(&GPUBenchmark::reductionSum<float>, 1048576, 10, 128, 256) * 1000000.0);
	printf("128x256, 1M elements, double:                     %12.3f microsec\n", execInnerTiming(&GPUBenchmark::reductionSum<double>, 1048576, 10, 128, 256) * 1000000.0);
	printf("512x512, 10M elements, int64:                     %12.3f microsec\n", execInnerTiming(&GPUBenchmark::reductionSum<__int64>, 10485760, 1, 512, 512) * 1000000.0);
	printf("512x512, 10M elements, float:                     %12.3f microsec\n", execInnerTiming(&GPUBenchmark::reductionSum<float>, 10485760, 1, 512, 512) * 1000000.0);
	printf("512x512, 10M elements, double:                    %12.3f microsec\n", execInnerTiming(&GPUBenchmark::reductionSum<double>, 10485760, 1, 512, 512) * 1000000.0);
	printf("\n");

	printf("***** Dependent FLOP, GFLOPs\n");
	printf("ADD (512x128, float):                             %12.3f GFLOPs\n", (double)16384 * 512 * 128 / execOuterTiming(&GPUBenchmark::kernelAdd<float>, 16384, 512, 128) / 1E9);
	printf("MUL (512x128, float):                             %12.3f GFLOPs\n", (double)16384 * 512 * 128 / execOuterTiming(&GPUBenchmark::kernelMul<float>, 16384, 512, 128) / 1E9);
	printf("DIV (512x128, float):                             %12.3f GFLOPs\n", (double)16384 * 512 * 128 / execOuterTiming(&GPUBenchmark::kernelDiv<float>, 16384, 512, 128) / 1E9);
	printf("SIN (512x128, float):                             %12.3f GFLOPs\n", (double)16384 * 512 * 128 / execOuterTiming(&GPUBenchmark::kernelSin<float>, 16384, 512, 128) / 1E9);
	printf("\n");
	printf("ADD (512x128, double):                            %12.3f GFLOPs\n", (double)16384 * 512 * 128 / execOuterTiming(&GPUBenchmark::kernelAdd<double>, 16384, 512, 128) / 1E9);
	printf("MUL (512x128, double):                            %12.3f GFLOPs\n", (double)16384 * 512 * 128 / execOuterTiming(&GPUBenchmark::kernelMul<double>, 16384, 512, 128) / 1E9);
	printf("DIV (512x128, double):                            %12.3f GFLOPs\n", (double)16384 * 512 * 128 / execOuterTiming(&GPUBenchmark::kernelDiv<double>, 16384, 512, 128) / 1E9);
	printf("SIN (512x128, double):                            %12.3f GFLOPs\n", (double)16384 * 512 * 128 / execOuterTiming(&GPUBenchmark::kernelSin<double>, 16384, 512, 128) / 1E9);
	printf("\n");
	printf("ADD (512x512, float):                             %12.3f GFLOPs\n", (double)16384 * 512 * 512 / execOuterTiming(&GPUBenchmark::kernelAdd<float>, 16384, 512, 512) / 1E9);
	printf("MUL (512x512, float):                             %12.3f GFLOPs\n", (double)16384 * 512 * 512 / execOuterTiming(&GPUBenchmark::kernelMul<float>, 16384, 512, 512) / 1E9);
	printf("DIV (512x512, float):                             %12.3f GFLOPs\n", (double)16384 * 512 * 512 / execOuterTiming(&GPUBenchmark::kernelDiv<float>, 16384, 512, 512) / 1E9);
	printf("SIN (512x512, float):                             %12.3f GFLOPs\n", (double)16384 * 512 * 512 / execOuterTiming(&GPUBenchmark::kernelSin<float>, 16384, 512, 512) / 1E9);
	printf("\n");
	printf("ADD (512x512, double):                            %12.3f GFLOPs\n", (double)16384 * 512 * 512 / execOuterTiming(&GPUBenchmark::kernelAdd<double>, 16384, 512, 512) / 1E9);
	printf("MUL (512x512, double):                            %12.3f GFLOPs\n", (double)16384 * 512 * 512 / execOuterTiming(&GPUBenchmark::kernelMul<double>, 16384, 512, 512) / 1E9);
	printf("DIV (512x512, double):                            %12.3f GFLOPs\n", (double)16384 * 512 * 512 / execOuterTiming(&GPUBenchmark::kernelDiv<double>, 16384, 512, 512) / 1E9);
	printf("SIN (512x512, double):                            %12.3f GFLOPs\n", (double)16384 * 512 * 512 / execOuterTiming(&GPUBenchmark::kernelSin<double>, 16384, 512, 512) / 1E9);
	printf("\n");

	printf("***** Independent FLOP, GFLOPs\n");
	printf("ADD (512x128, float):                             %12.3f GFLOPs\n", (double)65536 * 512 * 128 / execOuterTiming(&GPUBenchmark::kernelAdd2<float>, 65536, 512, 128) / 1E9);
	printf("ADD/MUL mix (512x128, float):                     %12.3f GFLOPs\n", (double)65536 * 512 * 128 / execOuterTiming(&GPUBenchmark::kernelAddMulMix<float>, 65536, 512, 128) / 1E9);
	printf("ADD (512x128, double):                            %12.3f GFLOPs\n", (double)65536 * 512 * 128 / execOuterTiming(&GPUBenchmark::kernelAdd2<double>, 65536, 512, 128) / 1E9);
	printf("ADD/MUL mix (512x128, double):                    %12.3f GFLOPs\n", (double)65536 * 512 * 128 / execOuterTiming(&GPUBenchmark::kernelAddMulMix<double>, 65536, 512, 128) / 1E9);
	printf("\n");
	printf("ADD (512x512, float):                             %12.3f GFLOPs\n", (double)65536 * 512 * 512 / execOuterTiming(&GPUBenchmark::kernelAdd2<float>, 65536, 512, 512) / 1E9);
	printf("ADD/MUL mix (512x512, float):                     %12.3f GFLOPs\n", (double)65536 * 512 * 512 / execOuterTiming(&GPUBenchmark::kernelAddMulMix<float>, 65536, 512, 512) / 1E9);
	printf("ADD (512x512, double):                            %12.3f GFLOPs\n", (double)65536 * 512 * 512 / execOuterTiming(&GPUBenchmark::kernelAdd2<double>, 65536, 512, 512) / 1E9);
	printf("ADD/MUL mix (512x512, double):                    %12.3f GFLOPs\n", (double)65536 * 512 * 512 / execOuterTiming(&GPUBenchmark::kernelAddMulMix<double>, 65536, 512, 512) / 1E9);
	printf("\n");

	printf("***** Memory management, milliseconds\n");
	printf("Device Memory allocate/release (16 bytes):        %12.3f millisec\n", execOuterTiming(&GPUBenchmark::deviceMemAllocRelease, 16, 10) / 10 * 1000);
	printf("Mapped Memory allocate/release (16 bytes):        %12.3f millisec\n", execOuterTiming(&GPUBenchmark::mappedMemAllocRelease, 16, 10) / 10 * 1000);
	printf("Host Memory register/unregister (16 bytes):       %12.3f millisec\n", execOuterTiming(&GPUBenchmark::hostMemRegisterUnregister, 16, 10) / 10 * 1000);
	printf("Host Write Combined allocate/release (16 bytes):  %12.3f millisec\n", execOuterTiming(&GPUBenchmark::hostMemWriteCombinedAllocRelease, 16, 10) / 10 * 1000);
	printf("\n");
	printf("Device Memory allocate/release (10M bytes):       %12.3f millisec\n", execOuterTiming(&GPUBenchmark::deviceMemAllocRelease, 10485760, 1) / 1 * 1000);
	printf("Mapped Memory allocate/release (10M bytes):       %12.3f millisec\n", execOuterTiming(&GPUBenchmark::mappedMemAllocRelease, 10485760, 1) / 1 * 1000);
	printf("Host Memory register/unregister (10M bytes):      %12.3f millisec\n", execOuterTiming(&GPUBenchmark::hostMemRegisterUnregister, 10485760, 1) / 1 * 1000);
	printf("Host Write Combined allocate/release (10M bytes): %12.3f millisec\n", execOuterTiming(&GPUBenchmark::hostMemWriteCombinedAllocRelease, 10485760, 1) / 1 * 1000);
	printf("\n");
	printf("Device Memory allocate/release (100M bytes):      %12.3f millisec\n", execOuterTiming(&GPUBenchmark::deviceMemAllocRelease, 104857600, 1) / 1 * 1000);
	printf("Mapped Memory allocate/release (100M bytes):      %12.3f millisec\n", execOuterTiming(&GPUBenchmark::mappedMemAllocRelease, 104857600, 1) / 1 * 1000);
	printf("Host Memory register/unregister (100M bytes):     %12.3f millisec\n", execOuterTiming(&GPUBenchmark::hostMemRegisterUnregister, 104857600, 1) / 1 * 1000);
	printf("Host Write Combined allocate/release (100M bytes):%12.3f millisec\n", execOuterTiming(&GPUBenchmark::hostMemWriteCombinedAllocRelease, 104857600, 1) / 1 * 1000);
	printf("\n");

	printf("***** Memory transfer speed (100MB blocks)\n");
	printf("Regular (Host to GPU):                            %12.3f GB/sec\n", (double)104857600.0 / execInnerTiming(&GPUBenchmark::memTransfer, 104857600, 0, 0) / 1024 / 1024 / 1024);
	printf("Page locked (Host to GPU):                        %12.3f GB/sec\n", (double)104857600.0 / execInnerTiming(&GPUBenchmark::memTransfer, 104857600, 1, 0) / 1024 / 1024 / 1024);
	printf("Write Combined (Host to GPU):                     %12.3f GB/sec\n", (double)104857600.0 / execInnerTiming(&GPUBenchmark::memTransfer, 104857600, 2, 0) / 1024 / 1024 / 1024);
	printf("\n");
	printf("Regular (GPU to Host):                            %12.3f GB/sec\n", (double)104857600.0 / execInnerTiming(&GPUBenchmark::memTransfer, 104857600, 0, 1) / 1024 / 1024 / 1024);
	printf("Page locked (GPU to Host):                        %12.3f GB/sec\n", (double)104857600.0 / execInnerTiming(&GPUBenchmark::memTransfer, 104857600, 1, 1) / 1024 / 1024 / 1024);
	printf("Write Combined (GPU to Host):                     %12.3f GB/sec\n", (double)104857600.0 / execInnerTiming(&GPUBenchmark::memTransfer, 104857600, 2, 1) / 1024 / 1024 / 1024);
	printf("\n");
	printf("Device (GPU to GPU):                              %12.3f GB/sec\n", (double)104857600.0 / execInnerTiming(&GPUBenchmark::memTransfer, 104857600, 0, 2) / 1024 / 1024 / 1024);
	printf("\n");

	printf("***** Device memory access speed (1024 x 512)\n");
	printf("Aligned read (int):                               %12.3f GB/sec\n", (double)104857600.0 * sizeof(int) / execInnerTiming(&GPUBenchmark::deviceMemAccess<int>, 104857600, 10, 1024, 512, maAlignedRead) / 1024 / 1024 / 1024);
	printf("Aligned read (float):                             %12.3f GB/sec\n", (double)104857600.0 * sizeof(float) / execInnerTiming(&GPUBenchmark::deviceMemAccess<float>, 104857600, 10, 1024, 512, maAlignedRead) / 1024 / 1024 / 1024);
	printf("Aligned read (double):                            %12.3f GB/sec\n", (double)104857600.0 * sizeof(double) / execInnerTiming(&GPUBenchmark::deviceMemAccess<double>, 104857600, 10, 1024, 512, maAlignedRead) / 1024 / 1024 / 1024);
	printf("\n");
	printf("Aligned write (int):                              %12.3f GB/sec\n", (double)104857600.0 * sizeof(int) / execInnerTiming(&GPUBenchmark::deviceMemAccess<int>, 104857600, 10, 1024, 512, maAlignedWrite) / 1024 / 1024 / 1024);
	printf("Aligned write (float):                            %12.3f GB/sec\n", (double)104857600.0 * sizeof(float) / execInnerTiming(&GPUBenchmark::deviceMemAccess<float>, 104857600, 10, 1024, 512, maAlignedWrite) / 1024 / 1024 / 1024);
	printf("aligned write (double):                           %12.3f GB/sec\n", (double)104857600.0 * sizeof(double) / execInnerTiming(&GPUBenchmark::deviceMemAccess<double>, 104857600, 10, 1024, 512, maAlignedWrite) / 1024 / 1024 / 1024);
	printf("\n");
	printf("Not aligned read (int):                           %12.3f GB/sec\n", (double)104857600.0 * sizeof(int) / execInnerTiming(&GPUBenchmark::deviceMemAccess<int>, 104857600, 1, 1024, 512, maNotAlignedRead) / 1024 / 1024 / 1024);
	printf("Not aligned read (float):                         %12.3f GB/sec\n", (double)104857600.0 * sizeof(float) / execInnerTiming(&GPUBenchmark::deviceMemAccess<float>, 104857600, 1, 1024, 512, maNotAlignedRead) / 1024 / 1024 / 1024);
	printf("Not aligned read (double):                        %12.3f GB/sec\n", (double)104857600.0 * sizeof(double) / execInnerTiming(&GPUBenchmark::deviceMemAccess<double>, 104857600, 1, 1024, 512, maNotAlignedRead) / 1024 / 1024 / 1024);
	printf("\n");
	printf("Not aligned write (int):                          %12.3f GB/sec\n", (double)104857600.0 * sizeof(int) / execInnerTiming(&GPUBenchmark::deviceMemAccess<int>, 104857600, 1, 1024, 512, maNotAlignedWrite) / 1024 / 1024 / 1024);
	printf("Not aligned write (float):                        %12.3f GB/sec\n", (double)104857600.0 * sizeof(float) / execInnerTiming(&GPUBenchmark::deviceMemAccess<float>, 104857600, 1, 1024, 512, maNotAlignedWrite) / 1024 / 1024 / 1024);
	printf("Not aligned write (double):                       %12.3f GB/sec\n", (double)104857600.0 * sizeof(double) / execInnerTiming(&GPUBenchmark::deviceMemAccess<double>, 104857600, 1, 1024, 512, maNotAlignedWrite) / 1024 / 1024 / 1024);
	printf("\n");

	printf("***** Pinned memory access speed (1024 x 512)\n");
	printf("Aligned read (int):                               %12.3f GB/sec\n", (double)104857600.0 * sizeof(int) / execInnerTiming(&GPUBenchmark::deviceMemAccess<int>, 104857600, 1, 1024, 512, maPinnedAlignedRead) / 1024 / 1024 / 1024);
	printf("Aligned read (float):                             %12.3f GB/sec\n", (double)104857600.0 * sizeof(float) / execInnerTiming(&GPUBenchmark::deviceMemAccess<float>, 104857600, 1, 1024, 512, maPinnedAlignedRead) / 1024 / 1024 / 1024);
	printf("Aligned read (double):                            %12.3f GB/sec\n", (double)104857600.0 * sizeof(double) / execInnerTiming(&GPUBenchmark::deviceMemAccess<double>, 104857600, 1, 1024, 512, maPinnedAlignedRead) / 1024 / 1024 / 1024);
	printf("\n");
	printf("Aligned write (int):                              %12.3f GB/sec\n", (double)104857600.0 * sizeof(int) / execInnerTiming(&GPUBenchmark::deviceMemAccess<int>, 104857600, 1, 1024, 512, maPinnedAlignedWrite) / 1024 / 1024 / 1024);
	printf("Aligned write (float):                            %12.3f GB/sec\n", (double)104857600.0 * sizeof(float) / execInnerTiming(&GPUBenchmark::deviceMemAccess<float>, 104857600, 1, 1024, 512, maPinnedAlignedWrite) / 1024 / 1024 / 1024);
	printf("Aligned write (double):                           %12.3f GB/sec\n", (double)104857600.0 * sizeof(double) / execInnerTiming(&GPUBenchmark::deviceMemAccess<double>, 104857600, 1, 1024, 512, maPinnedAlignedWrite) / 1024 / 1024 / 1024);
	printf("\n");
	printf("Not aligned read (int):                           %12.3f GB/sec\n", (double)104857600.0 * sizeof(int) / execInnerTiming(&GPUBenchmark::deviceMemAccess<int>, 104857600, 1, 1024, 512, maPinnedNotAlignedRead) / 1024 / 1024 / 1024);
	printf("Not aligned read (float):                         %12.3f GB/sec\n", (double)104857600.0 * sizeof(float) / execInnerTiming(&GPUBenchmark::deviceMemAccess<float>, 104857600, 1, 1024, 512, maPinnedNotAlignedRead) / 1024 / 1024 / 1024);
	printf("Not aligned read (double):                        %12.3f GB/sec\n", (double)104857600.0 * sizeof(double) / execInnerTiming(&GPUBenchmark::deviceMemAccess<double>, 104857600, 1, 1024, 512, maPinnedNotAlignedRead) / 1024 / 1024 / 1024);
	printf("\n");
	printf("Not aligned write (int):                          %12.3f GB/sec\n", (double)104857600.0 * sizeof(int) / execInnerTiming(&GPUBenchmark::deviceMemAccess<int>, 104857600, 1, 1024, 512, maPinnedNotAlignedWrite) / 1024 / 1024 / 1024);
	printf("Not aligned write (float):                        %12.3f GB/sec\n", (double)104857600.0 * sizeof(float) / execInnerTiming(&GPUBenchmark::deviceMemAccess<float>, 104857600, 1, 1024, 512, maPinnedNotAlignedWrite) / 1024 / 1024 / 1024);
	printf("Not aligned write (double):                       %12.3f GB/sec\n", (double)104857600.0 * sizeof(double) / execInnerTiming(&GPUBenchmark::deviceMemAccess<double>, 104857600, 1, 1024, 512, maPinnedNotAlignedWrite) / 1024 / 1024 / 1024);
	printf("\n");
}