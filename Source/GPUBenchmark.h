#include <cstdlib>
#include <cstdio>

class GPUBenchmark;

typedef void (GPUBenchmark::*benchmarkOuterFunc)(int, int, int, int, int);
typedef double (GPUBenchmark::*benchmarkInnerFunc)(int, int, int, int, int);

enum DeviceMemAccessEnum : int {
	maAlignedRead, 
	maAlignedWrite,
	maNotAlignedRead, 
	maNotAlignedWrite,
	maPinnedAlignedRead, 
	maPinnedAlignedWrite,
	maPinnedNotAlignedRead, 
	maPinnedNotAlignedWrite
};

class GPUBenchmark
{
	GPUBenchmark(const GPUBenchmark&);
	GPUBenchmark& operator=(const GPUBenchmark&);

	__int64 deviceClockRate;

	void deviceMemAllocRelease(int size, int repeatCount, int, int, int);
	void mappedMemAllocRelease(int size, int repeatCount, int, int, int);
	void hostMemRegisterUnregister(int size, int repeatCount, int, int, int);
	void hostMemWriteCombinedAllocRelease(int size, int repeatCount, int, int, int);

	double memTransfer(int size, int mode, int direction, int, int);
	template<typename T> double deviceMemAccess(int count, int repeatCount, int blockCount, int threadsPerBlock, int memAccess);

	template<typename T> double reductionSum(int count, int repeatCount, int blockCount, int threadsPerBlock, int);
	double kernelExecuteTinyTask(int blockCount, int threadCount, int, int, int);
	double kernelScheduleTinyTask(int blockCount, int threadCount, int, int, int);
	template<typename T> void kernelAdd(int count, int blockCount, int threadCount, int, int);
	template<typename T> void kernelAdd2(int count, int blockCount, int threadCount, int, int);
	template<typename T> void kernelAddMulMix(int count, int blockCount, int threadCount, int, int);
	template<typename T> void kernelMul(int count, int blockCount, int threadCount, int, int);
	template<typename T> void kernelDiv(int count, int blockCount, int threadCount, int, int);
	template<typename T> void kernelSin(int count, int blockCount, int threadCount, int, int);
#ifdef CUDA50
	double kernelDynamicExecuteTinyTask(int blockCount, int threadCount, int, int, int);
	double kernelDynamicScheduleTinyTask(int blockCount, int threadCount, int, int, int);
#endif
	double execOuterTiming(benchmarkOuterFunc func, int p1 = 0, int p2 = 0, int p3 = 0, int p4 = 0, int p5 = 0);
	double execInnerTiming(benchmarkInnerFunc func, int p1 = 0, int p2 = 0, int p3 = 0, int p4 = 0, int p5 = 0);
public:
	GPUBenchmark() {};

	void run();
};

template<typename T> 
void cuda_reductionSum(T *data, T *sum, int count, int repeatCount, int blockCount, int threadsPerBlock);
template<typename T> 
void cuda_alignedRead(T *data, int count, int repeatCount, int blockCount, int threadsPerBlock);
template<typename T> 
void cuda_notAlignedRead(T *data, int count, int repeatCount, int blockCount, int threadsPerBlock);
template<typename T> 
void cuda_alignedWrite(T *data, int count, int repeatCount, int blockCount, int threadsPerBlock);
template<typename T> 
void cuda_notAlignedWrite(T *data, int count, int repeatCount, int blockCount, int threadsPerBlock);

template<typename T> 
void cuda_doAdd(int count, int blockCount, int threadCount);
template<typename T> 
void cuda_doAdd2(int count, int blockCount, int threadCount);
template<typename T> 
void cuda_doAddMulMix(int count, int blockCount, int threadCount);
template<typename T> 
void cuda_doMul(int count, int blockCount, int threadCount);
template<typename T> 
void cuda_doDiv(int count, int blockCount, int threadCount);
template<typename T> 
void cuda_doSin(int count, int blockCount, int threadCount);

void cuda_doTinyTask(int blockCount, int threadCount);
#ifdef CUDA50
double cuda_doDynamicTinyTask(int blockCount, int threadCount, bool waitForCompletion);
#endif