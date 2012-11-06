#include "CUDATools.h"

size_t cudaFreeMemGet()
{
	size_t LTotal, LFree;
	cudaSafeCall(cudaMemGetInfo(&LFree, &LTotal));
	return LFree;
}

void cudaMemInfoDump()
{
	size_t LTotal, LFree;
	cudaSafeCall(cudaMemGetInfo(&LFree, &LTotal));
	char S[30];
	IntToCharBufF(LTotal, S, 30);
	cout << "CUDA Memory Info: " << endl;
	cout << "    Total: " << S << " byte(s)" << endl;
	IntToCharBufF(LFree, S, 30);
	cout << "    Free:  " << S << " byte(s)" << endl;
}

void cudaDevicePropertiesDump()
{
	cudaDeviceProp LProps;
	int LDevice;
	cudaSafeCall(cudaGetDevice(&LDevice));
	cudaSafeCall(cudaGetDeviceProperties(&LProps, LDevice));
	cout << "CUDA Device properties (Device " << LDevice << ")" << endl;
	cout << "    Name:                        " << LProps.name << endl;
	cout << "    Version:                     " << IntToStrF(LProps.major) << "." << IntToStrF(LProps.minor) << endl;
	cout << "    totalGlobalMem:              " << IntToStrF(LProps.totalGlobalMem) << endl;
	cout << "    sharedMemPerBlock:           " << IntToStrF(LProps.sharedMemPerBlock) << endl;
	cout << "    regsPerBlock:                " << IntToStrF(LProps.regsPerBlock) << endl;
	cout << "    warpSize:                    " << IntToStrF(LProps.warpSize) << endl;
	cout << "    memPitch:                    " << IntToStrF(LProps.memPitch) << endl;
	cout << "    maxThreadsPerBlock:          " << IntToStrF(LProps.maxThreadsPerBlock) << endl;
	cout << "    maxThreadsDim[3]:            " << 
		IntToStrF(LProps.maxThreadsDim[0]) << " x " << 
		IntToStrF(LProps.maxThreadsDim[1]) << " x " << 
		IntToStrF(LProps.maxThreadsDim[2]) << endl;
	cout << "    maxGridSize[3]:              " << 
		IntToStrF(LProps.maxGridSize[0]) << " x " << 
		IntToStrF(LProps.maxGridSize[1]) << " x " << 
		IntToStrF(LProps.maxGridSize[2]) << endl;
	cout << "    clockRate:                   " << IntToStrF(LProps.clockRate) << endl;
	cout << "    totalConstMem:               " << IntToStrF(LProps.totalConstMem) << endl;
	cout << "    textureAlignment:            " << IntToStrF(LProps.textureAlignment) << endl;
	cout << "    texturePitchAlignment:       " << IntToStrF(LProps.texturePitchAlignment) << endl;
	cout << "    deviceOverlap:               " << BoolToStrYesNo(LProps.deviceOverlap > 0) << endl;
	cout << "    multiProcessorCount:         " << IntToStrF(LProps.multiProcessorCount) << endl;
	cout << "    kernelExecTimeoutEnabled:    " << BoolToStrYesNo(LProps.kernelExecTimeoutEnabled > 0) << endl;
	cout << "    integrated:                  " << BoolToStrYesNo(LProps.integrated > 0) << endl;
	cout << "    canMapHostMemory:            " << BoolToStrYesNo(LProps.canMapHostMemory > 0) << endl;
	cout << "    computeMode:                 " << IntToStrF(LProps.computeMode) << endl;
	cout << "    concurrentKernels:           " << BoolToStrYesNo(LProps.concurrentKernels > 0) << endl;
	cout << "    ECCEnabled:                  " << BoolToStrYesNo(LProps.ECCEnabled > 0) << endl;
	cout << "    pciBusID:                    " << IntToStrF(LProps.pciBusID) << endl;
	cout << "    pciDeviceID:                 " << IntToStrF(LProps.pciDeviceID) << endl;
	cout << "    pciDomainID:                 " << IntToStrF(LProps.pciDomainID) << endl;
	cout << "    tccDriver:                   " << IntToStrF(LProps.tccDriver) << endl;
	cout << "    asyncEngineCount:            " << IntToStrF(LProps.asyncEngineCount) << endl;
	cout << "    unifiedAddressing:           " << BoolToStrYesNo(LProps.unifiedAddressing > 0) << endl;
	cout << "    memoryClockRate:             " << IntToStrF(LProps.memoryClockRate) << endl;
	cout << "    memoryBusWidth:              " << IntToStrF(LProps.memoryBusWidth) << endl;
	cout << "    L2CacheSize:                 " << IntToStrF(LProps.l2CacheSize) << endl;
	cout << "    maxThreadsPerMultiProcessor: " << IntToStrF(LProps.maxThreadsPerMultiProcessor) << endl;
}

void cudaSafeCall(cudaError AError)
{
    if (AError != cudaSuccess)
		throw new cudaException(cudaGetErrorString(AError), AError);
}

void cufftSafeCall(cufftResult AResult)
{
    if (AResult != CUFFT_SUCCESS)
		throw new cufftException("cuFFT call failed", AResult);
}

void curandSafeCall(curandStatus_t status)
{
    if (status != CURAND_STATUS_SUCCESS)
		throw new curandException("cuRAND call failed", status);
}