#pragma once

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <iostream>

#include <cuda_runtime_api.h>
#include <curand.h>
#include <cufft.h>

#include "CTools.h"

#define CUDA_TOOLS_THREADS_PER_BLOCK	128
#define CUDA_TOOLS_MAX_BLOCKS			256

using namespace std;

void 
	cudaDevicePropertiesDump();
size_t 
	cudaFreeMemGet();
void 
	cudaMemInfoDump();
void 
	cudaSafeCall(cudaError AError);
void 
	cufftSafeCall(cufftResult AResult);
void 
	curandSafeCall(curandStatus_t status);

class cudaException : public customException {
public:
	cudaException(const char *errorMessage, int errorCode) : 
		customException(errorMessage, errorCode) {}
};

class cufftException : public customException {
public:
	cufftException(const char *errorMessage, int errorCode) : 
		customException(errorMessage, errorCode) {}
};

class curandException : public customException {
public:
	curandException(const char *errorMessage, int errorCode) : 
		customException(errorMessage, errorCode) {}
};

template<class T> class deviceMem;
template<class T> class hostMem;
template<class T> class mappedMem;

template<class T>
class deviceMem
{
	deviceMem(const deviceMem&);
	deviceMem& operator=(const deviceMem&);
public:
	T *dptr;
	size_t size;

	deviceMem() : size(0), dptr(NULL) {};
	deviceMem(size_t count) : dptr(NULL) {
		allocate(count);
	};
	~deviceMem() {
		release();
	};
	void allocate(size_t count) {
		release(); 
		size = count * sizeof(T); 
		cudaSafeCall(cudaMalloc((void**)&dptr, size));
	};
	void release() {
		if (dptr == NULL)
			return;
		cudaSafeCall(cudaFree(dptr)); 
		dptr = NULL; 
		size = 0;
	};
	void clear() {
		cudaSafeCall(cudaMemset(dptr, 0, size));
	}
	void copyFrom(T *hptr) {
		cudaSafeCall(cudaMemcpy(dptr, hptr, size, cudaMemcpyHostToDevice));
	};
	void copyFrom(hostMem<T> &mem) {
		copyFrom(mem.hptr);
	};
	void copyFromAsync(T *hptr, cudaStream_t stream = 0) {
		cudaSafeCall(cudaMemcpyAsync(dptr, hptr, size, cudaMemcpyHostToDevice, stream));
	};
	void copyFromAsync(hostMem<T> &mem, cudaStream_t stream = 0) {
		copyFromAsync(mem.hptr, stream);
	};
	void copyTo(T *hptr) {
		cudaSafeCall(cudaMemcpy(hptr, dptr, size, cudaMemcpyDeviceToHost));
	};
	void copyTo(hostMem<T> &mem) {
		copyTo(mem.hptr);
	};
	void copyTo(deviceMem<T> &mem) {
		cudaSafeCall(cudaMemcpy(mem.dptr, dptr, size, cudaMemcpyDeviceToDevice));
	};
	void copyToAsync(T *hptr, cudaStream_t stream = 0) {
		cudaSafeCall(cudaMemcpyAsync(hptr, dptr, size, cudaMemcpyDeviceToHost, stream));
	};
	void copyToAsync(hostMem<T> &mem, cudaStream_t stream = 0) {
		copyToAsync(mem.hptr, stream);
	};
	void copyToAsync(deviceMem<T> &mem, cudaStream_t stream = 0) {
		cudaSafeCall(cudaMemcpyAsync(mem.dptr, dptr, size, cudaMemcpyDeviceToDevice, stream));
	};
};

enum hostMemType {hmtRegular = 0, hmtSpecial = 1};

template<class T>
class hostMem
{
	hostMem(const hostMem&);
	hostMem& operator=(const hostMem&);

public:
	T *hptr;
	size_t size;
	unsigned int flags;
	hostMemType type;

	hostMem() : type(hmtRegular), size(0), hptr(NULL) {};
	hostMem(size_t count, unsigned int flags) : type(hmtSpecial), flags(flags), hptr(NULL) {
		allocate(count);
	};
	hostMem(size_t count) : type(hmtRegular), flags(0), hptr(NULL) {
		allocate(count);
	};
	~hostMem() {
		release();
	};
	void allocate(size_t count) {
		assert(type == hmtRegular || type == hmtSpecial);
		release();
		size = count * sizeof(T); 
		if (type == hmtRegular)
			hptr = new T[count];
		else
			cudaSafeCall(cudaHostAlloc((void**)&hptr, size, flags));
	};
	void allocate(size_t count, unsigned int aFlags) {
		release();
		type = hmtSpecial;
		flags = aFlags;
		size = count * sizeof(T); 
		cudaSafeCall(cudaHostAlloc((void**)&hptr, size, flags));
	};
	void release() {
		assert(type == hmtRegular || type == hmtSpecial);
		if (hptr == NULL)
			return;
		if (type == hmtRegular)
			delete [] hptr;
		else
			cudaSafeCall(cudaFreeHost(hptr)); 
		hptr = NULL;
		size = 0;
	}
	T& operator[](size_t index) {
		return hptr[index];
	};
	void copyFrom(deviceMem<T> &mem) {
		cudaSafeCall(cudaMemcpy(hptr, mem.dptr, size, cudaMemcpyDeviceToHost));
	};
	void copyFromAsync(deviceMem<T> &mem, cudaStream_t stream = 0) {
		cudaSafeCall(cudaMemcpyAsync(hptr, mem.dptr, size, cudaMemcpyDeviceToHost, stream));
	};
	void copyTo(deviceMem<T> &mem) {
		cudaSafeCall(cudaMemcpy(mem.dptr, hptr, size, cudaMemcpyHostToDevice));
	};
	void copyToAsync(deviceMem<T> &mem, cudaStream_t stream = 0) {
		cudaSafeCall(cudaMemcpyAsync(mem.dptr, hptr, size, cudaMemcpyHostToDevice, stream));
	};
#ifndef linux
	void saveToFile(const char *FLP) {
		FILE *file;
		if (fopen_s(&file, FLP, "wb") != 0)
			assert(false);
		if (hptr != NULL)
			fwrite(hptr, size, 1, file);
		fclose(file);
	}
#endif
};

enum mappedMemType {mmtNone = 0, mmtAlloc = 1, mmtRegisterHost = 2};

template<class T>
class mappedMem
{
	mappedMem(const mappedMem&);
	mappedMem& operator=(const mappedMem&);
public:
	T *hptr;
	T *dptr;
	size_t size;
	mappedMemType type;

	mappedMem() : size(0), hptr(NULL), dptr(NULL), type(mmtNone) {};
	mappedMem(size_t count) : hptr(NULL), dptr(NULL), type(mmtNone) {
		allocate(count);
	};
	~mappedMem() {
		release();
	};
	void allocate(size_t count) {
		release(); 
		size = count * sizeof(T); 
		type = mmtAlloc;
		cudaSafeCall(cudaHostAlloc((void**)&hptr, size, cudaHostAllocMapped)); 
		cudaSafeCall(cudaHostGetDevicePointer((void**)&dptr, hptr, 0));
	};
	void registerHost(T *hostPtr, size_t count) {
		release(); 
		size = count * sizeof(T); 
		type = mmtRegisterHost;
		cudaSafeCall(cudaHostRegister(hostPtr, size, cudaHostRegisterMapped)); 
		hptr = hostPtr;
		cudaSafeCall(cudaHostGetDevicePointer((void**)&dptr, hptr, 0));
	};
	void release() {
		if (type == mmtNone)
			return;
		if (hptr == NULL)
		{
			type = mmtNone;
			return;
		}
		assert(type == mmtAlloc || type == mmtRegisterHost);
		if (type == mmtAlloc) 
			cudaSafeCall(cudaFreeHost(hptr)); 
		if (type == mmtRegisterHost) 
			cudaSafeCall(cudaHostUnregister(hptr));
		type = mmtNone;
		hptr = NULL; 
		dptr = NULL;
		size = 0;
	};
	T& operator[](size_t index) {return hptr[index];};
};

template<class T> void cuda_arraySub(T *A, T b, int count);
template<class T> void cuda_arrayMax(T *R, T *A, T b, int count);
template<class T> void cuda_arraySum(T *R, T *A, int count);
template<class T> void cuda_arrayStd(T *R, T *A, int count);