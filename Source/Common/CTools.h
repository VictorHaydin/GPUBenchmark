#pragma once

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <exception>
#include <algorithm>

#ifndef linux
#define NOMINMAX
#include <windows.h>
#endif

#define DeleteAndNull(a) if (a != NULL) {delete a; a = NULL;}

#ifdef linux
typedef long long __int64;
typedef char byte;
#endif

class customException : public std::exception
{
	std::string errorMessage;
public:
	int errorCode;
	explicit customException(const char *errorMessage) : 
		errorMessage(errorMessage), errorCode(-1) {}
	customException(const char *errorMessage, int errorCode) : 
		errorMessage(errorMessage), errorCode(errorCode) {}
	customException(const std::string &errorMessage, int errorCode) : 
		errorMessage(errorMessage), errorCode(errorCode) {}
	virtual ~customException() throw() {}
	virtual const char * what() const throw() {return errorMessage.c_str();}
};

struct TimingCounter {
	__int64 Counter;
	__int64 CounterTotal;

	TimingCounter() : Counter(0), CounterTotal(0) {}
};

enum CompareResult {crLess = -1, crEqual = 0, crGreater = 1};

std::string 
	BoolToStrYesNo(bool AValue);
CompareResult
	Compare(float v1, float v2);
void 
	IntToCharBufF(__int64 AValue, char *ABuf, size_t ASize);
std::string 
	IntToStrF(__int64 AValue);
void 
	SystemPause();
void 
	TimingClearAndStart(TimingCounter &ACounter);
void
	TimingFinish(TimingCounter &ACounter);
void 
	TimingFinish(TimingCounter &ACounter, const char *AStdOutTimingDesc);
void
	TimingInitialize();
double 
	TimingSeconds(TimingCounter &ACounter);
double 
	TimingSeconds();
void
	TimingStart(TimingCounter &ACounter);

template<class T>
class dynArray
{
	dynArray(const dynArray&);
	dynArray& operator=(const dynArray&);

public:
	T *ptr;
	size_t size;
	size_t count;

	dynArray() : size(0), count(0), ptr(NULL) {};
	explicit dynArray(size_t aCount) : ptr(NULL) {
		allocate(aCount);
	};
	~dynArray() {
		release();
	};
	void allocate(size_t aCount) {
		release();
		count = aCount;
		size = count * sizeof(T); 
		if (count > 0)
			ptr = new T[count];
	};
	void release() {
		if (ptr == NULL)
			return;
		delete [] ptr;
		ptr = NULL;
		size = 0;
		count = 0;
	}
	T& operator[](size_t index) {
		return ptr[index];
	};
#ifdef WIN
	void saveToFile(const char *FLP) {
		FILE *file;
		if (fopen_s(&file, FLP, "wb") != 0)
			assert(false);
		if (ptr != NULL)
			fwrite(ptr, size, 1, file);
		fclose(file);
	}
#endif
};

template<class T>
class dynArray2D
{
	dynArray2D(const dynArray2D&);
	dynArray2D& operator=(const dynArray2D&);

public:
	T *ptr;
	size_t size;
	size_t dim1;
	size_t dim2;

	dynArray2D() : size(0), dim1(0), dim2(0), ptr(NULL) {};
	dynArray2D(size_t dim1, size_t dim2) : ptr(NULL) {
		allocate(dim1, dim2);
	};
	~dynArray2D() {
		release();
	};
	void allocate(size_t aDim1, size_t aDim2) {
		release();
		dim1 = aDim1;
		dim2 = aDim2;
		size = dim1 * dim2 * sizeof(T); 
		if (size > 0)
			ptr = new T[dim1 * dim2];
	};
	void release() {
		if (ptr == NULL)
			return;
		delete [] ptr;
		ptr = NULL;
		size = 0;
		dim1 = 0;
		dim2 = 0;
	}
	T& operator[](size_t index) {
		return ptr[index];
	};
	T& operator()(size_t index1, size_t index2) {
		return ptr[index1 * dim2 + index2];
	};
#ifdef WIN
	void saveToFile(const char *FLP) {
		FILE *file;
		if (fopen_s(&file, FLP, "wb") != 0)
			assert(false);
		if (ptr != NULL)
			fwrite(ptr, size, 1, file);
		fclose(file);
	}
#endif
};

template<class T>
class dynArrayOfArray
{
	dynArrayOfArray(const dynArrayOfArray&);
	dynArrayOfArray& operator=(const dynArrayOfArray&);

public:
	dynArray<dynArray<T> > data;

	dynArrayOfArray() {};
	dynArrayOfArray(size_t aDim1, size_t aDim2) {
		allocate(aDim1, aDim2);
	};
	~dynArrayOfArray() {
		release();
	};
	void allocate(size_t aDim1, size_t aDim2) {
		data.allocate(aDim1);
		for (int i = 0; i < aDim1; i++)
			data[i].allocate(aDim2);
	}
	void release() {
		for (int i = 0; i < data.count; i++)
			data[i].release();
		data.release();
	}
	dynArray<T>& operator[](size_t index) {
		return data[index];
	};
};

template<class T>
class dynArrayOfArrayOfArray
{
	dynArrayOfArrayOfArray(const dynArrayOfArrayOfArray&);
	dynArrayOfArrayOfArray& operator=(const dynArrayOfArrayOfArray&);

public:
	dynArray<dynArrayOfArray<T> > data;

	dynArrayOfArrayOfArray() {};
	dynArrayOfArrayOfArray(size_t aDim1, size_t aDim2, size_t aDim3) {
		allocate(aDim1, aDim2, aDim3);
	};
	~dynArrayOfArrayOfArray() {
		release();
	};
	void allocate(size_t aDim1, size_t aDim2, size_t aDim3) {
		data.allocate(aDim1);
		for (int i = 0; i < aDim1; i++)
			data[i].allocate(aDim2, aDim3);
	}
	void release() {
		for (int i = 0; i < data.count; i++)
			data[i].release();
		data.release();
	}
	dynArrayOfArray<T>& operator[](size_t index) {
		return data[index];
	};
};
