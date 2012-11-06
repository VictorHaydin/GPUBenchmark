#include "CTools.h"

static __int64 FPerformanceFrequency;

std::string BoolToStrYesNo(bool AValue)
{
	return std::string(AValue ? "Yes" : "No");
}

CompareResult Compare(float v1, float v2)
{
	return v1 < v2 ? crLess : v1 > v2 ? crGreater : crEqual;
}

void IntToCharBufF(__int64 AValue, char *ABuf, size_t ASize)
{
	const size_t MAX_BUF = 27;
	char LBuf[MAX_BUF], *LPBuf;
	LPBuf = LBuf + MAX_BUF - 1;
	*LPBuf-- = 0;
//	lldiv_t LValue = {AValue, 0};
	__int64 LQuot = AValue;
	char LRem = 0;
	int LCount = 0;
	if (LQuot == 0)
		*LPBuf-- = '0';
	else
		while (LQuot > 0)
		{
			LRem = LQuot % 10;
			LQuot = LQuot / 10;
			//LValue = lldiv(LValue.quot, 10);
			*LPBuf-- = '0' + LRem;
			if ((++LCount == 3) && (LQuot > 0))
			{
				*LPBuf-- = ',';
				LCount = 0;
			}
		}
	std::memcpy(ABuf, LPBuf + 1, std::min((size_t)(LBuf + MAX_BUF - LPBuf - 1), ASize));
}

std::string IntToStrF(__int64 AValue)
{
	char LResult[27];
	IntToCharBufF(AValue, LResult, 27);
	return std::string(LResult);
}

void TimingClearAndStart(TimingCounter &ACounter)
{
	ACounter.CounterTotal = 0;
	TimingStart(ACounter);
}

void TimingFinish(TimingCounter &ACounter)
{
	__int64 LCounter;
#ifdef linux
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	LCounter = FPerformanceFrequency * ts.tv_sec + ts.tv_nsec;
#else
	QueryPerformanceCounter((LARGE_INTEGER*)&(LCounter));
#endif
	ACounter.CounterTotal += (LCounter - ACounter.Counter);
}

void TimingFinish(TimingCounter &ACounter, const char *AStdOutTimingDesc)
{
	TimingFinish(ACounter);
	printf("Finished in %.6f sec(s)  - %s\n", TimingSeconds(ACounter), AStdOutTimingDesc);
}

void TimingInitialize()
{
#ifdef linux
	FPerformanceFrequency = 1000000000; // nanosec
#else
	QueryPerformanceFrequency((LARGE_INTEGER*)&FPerformanceFrequency);
#endif
}

double TimingSeconds()
{
	__int64 LCounter;
#ifdef linux
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	LCounter = FPerformanceFrequency * ts.tv_sec + ts.tv_nsec;
#else
	QueryPerformanceCounter((LARGE_INTEGER*)&(LCounter));
#endif
	return (double)LCounter / FPerformanceFrequency;
}

double TimingSeconds(TimingCounter &ACounter)
{
	return ((double)(ACounter.CounterTotal) / FPerformanceFrequency);
}

void TimingStart(TimingCounter &ACounter)
{
#ifdef linux
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	ACounter.Counter = FPerformanceFrequency * ts.tv_sec + ts.tv_nsec;
#else
	QueryPerformanceCounter((LARGE_INTEGER*)&(ACounter.Counter));
#endif
}
	
void SystemPause()
{
#ifdef linux
	printf("Press ENTER to continue ...\n");
	getchar();
#else
	system("pause");
#endif
}
