#include "Common/CTools.h"
#include "Common/CUDATools.h"
#include "GPUBenchmark.h"

int main(int argc, char * argv[])
{
	try
	{
		printf("Started GPUBenchmark\n");
		TimingInitialize();

		GPUBenchmark benchmark;
		benchmark.run();

		printf("Completed successfully\n");
	}
	catch (customException *E) 
	{
		printf("Exception: %s (Error Code: %d)\n", E->what(), E->errorCode);
		SystemPause();
		return 1;
	}
	catch (exception *E) 
	{
		printf("Exception: %s\n", E->what());
		SystemPause();
		return 1;
	}
	catch (...) 
	{
		printf("Unknown exception occured\n");
		SystemPause();
		return 1;
	}
	SystemPause();
	return 0;
}
