SOURCE_PATH = ../../Source/

CPP_FLAGS = -Xcompiler "-fPIC -O2 -g -std=gnu++0x"
CPP_FLAGS_COMPUTE_20 = -D CUDA42
CPP_FLAGS_COMPUTE_35 = -D CUDA50

CU_FLAGS = -use_fast_math -Xcompiler "-fPIC -O2 -g"
CU_FLAGS_COMPUTE_20 = -D CUDA42 -gencode=arch=compute_20,code=\"sm_20,compute_20\"
CU_FLAGS_COMPUTE_35 = -dc -D CUDA50 -gencode=arch=compute_35,code=\"sm_35,compute_35\"

LINK_FLAGS = -lcuda -lcudart -lcurand
LINK_FLAGS_COMPUTE_20 = 
LINK_FLAGS_COMPUTE_35 = -arch=sm_35 -rdc=true -lcudadevrt

CC = nvcc

CPP_FILES = \
	$(SOURCE_PATH)Common/CTools.cpp \
	$(SOURCE_PATH)Common/CUDATools.cpp \
	$(SOURCE_PATH)GPUBenchmark.cpp \
	$(SOURCE_PATH)Main.cpp

CU_FILES = \
	$(SOURCE_PATH)Common/CUDATools.cu \
	$(SOURCE_PATH)GPUBenchmark.cu

CPP_OBJECTS = $(CPP_FILES:.cpp=.o)
CU_OBJECTS = $(CU_FILES:.cu=_cu.o)

EXECUTABLE = GPUBenchmark
           
all:	$(CPP_FILES) $(CU_FILES) $(EXECUTABLE)

$(EXECUTABLE): $(CPP_OBJECTS) $(CU_OBJECTS)
	$(CC) $(LINK_FLAGS) $(LINK_FLAGS_COMPUTE_20) $(addprefix COMPUTE_20/,$(notdir $(CPP_OBJECTS))) $(addprefix COMPUTE_20/,$(notdir $(CU_OBJECTS))) -o $(addprefix COMPUTE_20/,$@);
	$(CC) $(LINK_FLAGS) $(LINK_FLAGS_COMPUTE_35) $(addprefix COMPUTE_35/,$(notdir $(CPP_OBJECTS))) $(addprefix COMPUTE_35/,$(notdir $(CU_OBJECTS))) -o $(addprefix COMPUTE_35/,$@);

%.o: %.cpp BuildDirStructure
	$(CC) -c $(CPP_FLAGS) $(CPP_FLAGS_COMPUTE_20) $< -o $(addprefix COMPUTE_20/,$(notdir $@));
	$(CC) -c $(CPP_FLAGS) $(CPP_FLAGS_COMPUTE_35) $< -o $(addprefix COMPUTE_35/,$(notdir $@));

%_cu.o: %.cu BuildDirStructure
	$(CC) -c $(CU_FLAGS) $(CU_FLAGS_COMPUTE_20) $< -o $(addprefix COMPUTE_20/,$(notdir $@));
	$(CC) -c $(CU_FLAGS) $(CU_FLAGS_COMPUTE_35) $< -o $(addprefix COMPUTE_35/,$(notdir $@));

BuildDirStructure:
	mkdir -p COMPUTE_20;
	mkdir -p COMPUTE_35;

clean:
	rm COMPUTE_20/*.o COMPUTE_35/*.o COMPUTE_20/$(EXECUTABLE) COMPUTE_35/$(EXECUTABLE)
