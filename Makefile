# TODO 
# add environment check

# make -j 5
# srun ./test

# RV , change this to the place where you place the CUB
CUB_HOME  := /uufs/chpc.utah.edu/common/home/u0867999/BoWLocalizationCuda/cub-1.8.0/
#CUB_HOME  := ${REL_ROOT}/cub-1.8.0

CC		  := /usr/local/cuda-9.1/bin/nvcc 

CPPFLAGS  +=-std=c++11

CPPFLAGS  += -ccbin
CPPFLAGS  += g++
CPPFLAGS  += -m64
CPPFLAGS  += -gencode arch=compute_60,code=sm_60
CPPFLAGS  += -gencode arch=compute_61,code=sm_61
CPPFLAGS  += -gencode arch=compute_70,code=sm_70
CPPFLAGS  += -gencode arch=compute_70,code=compute_70

CC_INCLUDE +=-I /usr/local/cuda-9.1/include/
CC_INCLUDE +=-I $(CUB_HOME)

#
#TODO
# if DEBUG
CPPFLAGS += -G
#

all: BoW

BoW:
	$(CC) $(CPPFLAGS) $(CC_INCLUDE) main.cpp

clean:
	rm -rf *.o *.test

