NVCCFLAGS  +=-std=c++11
NVCCFLAGS  += g++
#NVCCFLAGS += -g


histogramizer:histo.cu
	nvcc $^ -o $@

clean:
	-rm histogramizer
