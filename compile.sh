#!/bin/sh
cd sifter
cd CudaSift
if [ -e slurm* ]
then
	rm slurm*
fi
cd ..
make
cp cudasift CudaSift
