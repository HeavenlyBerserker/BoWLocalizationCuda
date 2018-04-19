#!/bin/sh
if [ -e slurm* ]
then
	rm slurm*
fi
make
sbatch runscript.sh
