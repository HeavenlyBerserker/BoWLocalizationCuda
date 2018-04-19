#!/bin/bash
if [ -e sifter/ ]
then
	rm -r sifter/
fi
mkdir sifter
cd sifter
wget "https://github.com/rajathjavali/CudaSift/archive/Maxwell.zip"
module load opencv
module load cuda/9.1
unzip Maxwell.zip
rm Maxwell.zip
mv CudaSift-Maxwell CudaSift
cmake CudaSift
cd ..
cp funcs/mainSift.cpp sifter/CudaSift
cp funcs/runscript.sh sifter/CudaSift
chmod -R 777 sifter/
