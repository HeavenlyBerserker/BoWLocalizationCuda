//--------------------------------------------//
//Histogram for bag of words CPU
//Authors: Hong Xu and Rajath Javali
//--------------------------------------------//
#include <stdio.h>
#include <iterator>
#include <sys/types.h>
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iterator>
#include <iostream>
#include <algorithm>
#include <typeinfo>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

using namespace std;

#define BLOCK_SIZE 1024
#define BLOCK_SIZE2 32
#define MAX_K 20000

typedef std::vector<std::string> stringvec;

//List of structs
struct Sift{
	float* sift;
	int numSift;
	int totalNum;
	int elementSize;
};

struct Histogram{
	float* vals;
	int size;
	string filename;
};

//List of functions
int countlines(char *filename);
void readFromFile(int size,float *v, char *file);
Histogram createHist(Sift cent, Sift im);
Sift readSift(char *filename);
void printSift(Sift sift, char option);
void freeSift(Sift sift);
void printHisto(Histogram h, char option);
void freeHisto(Histogram h);
void printFiles(stringvec v);
void read_directory(const std::string& name, stringvec& v);
__global__ void createHistCuda (float* siftCentroids, float* siftImage, int linesCent, int linesIm,float*temp);
vector<Sift> readMultiSifts(stringvec v);
vector<Histogram> createHistogramCudaAux(Sift& siftCentroids, vector<Sift>& imSifts, stringvec& v);
vector<vector<int> > matchCudaAux(vector<Histogram>& query, vector<Histogram>& database);
__global__ void matchHistCuda(float*qSet, float*dbSet, size_t qSize, size_t dbSize, size_t hSize, float*out);
void histogram2array(float* array, vector<Histogram>& hists);
template <class T> 
void arrayto2Dvec(T* array, vector<vector<T> >& vec, size_t x, size_t y);
template <class T> 
void print2DVec(vector<vector<T> >& vec);
template<class T>
void print_array(T& array, int m, int n);
__global__ void rowMin(float* input, int* output, size_t rowS, size_t rowNum);
string extName(string s);
vector<vector<string> > comFinal(vector<Histogram>& query, vector<Histogram>& database, vector<vector<int> >& values, int num);
void analize(vector<vector<string> >& results);
string extRight(string s);
void freeHistograms(vector<Histogram> histos);

//---------------------------------------------//
//Main
//---------------------------------------------//
int main(int argc, char *argv[])
{   
	//Header
	printf("\t\t###########################################\n\t\t#######   Histogram Program   #############\n\t\t###########################################\n\n");
	
	//Getting query file names
	stringvec vquery;
	string name = "../input/querySift/";
	read_directory(name, vquery);
	printFiles(vquery);
	//Getting database file names
	stringvec vdatabase;
	name = "../input/databaseSift/";
	read_directory(name, vdatabase);
	printFiles(vdatabase);
	
	//Readsing Centroid sifts
	Sift siftCentroids = readSift((char *)"../input/querySift/1.png.txt");
	printSift(siftCentroids, 'i');

	//Reads query sifts from file
	std::vector<Sift> imSiftsQuery = readMultiSifts(vquery);
	//Reads database sifts from file
	std::vector<Sift> imSiftsDatabase = readMultiSifts(vdatabase);
	
	//Creates query histograms CUDA
	vector<Histogram> queryHistograms = createHistogramCudaAux(siftCentroids, imSiftsQuery,vquery);
	//Creates database histograms CUDA
	vector<Histogram> databaseHistograms = createHistogramCudaAux(siftCentroids, imSiftsDatabase,vdatabase);
	
	//Freeing sift memory
	freeSift(siftCentroids);
	for(int i = 0; i < imSiftsQuery.size(); i++){
		freeSift(imSiftsQuery[i]);
	}
	for(int i = 0; i < imSiftsDatabase.size(); i++){
		freeSift(imSiftsDatabase[i]);
	}
	
	//Matching, and sorting the histograms
	vector<vector<int> > values = matchCudaAux(queryHistograms, databaseHistograms);
	cout << "Matched" << endl;
	
	//Computing results
	vector<vector<string> > results = comFinal(queryHistograms, databaseHistograms, values, 5);
	cout << "Results Computed" << endl;
	
	//Final analysis and results
	analize(results);
	
	//Free histograms
	freeHistograms(queryHistograms);
	freeHistograms(databaseHistograms);
	
	printf("Done!\n\n");
	return 0;
}

//---------------------------------------------//
//Free vector of histograms
//---------------------------------------------//
void freeHistograms(vector<Histogram> histos){
	for(int i = 0; i < histos.size(); i++){
		freeHisto(histos[i]);
	}
}

//---------------------------------------------//
//Analyses and displays final results
//---------------------------------------------//
void analize(vector<vector<string> >& results){
	
	printf("\t\t###########################################\n\t\t#########   Result Analysis   #############\n\t\t###########################################\n\n");	
	
	cout << "\n-------------------\nMatches\n-------------------\n"<< endl;
	int corrects = 0;
	for(int i = 0; i < results.size(); i++){
		string dbIm = results[i][0];
		string rightAns = extRight(dbIm);
		int right = 0;
		cout << "Database image: " << dbIm << "\nRight Answer: "<< rightAns << endl;
		for(int j = 1; j < results[i].size(); j++){
			if(rightAns == results[i][j]){
				cout << "\t==" << results[i][j] << endl;
				right = 1;
			}
			else{
				cout << "\t" << results[i][j] << endl;
			}
		}
		if(right == 1){
			corrects++;
		}
		cout << endl;
	}
	float accuracy = ((float)corrects)/((float)results.size()); 
	cout << "-------------------------------\n----Final Accuracy: " << accuracy << "-------\n-------------------------------\n" << endl;
}

//---------------------------------------------//
//Extracts right answer
//---------------------------------------------//
string extRight(string s){
	int lastSlash = s.length() - 1;
	
	while(lastSlash >= 0 && s[lastSlash] != '_'){
		lastSlash--;
	}
	
	return s.substr(0, lastSlash);
}

//---------------------------------------------//
//Computes final results
//---------------------------------------------//
vector<vector<string> > comFinal(vector<Histogram>& query, vector<Histogram>& database, vector<vector<int> >& values, int num){
	vector<vector<string> > ans;
	
	for(int i = 0; i < values.size(); i++){
		vector<string> temp;
		temp.push_back(extName(database[i].filename));
		//cout << extName(database[i].filename) << " ";
		for(int j = 0; j < num; j++){
			temp.push_back(extName(query[values[i][j]].filename));
			//cout << extName(query[values[i][j]].filename) << " ";
		}
		ans.push_back(temp);
		//cout << endl;
	}
	
	return ans;
}

//---------------------------------------------//
//Name extraction
//---------------------------------------------//
string extName(string s){
	int lastSlash = s.length() - 1;
	
	while(lastSlash >= 0 && s[lastSlash] != '/'){
		lastSlash--;
	}
	
	int count = 0;
	int dotAfterSlash = lastSlash;
	while(dotAfterSlash < s.length() && s[dotAfterSlash] != '.'){
		dotAfterSlash++;
		count++;
	}
	
	return s.substr(lastSlash+1, count - 1);
}

//---------------------------------------------//
//Cuda Creates Histogram
//---------------------------------------------//
__global__ void createHistCuda (float* siftCentroids, float* siftImage, int linesCent, int linesIm, float* temp)
{
	__shared__ float cosines[BLOCK_SIZE][2];
	
	size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	size_t idy = blockIdx.y;
	size_t tid = threadIdx.x;
	
	if(idx < linesCent){
		int centin = idx * 128;
		int imin = idy * 128;
		
		//Cosine similarity code ------------
		float sumab = 0;
		float suma2 = 0;
		float sumb2 = 0;

		for(int k = 0; k < 128; k++){
			sumab += siftCentroids[centin + k] * siftImage[imin + k];
			suma2 += siftImage[imin + k] * siftImage[imin + k];
			sumb2 += siftCentroids[centin + k] * siftCentroids[centin + k];  
		}	
		
		float cossim = sumab/(sqrtf(suma2)/sqrtf(sumb2));
		
		//debug[idy*linesCent + idx] = cossim;
		cosines[threadIdx.x][0] = cossim;
		cosines[threadIdx.x][1] = idx;
		
		__syncthreads();
		
		for (unsigned int s=blockDim.x/2; s>0; s>>=1) 
		{
			if (tid < s){
				size_t tid2 = tid + s;
				if(cosines[tid2][0] > cosines[tid][0]){
					cosines[tid][0] = cosines[tid2][0];
					cosines[tid][1] = cosines[tid2][1];
				}
			}
			__syncthreads();
		}
		
		if (tid == 0){
			temp[(blockIdx.y*gridDim.x + blockIdx.x)*2] = cosines[0][0];
			temp[(blockIdx.y*gridDim.x + blockIdx.x)*2+1] = cosines[0][1];
		}
		
	}

}

//---------------------------------------------//
//Histogram Matching CUDA
//---------------------------------------------//
__global__ void matchHistCuda(float*qSet, float*dbSet, size_t qSize, size_t dbSize, size_t hSize, float*out){
	size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	size_t idy = blockIdx.y*blockDim.y + threadIdx.y;
	
	if(idx < qSize && idy < dbSize){
		size_t qi = idx*hSize;
		size_t dbi = idy*hSize;
		
		//Cosine similarity code ------------
		float sumab = 0;
		float suma2 = 0;
		float sumb2 = 0;

		for(int k = 0; k < hSize; k++){
			sumab += qSet[qi+k] * dbSet[dbi+k];
			suma2 += qSet[qi+k] * qSet[qi+k];
			sumb2 += dbSet[dbi+k] * dbSet[dbi+k];
		}
		
		float cossim = sumab/(sqrtf(suma2)/sqrtf(sumb2));
		out[idy*qSize + idx] = cossim;
	}
}

//---------------------------------------------//
//Sorting each row
//---------------------------------------------//
__global__ void rowMin(float* input, int* output, size_t rowS, size_t rowNum){
	size_t id = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(id < rowNum){
		float temp[MAX_K/2][2];
		size_t inId = id * rowS;
		
		for(int i = 0; i< rowS;i++){
			temp[i][0] = input[inId + i];
			temp[i][1] = (float)i;
		}
		
		for(int i = 0; i< rowS; i++){
			float best = temp[i][0];
			int bestInd = i;
			for(int j = i; j < rowS; j++){
				if(temp[j][0] > best){
					best = temp[j][0];
					bestInd = j;
				}
			}
			float iVal = temp[i][0];
			float iInd = temp[i][1];
			temp[i][0] = temp[bestInd][0];
			temp[i][1] = temp[bestInd][1];
			temp[bestInd][0] = iVal;
			temp[bestInd][1] = iInd;
		}
		
		for(int i = 0; i< rowS; i++){
			output[inId+i] = (int)temp[i][1];
		}
	}
}

//---------------------------------------------//
//Histogram Matching Auxiliary
//Input: Query histograms and database histograms
//---------------------------------------------//
vector<vector<int> > matchCudaAux(vector<Histogram>& query, vector<Histogram>& database){
		//vector<vector<float> > chart;
	
	//Get constant sizes
	const size_t hSize = query[0].size;
	const size_t querySize = query.size();
	const size_t databaseSize = database.size();
	
	//Get the histogram info into 1D arrays
	float* h_qSet = (float*) malloc(querySize*hSize*sizeof(float));
	histogram2array(h_qSet, query);
	float* h_dbSet = (float*) malloc(databaseSize*hSize*sizeof(float));
	histogram2array(h_dbSet, database);
	
	//Malloc GPU memory for histograms
	float* d_qSet;
	cudaMalloc((void **) &d_qSet, querySize*hSize*sizeof(float));
	cudaMemcpy(d_qSet, h_qSet, querySize*hSize*sizeof(float),cudaMemcpyHostToDevice);
	float* d_dbSet;
	cudaMalloc((void **) &d_dbSet, databaseSize*hSize*sizeof(float));
	cudaMemcpy(d_dbSet, h_dbSet, databaseSize*hSize*sizeof(float),cudaMemcpyHostToDevice);

	//Malloc GPU memory for results array
	float * d_results;
	cudaMalloc((void **) &d_results, querySize*databaseSize*sizeof(float));
	
	dim3 grid((querySize + BLOCK_SIZE2 - 1)/BLOCK_SIZE2, (databaseSize+ BLOCK_SIZE2 - 1)/BLOCK_SIZE2);
	dim3 block(BLOCK_SIZE2,BLOCK_SIZE2);
	
	matchHistCuda<<<grid,block>>>(d_qSet, d_dbSet, querySize, databaseSize, hSize, d_results);
	
	//Copying results back
	float* out_results = (float*) malloc(querySize*databaseSize*sizeof(float));
	cudaMemcpy(out_results, d_results,querySize*databaseSize*sizeof(float),cudaMemcpyDeviceToHost);
	
		//arrayto2Dvec<float>(out_results, chart, querySize, databaseSize);
		//print_array<float*>(out_results, querySize, databaseSize);
		//print2DVec<float>(chart);
		
	//Free matchHistCuda memory
	free(h_qSet);
	free(h_dbSet);
	cudaFree(d_qSet);
	cudaFree(d_dbSet);
	
	vector<vector<int> > res;
	
	//Preparing input for row min, which is the same as the results of the last kernel
	float * d_input;
	cudaMalloc((void **) &d_input, querySize*databaseSize*sizeof(float));
	cudaMemcpy(d_input, out_results, databaseSize*querySize*sizeof(float),cudaMemcpyHostToDevice);
	
	//Preparing output for row min
	int * d_output;
	cudaMalloc((void **) &d_output, querySize*databaseSize*sizeof(int));
		
	dim3 grid2((databaseSize + BLOCK_SIZE - 1)/BLOCK_SIZE);
	dim3 block2(BLOCK_SIZE);
	
	rowMin<<<grid2,block2>>>(d_input, d_output, querySize, databaseSize);
	
	//Copying back results
	int * out_output = (int*) malloc(querySize*databaseSize*sizeof(int));
	cudaMemcpy(out_output, d_output,querySize*databaseSize*sizeof(int),cudaMemcpyDeviceToHost);
	
	//print_array<int*>(out_output, querySize, databaseSize);
	
	arrayto2Dvec<int>(out_output, res, querySize, databaseSize);
	
	//Freeing memory
	free(out_results);
	cudaFree(d_input);
	cudaFree(d_output);
	free(out_output);
	
	return res;
}

//---------------------------------------------//
//Print_array
//---------------------------------------------//
template<class T>
void print_array(T& array, int m, int n) {
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            std::cout << array[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
}

//---------------------------------------------//
//1D array to 2D vector
//---------------------------------------------//
template <class T> 
void arrayto2Dvec(T* array, vector<vector<T> >& vec, size_t x, size_t y){
	for(int j = 0; j < y;j++){
		vector<T> vecLine;
		for(int i = 0; i < x; i++){
			vecLine.push_back(array[j*x + i]);
		}
		vec.push_back(vecLine);
	}
}

//---------------------------------------------//
//Print 2D Vector
//---------------------------------------------//
template <class T> 
void print2DVec(vector<vector<T> >& vec){
	cout << "Vec x size: " << vec.size() << "\nVec y size: " << vec[0].size() << "\nType: " << typeid(vec[0][0]).name() << endl;
	for(size_t i = 0; i < vec.size(); i++){
		for(size_t j = 0; j < vec[0].size(); j++){
			cout << vec[i][j] << " ";
		}
		cout << endl;
	}
}

//---------------------------------------------//
//Histograms to 1D array
//---------------------------------------------//
void histogram2array(float* array, vector<Histogram>& histos){
	for(int i = 0; i < histos.size(); i++){
		for(int j = 0; j < histos[i].size;j++){
			array[i*histos[i].size + j] = histos[i].vals[j];
		}
	}
	
}


//---------------------------------------------//
//Reads in sifts
//---------------------------------------------//
vector<Sift> readMultiSifts(stringvec v){
	std::vector<Sift> imSifts;
	for(int i = 0; i < v.size(); i++){
		if(v[i] != "." && v[i] != ".." && v[i] != "all.txt"){
			char* fn = new char[v[i].size()+1];
			strcpy(fn,v[i].c_str());
			Sift siftIm = readSift(fn);
			imSifts.push_back(siftIm);
			delete[] fn;
		}
	}
	return imSifts;
}

//---------------------------------------------//
//Auxiliary creates histogram cuda
//Inputs: siftCentroids, vector of sift images, vector of filenames
//Output: vector of histograms
//---------------------------------------------//
vector<Histogram> createHistogramCudaAux(Sift& siftCentroids, vector<Sift>& imSifts, stringvec& v){
	
	vector<Histogram> histograms;
	
	//Passed Vectors
	float* d_cent;
	
	//Memory allocation for centroids
	int centS = siftCentroids.numSift;
	int centTotal = siftCentroids.totalNum;
	cudaMalloc((void **) &d_cent, centTotal*sizeof(float));
	cudaMemcpy(d_cent, siftCentroids.sift, centTotal*sizeof(float),cudaMemcpyHostToDevice);
	
	for(int i = 0; i < imSifts.size(); i++){
		//Memory allocation for image sifts
		float* d_im;
		int imS = imSifts[i].numSift;
		int imTotal = imSifts[i].totalNum;
		cudaMalloc((void **) &d_im, imTotal*sizeof(float));
		cudaMemcpy(d_im, imSifts[i].sift, imTotal*sizeof(float),cudaMemcpyHostToDevice);
		
		//Memory allocation for histogram
		float* hist = (float*) malloc(centS*sizeof(float));
		memset(hist,0,centS*sizeof(float));
		
		dim3 grid((centS+BLOCK_SIZE-1)/BLOCK_SIZE, imS);
		dim3 block(BLOCK_SIZE);
		
		//Temporary array creation
		float* d_temp;
		cudaMalloc((void **) &d_temp, imS*(centS+BLOCK_SIZE-1)/BLOCK_SIZE*2*sizeof(float));
		cudaMemset(d_temp, 0, imS*(centS+BLOCK_SIZE-1)/BLOCK_SIZE*2*sizeof(float));
		
		//Runs cuda code
		createHistCuda<<<grid,block>>>(d_cent,d_im,centS,imS,d_temp);
		
		//Copy temp back
		size_t tempS = imS*(centS+BLOCK_SIZE-1)/BLOCK_SIZE*2;
		float* out_temp = (float*) malloc(tempS*sizeof(float));
		cudaMemcpy(out_temp, d_temp,tempS*sizeof(float),cudaMemcpyDeviceToHost);
		
		size_t tempSx = imS;
		size_t tempSy = (centS+BLOCK_SIZE-1)/BLOCK_SIZE;
		
		for(size_t j = 0; j < tempSx; j++){
			float max = 0;
			int maxInd = 0;
			for(size_t k = 0; k < tempSy; k++){
				size_t ind = (j*tempSy + k)*2;
				if(out_temp[ind] > max){
					max = out_temp[ind];
					maxInd = (int)out_temp[ind+1];
				}
			}
			hist[maxInd] += 1.0f/(float)imS;
		}
		
		Histogram th = {.vals = hist, .size = centS, .filename = v[i]};
		histograms.push_back(th);
		
		cudaFree(d_im);
		cudaFree(d_temp);
		free(out_temp);
	}
	
	cudaFree(d_cent);
	return histograms;
}

//---------------------------------------------//
//Print directory files
//---------------------------------------------//
void printFiles(stringvec v){
	printf("\nDirectory List\nSize: %d\n", v.size());
	for(int i = 0; i < v.size(); i++){
		std::cout << "\tName: " << v[i] << "\n";
	}
	printf("\n\n");
}

//---------------------------------------------//
//Extracts Filenames in Directory
//---------------------------------------------//
void read_directory(const std::string& name, stringvec& v)
{
    DIR* dirp = opendir(name.c_str());
    struct dirent * dp;
    while ((dp = readdir(dirp)) != NULL) {
		string temp = dp->d_name;
		string s = name + temp;
		if(temp.compare(".") != 0 && temp.compare("..") != 0 && temp.compare("all.txt") != 0)
			v.push_back(s);
    }
    closedir(dirp);
}

//---------------------------------------------//
//Free Histogram Memory
//---------------------------------------------//
void freeHisto(Histogram h){
	free(h.vals);
}

//---------------------------------------------//
//Prints Histo
//Options: 'i' for info, 's' for sifts, 'b' for both
//---------------------------------------------//
void printHisto(Histogram h, char option){
	if(option == 'i' || option == 'b'){
		printf("----------------------\nSize: %d\n----------------------\n", h.size);
	}
	if(option == 's' || option == 'b'){
		//Prints the histogram
		for(int i = 0; i < h.size; i++){
			printf("%f ", h.vals[i]);
		}
	}
}

//---------------------------------------------//
//Free Sift Memory
//---------------------------------------------//
void freeSift(Sift sift){
	free(sift.sift);
}

//---------------------------------------------//
//Prints Sift
//Options: 'i' for info, 's' for sifts, 'b' for both
//---------------------------------------------//
void printSift(Sift sift, char option){
	if(option == 'i' || option == 'b'){
			printf("----------------------\nNum of Sifts: %d\nTotalSize: %d floats\n----------------------\n", sift.numSift, sift.totalNum);
	}
	if(option == 's' || option == 'b'){
		for(int i = 0; i < sift.numSift; i++){
			for(int j = 0; j < sift.elementSize; j++){
				printf("%f ", sift.sift[i*sift.elementSize + j]);
			}
		}
	}
}

//---------------------------------------------//
//Reads sift from file
//---------------------------------------------//
Sift readSift(char *filename){
	int lines = countlines(filename);
  	float * siftCentroids = (float *)malloc(lines * 128* sizeof(float));
	readFromFile(lines * 128, siftCentroids, filename);
	printf("-----Reading-Sift-----\nFilename: %s\nSift Number: %d\n----------------------\n", filename, lines);
	Sift sift = {.sift = siftCentroids, .numSift = lines, .totalNum = lines*128, .elementSize = 128};
	return sift;
}


//---------------------------------------------//
//Creates histogram for single centroid-image pair
//Description: Creates a histogram for an image siftImage based on centroids siftCentroids.
//Input: centroids, image sifts, histogram vector, number of centroids, number of sifts for image
//Output: Histogram of siftImage according to siftCentroids
//---------------------------------------------//
Histogram createHist(Sift cent, Sift im){
	float * siftCentroids = cent.sift;
	float * siftImage = im.sift;
	int linesCent = cent.numSift;
	int linesIm = im.numSift;
	
	float * histo = (float*) malloc(linesCent*sizeof(float));
	
	//Initializes histogram to 0	
	for(int i = 0; i < linesCent; i++){
		histo[i] = 0;
	}
	
	//For descriptor in image
	for(int i = 0; i < linesIm; i++){
		float bestCos = 0;
		int bestJ = 0;
		//For centroid descriptors, find the one that best matches the image descriptor
		for(int j = 0; j < linesCent; j++){
			int centin = j*128;
			int imin = i*128;
			
			//Cosine similarity code ------------
			float sumab = 0;
			float suma2 = 0;
			float sumb2 = 0;

			for(int k = 0; k < 128; k++){
				sumab += siftCentroids[centin + k] * siftImage[imin + k];
				suma2 += siftImage[imin + k] * siftImage[imin + k];
				sumb2 += siftCentroids[centin + k] * siftCentroids[centin + k];  
			}	
			
			float cossim = sumab/(sqrtf(suma2)/sqrtf(sumb2));

			if(cossim > bestCos){
				bestCos = cossim;
				bestJ = j;
			}
			//Cosine similarity end --------------
		}
		histo[bestJ] += 1;
	}

	//Normalization
	for(int i = 0; i < linesCent; i++){
                histo[i] = histo[i]/linesIm;
        }
    Histogram h = {.vals = histo, .size = linesCent, .filename = ""};
    return h;
}

//---------------------------------------------//
//Reading file into array
//---------------------------------------------//
void readFromFile(int size,float *v, char *file){
	FILE *fp;
	//cout << "Openning" << endl;
	fp = fopen(file,"r");

	int i=0;

	float t;
	while(i < size){
		if(fscanf(fp,"%f",&t)==EOF){
			printf("Error reading file\n");
			exit(1);
		}
		//printf("%f ", t);
		v[i++]=t;
	}
	//cout << "Read" << endl;
	fclose(fp);
}


//---------------------------------------------//
//Counting lines in file
//---------------------------------------------//
int countlines(char*filename)
{
  FILE *fp;
  int count = 0;  // Line counter (result)
  char c;  // To store a character read from file

  // Open the file
  fp = fopen(filename, "r");

  if (fp == NULL)
    {
        printf("Could not open file %s", filename);
        return 0;
    }
 
    // Extract characters from file and store in character c
    for (c = getc(fp); c != EOF; c = getc(fp))
        if (c == '\n') // Increment count if this character is newline
            count = count + 1;
 
    // Close the file
    fclose(fp);
    //printf("The file %s has %d lines\n ", filename, count);
    
    return count;
}

