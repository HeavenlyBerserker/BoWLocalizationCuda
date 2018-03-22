

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include "string.h"
#include "cuda.h"

#define DEFAULT_THRESHOLD  4000

#define DEFAULT_FILENAME "BWstop-sign.ppm"

#define BLOCK_SIZE 16


__global__ void sobel_kernel(int * input, int * output, int xsize, int ysize, int thresh){
	
	

	int j = (blockIdx.x * blockDim.x) + threadIdx.x;
	int i = (blockIdx.y * blockDim.y) + threadIdx.y;
	unsigned int id = i*xsize + j;
	
	
	int magnitude = 0;
	//if(i-1 >= 0 && i+1 < ysize && j-1 >= 0  && j+1 < xsize){
		int sum1 = input[ xsize * (i-1) + j+1 ] -     input[ xsize*(i-1) + j-1 ] 
        	+ 2 * input[ xsize * (i)   + j+1 ] - 2 * input[ xsize*(i)   + j-1 ]
        	+     input[ xsize * (i+1) + j+1 ] -     input[ xsize*(i+1) + j-1 ];

		int sum2 = input[ xsize * (i-1) + j-1 ] + 
			2 * input[ xsize * (i-1) + j ]  + 
			input[ xsize * (i-1) + j+1 ]- 
			input[xsize * (i+1) + j-1 ] - 
			2 * input[ xsize * (i+1) + j ] - 
			input[ xsize * (i+1) + j+1 ];

		magnitude = sum1*sum1 + sum2*sum2;
	/*}
	else
		magnitude = 255;
	*/
	if(magnitude > thresh)
		output[id] = 255;
	else
		output[id] = 0;

	//output[id] = input[i*ysize + j];
	//output[id] = i;
}

unsigned int *read_ppm( char *filename, int & xsize, int & ysize, int & maxval ){
  
  if ( !filename || filename[0] == '\0') {
    fprintf(stderr, "read_ppm but no file name\n");
    return NULL;  // fail
  }

  fprintf(stderr, "read_ppm( %s )\n", filename);
  int fd = open( filename, O_RDONLY);
  if (fd == -1) 
    {
      fprintf(stderr, "read_ppm()    ERROR  file '%s' cannot be opened for reading\n", filename);
      return NULL; // fail 

    }

  char chars[1024];
  int num = read(fd, chars, 1000);

  if (chars[0] != 'P' || chars[1] != '6') 
    {
      fprintf(stderr, "Texture::Texture()    ERROR  file '%s' does not start with \"P6\"  I am expecting a binary PPM file\n", filename);
      return NULL;
    }

  unsigned int width, height, maxvalue;


  char *ptr = chars+3; // P 6 newline
  if (*ptr == '#') // comment line! 
    {
      ptr = 1 + strstr(ptr, "\n");
    }

  num = sscanf(ptr, "%d\n%d\n%d",  &width, &height, &maxvalue);
  fprintf(stderr, "read %d things   width %d  height %d  maxval %d\n", num, width, height, maxvalue);  
  xsize = width;
  ysize = height;
  maxval = maxvalue;
  
  unsigned int *pic = (unsigned int *)malloc( width * height * sizeof(unsigned int));
  if (!pic) {
    fprintf(stderr, "read_ppm()  unable to allocate %d x %d unsigned ints for the picture\n", width, height);
    return NULL; // fail but return
  }

  // allocate buffer to read the rest of the file into
  int bufsize =  3 * width * height * sizeof(unsigned char);
  if (maxval > 255) bufsize *= 2;
  unsigned char *buf = (unsigned char *)malloc( bufsize );
  if (!buf) {
    fprintf(stderr, "read_ppm()  unable to allocate %d bytes of read buffer\n", bufsize);
    return NULL; // fail but return
  }





  // TODO really read
  char duh[80];
  char *line = chars;

  // find the start of the pixel data.   no doubt stupid
  sprintf(duh, "%d\0", xsize);
  line = strstr(line, duh);
  //fprintf(stderr, "%s found at offset %d\n", duh, line-chars);
  line += strlen(duh) + 1;

  sprintf(duh, "%d\0", ysize);
  line = strstr(line, duh);
  //fprintf(stderr, "%s found at offset %d\n", duh, line-chars);
  line += strlen(duh) + 1;

  sprintf(duh, "%d\0", maxval);
  line = strstr(line, duh);


  fprintf(stderr, "%s found at offset %d\n", duh, line - chars);
  line += strlen(duh) + 1;

  long offset = line - chars;
  lseek(fd, offset, SEEK_SET); // move to the correct offset
  long numread = read(fd, buf, bufsize);
  fprintf(stderr, "Texture %s   read %ld of %ld bytes\n", filename, numread, bufsize); 

  close(fd);


  int pixels = xsize * ysize;
  for (int i=0; i<pixels; i++) pic[i] = (int) buf[3*i];  // red channel

 

  return pic; // success
}


void write_ppm( char *filename, int xsize, int ysize, int maxval, int *pic) 
{
  FILE *fp;
  
  fp = fopen(filename, "w");
  if (!fp) 
    {
      fprintf(stderr, "FAILED TO OPEN FILE '%s' for writing\n");
      exit(-1); 
    }
  //int x,y;
  
  
  fprintf(fp, "P6\n"); 
  fprintf(fp,"%d %d\n%d\n", xsize, ysize, maxval);
  
  int numpix = xsize * ysize;
  for (int i=0; i<numpix; i++) {
    unsigned char uc = (unsigned char) pic[i];
    fprintf(fp, "%c%c%c", uc, uc, uc); 
  }
  fclose(fp);

}




main( int argc, char **argv )
{

  int thresh = DEFAULT_THRESHOLD;
  char *filename;
  filename = strdup( DEFAULT_FILENAME);
  
  if (argc > 1) {
    if (argc == 3)  { // filename AND threshold
      filename = strdup( argv[1]);
       thresh = atoi( argv[2] );
    }
    if (argc == 2) { // default file but specified threshhold
      
      thresh = atoi( argv[1] );
    }

    fprintf(stderr, "file %s    threshold %d\n", filename, thresh); 
  }


  int xsize, ysize, maxval;
  unsigned int *pic = read_ppm( filename, xsize, ysize, maxval ); 


  int numbytes =  xsize * ysize * sizeof( int );
  int *result = (int *) malloc( numbytes );
  int *resultCuda = (int *) malloc( numbytes );
  if (!result) { 
    fprintf(stderr, "sobel() unable to malloc %d bytes\n", numbytes);
    exit(-1); // fail
  }

  int i, j, magnitude, sum1, sum2; 
  int *out = result;
  //int *outCuda = resultCuda;


//Vars and consts
int *deviceInput;
int *deviceOutput;
//Memory stuff-------------------------------------------------
//Allocation
cudaMalloc(&deviceInput, numbytes);
cudaMalloc(&deviceOutput, numbytes);
//Copy to GPU
cudaMemcpy(deviceInput, pic, numbytes,cudaMemcpyHostToDevice);
cudaMemcpy(deviceOutput, result, numbytes, cudaMemcpyHostToDevice);
//Initialize grid and block dimensions
//int numBlocks = (xsize*ysize + BLOCK_SIZE - 1)/BLOCK_SIZE;
//int numBlocks = (xsize*ysize)/BLOCK_SIZE;
dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
int grSx = (xsize+BLOCK_SIZE-1)/BLOCK_SIZE;
int grSy = (ysize+BLOCK_SIZE-1)/BLOCK_SIZE;
dim3 gridSize(grSx,grSy);
  //Timing
  cudaEvent_t start,stop;
  float elapsed_time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

//Launch GPU kernels
sobel_kernel<<<gridSize, blockSize>>>(deviceInput, deviceOutput, xsize, ysize, thresh);


  //Timing
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);        
  cudaEventElapsedTime(&elapsed_time,start, stop);

//Copy device to host
cudaMemcpy(resultCuda, deviceOutput, numbytes, cudaMemcpyDeviceToHost);


//Code check
int match = 0;
int nonZeros = 0;
int realNonZ = 0;
//CPU Code-----------------------------------------------------
  for (int col=0; col<ysize; col++) {
    for (int row=0; row<xsize; row++) { 
      *out++ = 0;
      //*outCuda++ = 0;
    }
  }

  for (i = 1;  i < ysize - 1; i++) {
    for (j = 1; j < xsize -1; j++) {

      int offset = i*xsize + j;

      sum1 =  pic[ xsize * (i-1) + j+1 ] -     pic[ xsize*(i-1) + j-1 ] 
        + 2 * pic[ xsize * (i)   + j+1 ] - 2 * pic[ xsize*(i)   + j-1 ]
        +     pic[ xsize * (i+1) + j+1 ] -     pic[ xsize*(i+1) + j-1 ];
      
      sum2 = pic[ xsize * (i-1) + j-1 ] + 2 * pic[ xsize * (i-1) + j ]  + pic[ xsize * (i-1) + j+1 ]
            - pic[xsize * (i+1) + j-1 ] - 2 * pic[ xsize * (i+1) + j ] - pic[ xsize * (i+1) + j+1 ];
      
      magnitude =  sum1*sum1 + sum2*sum2;

      if (magnitude > thresh)
        result[offset] = 255;
      else 
        result[offset] = 0;
	
      //result[offset] = i;

      if(result[offset] != resultCuda[offset]){
         match++;
	if(offset > 90000 && offset < 100000)
	 //printf("Correct = %d, Cuda = %d\n",result[offset], resultCuda[offset]);
	int hi;
	}
	
	
	if(resultCuda[offset] != 0)
		nonZeros++;
	if(result[offset] != 0)
		realNonZ++;	
      
    }
  }
//---------------------------------------------------------------
  write_ppm((char*)"resultCuda.ppm",xsize, ysize, 255, resultCuda);
  write_ppm( (char*)"result.ppm", xsize, ysize, 255, result);
	
//Freeing cuda
cudaFree(deviceInput);
cudaFree(deviceOutput);
  //printf("Number of nonZeros = %d, Real = %d\n",nonZeros, realNonZ);
  printf("Number of wrong entries = %d, Time = %2.6f\n", match,  elapsed_time);
  if(match == 0){
	printf("GPU successful, Time = %2.6f\n", match,  elapsed_time);
}
else{
	printf("GPU Failed.\n");
}

  fprintf(stderr, "sobel done\n"); 

}

