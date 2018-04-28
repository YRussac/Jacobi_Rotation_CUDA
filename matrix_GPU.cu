// Code to handle Matrix on the CUDA GPU
// Should allows us to transmit
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#define NR_END 0
#define FREE_ARG char*


void testCUDA(cudaError_t error, const char *file, int line){
  if (error != cudaSuccess){
    printf("Error in file %s at line %d \n", file , line);
    exit(EXIT_FAILURE);
  }
}

// Has to be define in the compilation in order to get the correct value of
// of the values __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__,__LINE__))




void nrerror(char error_text[])
/* standard error handler */
{
  fprintf(stderr, "Numerical Recipes run-time error ... \n");
  fprintf(stderr, "%s \n", error_text);
  fprintf(stderr, "... now exiting to system... \n");
  exit(1);
}



float **matrix(long nrl, long nrh, long ncl, long nch)
/* allocate a float matrix with subscript range m[nrl...nrh][ncl...nch] */
{
  long i, nrow = nrh-nrl+1, ncol = nch-ncl+1;
  float **m;
  // Allocate pointers to rows
  m = (float **) malloc((size_t) ((nrow + NR_END)*sizeof(float*)));
  if (!m) nrerror("allocaztion failure 1 in matrix()");
  m += NR_END;
  m -= nrl;

  // Allocate rows and set pointers to them
  m[nrl] = (float*) malloc((size_t)((nrow*ncol+NR_END)*sizeof(float)));
  if (!m[nrl]) nrerror("allocation failure 2 in matrix()");
  m[nrl] += NR_END;
  m[nrl] -= ncl;

  for (i=nrl+1;i<=nrh;i++) m[i] = m[i-1] + ncol;
  // return pointer to array of pointers to rows
  return m;
}

// allocation of a 2D array on device (GPU machine)


// l'info de la ligne i dans cpuArray est désormais contenue dans la ligne
// i de h_temp

__global__ void aff_matrix_GPU(float ** device2DArray, int d){
  // pour faire le print d'une matrice dans le GPU
  printf("------");
  printf("In the kernel\n");
  int i=0;
  int j=0;
  for (i = 0; i < d; i++) {
        for (j = 0 ; j < d; j++) {
           printf("%f\t", device2DArray[i][j]);
        }
        printf("\n");
     }

   for (i = 0; i < d; i++) {
         for (j = 0 ; j < d; j++) {
           device2DArray[i][j] += 3;
      }
      }
  printf("\n");
  for (i = 0; i < d; i++) {
        for (j = 0 ; j < d; j++) {
           printf("%f\t", device2DArray[i][j]);
        }
        printf("\n");
     }
    printf("End of the kernel\n");
    printf("------");
   }

int main(){
  float **m;
  int d = 5;
  int i,j;
  m = matrix(0,5,0,5);

  m[0][0] = 1.;
  m[0][4] = 2.;
  m[4][0] = 2.;
  m[1][1] = 1.;
  m[2][2] = 8.;
  m[2][3] = 3.;
  m[3][2] = 3.;
  m[3][0] = 5.;
  m[0][3] = 5.;
  m[3][3] = 4.;

  float **device2DArray; // GPU array
  float *h_temp[d];
  cudaMalloc((void **)&device2DArray, d*sizeof(float *));
  for(int i=0; i<d; i++){
     cudaMalloc( (void **)&h_temp[i], d*sizeof(float));
    }
  cudaMemcpy(device2DArray, h_temp, d*sizeof(float *), cudaMemcpyHostToDevice);

  // Copy of a host ** array on the device
  float **cpuArray;
  cpuArray = m;
  for(int i=0; i<d; i++)
  {
     cudaMemcpy(h_temp[i], cpuArray[i], d*sizeof(float), cudaMemcpyHostToDevice);
  }

  aff_matrix_GPU<<<1,1>>>(device2DArray,d);

  // float **host_mat;
  // host_mat = matrix(0,5,0,5);
  // float **cpuArray = host_mat;
  //Once done, to copy out the data back to host:
  for(int i=0; i<d; i++)
  {
     cudaMemcpy(cpuArray[i], h_temp[i], d*sizeof(float), cudaMemcpyDeviceToHost);
  }

  printf("Reception of the data\n");
  for (i = 0; i < d; i++) {
        for (j = 0 ; j < d; j++) {
           printf("%f\t", cpuArray[i][j]);
        }
        printf("\n");
     }
  return 0;
}
