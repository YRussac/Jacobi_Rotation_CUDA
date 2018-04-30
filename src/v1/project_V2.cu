#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#define NR_END 1
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


float *vector(long nl, long nh)
/* allocate a float vector with subscript range v[n1,....,nh] in case of need */
{
  float *v;
  v = (float *)malloc((size_t)  ((nh-nl + 1 + NR_END)*sizeof(float)));
  if (!v) nrerror("allocation failure in vector()");
  return v-nl+NR_END; // Faire decalage si on veut être indicé sur autre chose
  // que [0..length(vector)-1]
}

void free_vector(float *v, long nl, long nh)
/* free a float vector allocated with vector() */
{
free((FREE_ARG) (v+nl-NR_END));
}


int *ivector(long nl, long nh)
/* allocate a float vector with subscript range v[n1,....,nh] in case of need */
{
  int *v;
  v = (int *)malloc((size_t)  ((nh-nl + 1 + NR_END)*sizeof(int)));
  if (!v) nrerror("allocation failure in ivector()");
  return (v-nl+NR_END); // Faire decalage si on veut être indicé sur autre chose
  // que [0..length(vector)-1]
}

void free_ivector(int *v, long nl, long nh)
/* free an int vector allocated with ivector() */
{
free((FREE_ARG) (v+nl-NR_END));
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

void free_matrix(float **m, long nrl, long nrh, long ncl, long nch)
/* free a float matrix allocated by matrix() */
{
free((FREE_ARG) (m[nrl]+ncl-NR_END));
free((FREE_ARG) (m+nrl-NR_END));
}



void rotate(float **a, int i, int j, int k, int l, float c, float s){
  float h;
  float g;
  g = a[i][j];
  h = a[k][l];
  a[i][j] = c*g - s*h;
  a[k][l] = s*g+ c*h;
}

void Jacobi_product(float **a, int d, int P){
  int *ip; // The first indice of the different P Jacobi Rotations
  int *iq; // The second indice of the different P Jacobi Rotations
  int i,j;
  int step;
  int rand1;
  int rand2;

  ip = ivector(0,P);
  iq = ivector(0,P);

  float* theta; // contains the angle at each step

  theta = vector(0,P);
  float *s; // Contains the sine values of the angle for the P rotations
  float *c; // Contains the cosine values of the angle for the P rotations


  c = vector(0,P);
  s = vector(0,P);

  // TODO generate uniform sur 0,2Pi de longueur P
  double temp;

  for (i=0;i<P; i ++){
    temp = (double)(2*M_PI*rand())/(double)(RAND_MAX);
    theta[i] = temp;
    c[i] = cos(theta[i]);
    s[i] = sin(theta[i]);
    rand1 = rand()%d;
    rand2 = rand()%d;
    while (rand2 == rand1){
      rand2 = rand()%d;
    }
    ip[i] = rand1;
    iq[i] = rand2;
  }


  for (step = 0; step<P;step++){
    for (j=0; j< d; j ++){
      rotate(a,ip[step],j,iq[step],j,c[step],s[step]);
    }
   }


     // freeing the memory
     free_vector(s,0,P);
     free_vector(c,0,P);
     free_vector(theta,0,P);
     free_ivector(ip,0,P);
     free_ivector(iq,0,P);
}

void fullCPU_entire(int P, int N, int d){
  int count;
  int nb_mat;
  int i,j;
  cudaDeviceProp prop;
  testCUDA(cudaGetDeviceCount(&count));
  testCUDA(cudaGetDeviceProperties(&prop, count-1));
  float TimerV;
  cudaEvent_t start, stop;
  testCUDA(cudaEventCreate(&start));
  testCUDA(cudaEventCreate(&stop));
  testCUDA(cudaEventRecord(start,0));

  float ** a;

  for (nb_mat = 0; nb_mat<N;nb_mat++){
    // Random Matrix generation
    printf("Treatment of matrix %d out of %d \n", nb_mat,N);
    a = matrix(0, d,0, d);
    for (i = 0; i< d; i++){
      for (j=0; j<d; j++){
        a[i][j] = (double)(10.*rand())/(double)(RAND_MAX);
      }
    }
    Jacobi_product(a, d, P);

    // freeing the memory
    free_matrix(a, 0, d, 0, d);
  }

  testCUDA(cudaEventRecord(stop,0));
  testCUDA(cudaEventSynchronize(stop));
  testCUDA(cudaEventElapsedTime(&TimerV,start,stop));
  printf("Execution time: %f ms\n", TimerV);
}


int main(){

  int P;  // The number of rotations considered in the program
  int N; // Number of matrix a which must be multiplied (to be implemented)
  int d; // dimension of the original matrix
  d = 10;
  N = 1000;
  P = 1000;

  fullCPU_entire(P, N, d);

  return 0;
}
