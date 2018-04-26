#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#define NR_END 1
#define FREE_ARG char*



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

float **submatrix(float **a, long oldrl, long oldrh, long oldcl, long oldch,
                  long newrl, long newcl)
/* point a submatrix [newrl..][newcl..] to a[oldrl..oldrh][oldcl..oldch] */
{
  long i,j,nrow=oldrh-oldrl+1,ncol=oldcl-newcl;
  float **m;
  /* allocate array of pointers to rows */
  m=(float **) malloc((size_t) ((nrow+NR_END)*sizeof(float*)));
  if (!m) nrerror("allocation failure in submatrix()");
  m += NR_END;
  m -= newrl;
  /* set pointers to rows */
  for(i=oldrl,j=newrl;i<=oldrh;i++,j++) m[j]=a[i]+ncol;
  /* return pointer to array of pointers to rows */
  return m;
}


void free_submatrix(float **b, long nrl, long nrh, long ncl, long nch)
/* free a submatrix allocated by submatrix() */
{
free((FREE_ARG) (b+nrl-NR_END));
}


float **convert_matrix(float *a, long nrl, long nrh, long ncl, long nch)
/* allocate a float matrix m[nrl..nrh][ncl..nch] that points to the matrix
declared in the standard C manner as a[nrow][ncol], where nrow=nrh-nrl+1
and ncol=nch-ncl+1. The routine should be called with the address
&a[0][0] as the first argument. */
{
  long i,j,nrow=nrh-nrl+1,ncol=nch-ncl+1;
  float **m;
  /* allocate pointers to rows */
  m=(float **) malloc((size_t) ((nrow+NR_END)*sizeof(float*)));
  if (!m) nrerror("allocation failure in convert_matrix()");
  m += NR_END;
  m -= nrl;
  /* set pointers to rows */
  m[nrl]=a-ncl;
  for(i=1,j=nrl+1;i<nrow;i++,j++) m[j]=m[j-1]+ncol;
  /* return pointer to array of pointers to rows */
  return m;
}

void free_convert_matrix(float **b, long nrl, long nrh, long ncl, long nch)
/* free a matrix allocated by convert_matrix() */
{
free((FREE_ARG) (b+nrl-NR_END));
}


void rotate(float **a, int i, int j, int k, int l, float c, float s){
  float h;
  float g;
  g = a[i][j];
  h = a[k][l];
  a[i][j] = c*g + s*h;
  a[k][l] = -s*g+ c*h;
}

//void Jacobi_product(int P)




int main(){
  int *ip; // The first indice of the different P Jacobi Rotations
  int *iq; // The second indice of the different P Jacobi Rotations
  int P;  // The number of rotations considered in the program
  int N; // Number of matrix a which must be multiplied (to be implemented)
  int d; // dimension of the original matrix
  float *s; // Contains the sine values of the angle for the P rotations
  float *c; // Contains the cosine values of the angle for the P rotations
  int i,j;
  int step;
  d = 5;
  N = 1;
  P = 2;

  /// indices part
  ip = ivector(0,P);
  iq = ivector(0,P);

  ip[0] = 2;
  ip[1] = 3;
  iq[0] = 4;
  iq[1] = 1;

  // matrix part
  float ** a;

  // initial matrix definition
  a = matrix(0, d,0, d);
  for (i = 0; i< d; i++){
    for (j=0; j<d; j++){
      a[i][j] = 0.;
    }
  }

  a[0][0] = 1.;
  a[0][4] = 2.;
  a[4][0] = 2.;
  a[1][1] = 1.;
  a[2][2] = 8.;
  a[2][3] = 3.;
  a[3][2] = 3.;
  a[3][0] = 5.;
  a[0][3] = 5.;
  a[3][3] = 4.;

  // angle part
  float* theta; // contains the angle at each step
  theta = vector(0,P);
  c = vector(0,P);
  s = vector(0,P);


  theta[0] = M_PI/2.;
  theta[1] = M_PI/2.;


  for (i=0;i<P;i++){
    c[i] = cos(theta[i]);
    s[i] = sin(theta[i]);
  }

  printf(" a, theta, c, s, ip, iq initialized....\n ");

  printf("sanity check for theta= %f : c[1]= %f\n",theta[1], c[1]);
  printf("sanity check for theta = %f : s[1]= %f\n",theta[1], s[1]);


  for (i = 0; i < d; i++) {
        for (j = 0 ; j < d; j++) {
           printf("%f\t", a[i][j]);
        }
        printf("\n");
     }


  for (step = 0; step<P;step++){
    printf("Rotation n°%d out of %d \n",step, P-1);
    printf("(ip,iq) = (%d,%d) \n", ip[step],iq[step]);
    for (j=0; j< d; j ++){
      rotate(a,j,ip[step],j,iq[step],c[step],s[step]);
    }
    printf("rotated matrix is:\n");
    for (i = 0; i < d; i++) {
          for (j = 0 ; j < d; j++) {
             printf("%f\t", a[i][j]);
          }
          printf("\n");
       }
     }

  // freeing the Memory
  free_matrix(a, 0, d, 0, d);
  free_vector(s,0,P);
  free_vector(c,0,P);
  free_vector(theta,0,P);
  free_ivector(ip,0,P);
  free_ivector(iq,0,P);
  return 0;
}
