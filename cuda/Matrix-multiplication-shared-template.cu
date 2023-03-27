//-*-C++-*-
#include<iostream>
#include<cuda.h>

// block size for the matrix 
#define BLOCK_SIZE 16

using namespace std;

// Device call (matrix multiplication)
__global__ void matrix_mul(const float *d_a, const float *d_b, 
			   float *d_c, int width)
{
  // Shared memory allocation for the block matrix  
  __shared__ int a_block[BLOCK_SIZE][BLOCK_SIZE];
  ...

  // Indexing for the block matrix
  int tx = threadIdx.x;
  ...

  // Indexing global matrix to block matrix 
  int row = threadIdx.x+blockDim.x*blockIdx.x;
  ...

  // Allow threads only for size of rows and columns (we assume square matrix)
  if ((row < width) && (col< width))
    {
      // Save temporary value for the particular index
      float temp = 0;
      for(int i = 0; i < width / BLOCK_SIZE; ++i)
	{
	  // Allign the global matrix to block matrix 
	  a_block[ty][tx] = d_a[row * width + (i * BLOCK_SIZE + tx)];
	  b_block[ty][tx] = d_b[(i * BLOCK_SIZE + ty) * width + col];

          // Make sure all the threads are synchronized
          ....

	  // Multiply the block matrix 
	  for(int j = 0; j < BLOCK_SIZE; ++j)
	    {
	      temp += a_block[ty][j] * b_block[j][tx];    
	    }
	  // Make sure all the threads are synchronized
          ...
	}
      // Save block matrix entry to global matrix 
      ...
    }
}

// Host call (matix multiplication)
float * cpu_matrix_mul(float *h_a, float *h_b, float *h_c, int width)   
{                                                                 
  for(int row = 0; row < width ; ++row)                           
    {                                                             
      for(int col = 0; col < width ; ++col)                       
        {                                                         
          float single_entry = 0;                                       
          for(int i = 0; i < width ; ++i)                         
            {                                                     
              single_entry += h_a[row*width+i] * h_b[i*width+col];      
            }                                                     
          h_c[row*width+col] = single_entry;                            
        }                                                         
    }   
  return h_c;           
}


int main()
{  
  cout << "Programme assumes that matrix size is N*N "<<endl;
  cout << "Matrix dimensions are assumed to be multiples of BLOCK_SIZE=16" << endl;
  cout << "Please enter the N size number "<< endl;
  int N=0;
  cin >> N;

  // Initialize the memory on the host
  float *a, *b, *c, *host_check;       
  
  // Initialize the memory on the device
  float *d_a, *d_b, *d_c; 
  
  // Allocate host memory
  a   = (float*)malloc(sizeof(float) * (N*N));
  b   = (float*)malloc(sizeof(float) * (N*N));
  c   = (float*)malloc(sizeof(float) * (N*N));
  host_check = (float*)malloc(sizeof(float) * (N*N));
  
  // Initialize host arrays
  for(int i = 0; i < (N*N); i++)
    {
      a[i] = 2.0f;
      b[i] = 2.0f;
    }
  
  // Allocate device memory
  cudaMalloc((void**)&d_a, sizeof(float) * (N*N));
  cudaMalloc((void**)&d_b, sizeof(float) * (N*N));
  cudaMalloc((void**)&d_c, sizeof(float) * (N*N));
  
  // Transfer data from host to device memory
  cudaMemcpy(d_a, a, sizeof(float) * (N*N), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, sizeof(float) * (N*N), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, c, sizeof(float) * (N*N), cudaMemcpyHostToDevice);
  
  // Thread organization
  dim3 Block_dim(BLOCK_SIZE, BLOCK_SIZE, 1);                
  ...
 
  // Device fuction call 
  matrix_mul<<<Grid_dim, Block_dim>>>(d_a, d_b, d_c, N);
  
  // Transfer data back to host memory
  cudaMemcpy(c, d_c, sizeof(float) * (N*N), cudaMemcpyDeviceToHost);

  // Cpu computation for verification 
  cpu_matrix_mul(a,b,host_check,N);

  // Verification
  bool flag=1;
  for(int i = 0; i < N; i++)
    {
      for(int j = 0; j < N; j++)
	{
	  if(c[j*N+i]!= host_check[j*N+i])
	    {
	      flag=0;
	      break;
	    }
	}
    }
  if (flag==0)
    {
      cout <<"But,two matrices are not equal" << endl;
      cout <<"Matrix dimensions are assumed to be multiples of BLOCK_SIZE=16" << endl;
    }
  else
    cout << "Two matrices are equal" << endl;
  
  // Deallocate device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  
  // Deallocate host memory
  free(a); 
  free(b); 
  free(c);
  free(host_check);
  
  return 0;
}
