#include<cuda.h>
#include<stdio.h>
#include<math.h>

#define TILEWIDTH 32

__global__
void vecConvKernel(float* A, float* B, float* C, int n){
	//identify the index of the data to be read
	int tx=threadIdx.x; 
	int bx=blockIdx.x;
	
	int index=bx*blockDim.x+tx;
	
	__shared__ float Ads[TILEWIDTH];
	__shared__ float Bds[2*TILEWIDTH];
	//assuming n is multiple of TILEWIDTH
//	if(index<n){
		int i; float val=0.0;
		for(i=0;i<gridDim.x-1;i++){
			Ads[tx] = A[i*TILEWIDTH+tx];
			Bds[tx] = B[i*TILEWIDTH+tx];
			Bds[TILEWIDTH + tx] = B[(i+1)*TILEWIDTH + tx];
			
			__syncthreads();
			for(int k=0;k<TILEWIDTH;k++){
				val+= Ads[k]*Bds[tx+k];	
			}
			__syncthreads();
		}
		Ads[tx] = A[i*TILEWIDTH + tx];
		Bds[tx] = B[i*TILEWIDTH+tx];
		Bds[TILEWIDTH + tx] = B[tx];
		__syncthreads();
		for(int k=0;k<TILEWIDTH;k++){
			val+= Ads[k]*Bds[tx+k];	
		}
		__syncthreads();
		C[index] = val;
//	}
}

__host__
void vecConv(float* A,float* B,float* C, int n){
	int c=ceil(n/256.0);
	int size = n * sizeof(float);
	float *d_A, *d_B, *d_C;

	//Allocate device memory for A,B,C
	cudaMalloc((void**)&d_A, size);
	cudaMalloc((void**)&d_B, size);
	cudaMalloc((void**)&d_C, size);	

	//copy A,B to device memory
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

	dim3 dimBlock(TILEWIDTH,1,1);
	dim3 dimGrid(ceil(n/(float)TILEWIDTH),1,1);
	
	//call kernal function that the calculates sum and stores it in C
	vecAddKernel<<< dimGrid,dimBlock >>>(d_A,d_B,d_C,n);		
	//the y and z dimensions are set to 1 by default


	//copy C from devce memory
	cudaMemcpy( C,d_C, size, cudaMemcpyDeviceToHost);
	
	//free device memories
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

//Kernal function that runs in each thread


int main(){
	float *A,*B,*C;
	int n=10;
	A=(float*)malloc(n*sizeof(float));
	B=(float*)malloc(n*sizeof(float));
	C=(float*)malloc(n*sizeof(float));
	int i;
	for(i=0;i<n;i++){
		A[i]=(float)i;
		B[i]=(float)2*i;	
	}
	vecConv(A,B,C,n);
	for(i=0;i<n;i++){
		printf("%f ",C[i]);	
	}
	return 0;
}




