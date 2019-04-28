#include<cuda.h>
#include<stdio.h>
#include<math.h>

__global__
void vecAddKernel(float* A, float* B, float* C, int n){
	//identify the index of the data to be read
	int i= threadIdx.x + blockDim.x * blockIdx.x;
	//calculate the sum and store
	if(i<n)
		C[i] = A[i] + B[i];
}

__host__
void vecAdd(float* A,float* B,float* C, int n){
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

	//call kernal function that the calculates sum and stores it in C
	vecAddKernel<<< ceil(n/256.0),256 >>>(d_A,d_B,d_C,n);		
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
	vecAdd(A,B,C,n);
	for(i=0;i<n;i++){
		printf("%f ",C[i]);	
	}
	free(A);
	free(B);
	free(C);
	return 0;
}




