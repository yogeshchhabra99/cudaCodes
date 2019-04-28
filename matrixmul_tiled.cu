#include<cuda.h>
#include<stdio.h>
#include<math.h>
#define TILEWIDTH 32
__global__
void vecMulMatrixKernel(float* A, float* B, float* C, int n){
	//each block loads the corresponding row of blocks of A matrix and column of blocks of B matrix, one block at a time and then clculates the product for that part then product of a  the parts is added.
	// each thread loads 2 elements one from A and one from B in each phase
	// there are total gridDim.x phases
	// the element loaded is the element at the same position as this thread but in a different block
	//if run more thread than max then not run
	int tx=threadIdx.x; int ty=threadIdx.y;
	int bx=blockIdx.x; int by=blockIdx.y;
	
	int row=by*blockDim.y+ty;
	int col=bx*blockDim.x+tx;
	
	__shared__ float Ads[TILEWIDTH][TILEWIDTH];
	__shared__ float Bds[TILEWIDTH][TILEWIDTH];
	
	if(row<n && col <n){
		int i; float val=0.0;
		for(i=0;i<gridDim.x-1;i++){
			Ads[ty][tx] = A[ row*n + i*TILEWIDTH + tx];
			Bds[ty][tx] = B[ (i*TILEWIDTH + ty)*n + col];
			__syncthreads();
			for(int k=0;k<TILEWIDTH;k++){
				val+= Ads[ty][k]*Bds[tx][k];	
			}
			__syncthreads();
			
		}

		if(i*TILEWIDTH + tx <n )		//if n was a multiple of blockDim then this was not required
			Ads[ty][tx] = A[ row*n + i*TILEWIDTH + tx];
		if(i*TILEWIDTH + ty <n )
			Bds[ty][tx] = B[ (i*TILEWIDTH + ty)*n + col];
		__syncthreads();
		int m =n%TILEWIDTH;
		if(m==0)
			m=TILEWIDTH;
		for(int k=0;k<m;k++){//printf("add");
			val+= Ads[ty][k]*Bds[tx][k];	
		}
		__syncthreads();
		C[row*n + col]= val;
	}
}

int min2Power(int x){
	int res=1;
	while(res<x){
		res*=2;	
	}
	return res/2;
}

__host__
void vecMulMatrix(float* A,float* B,float* C, int n){
	int size = n * n * sizeof(float);
	float *d_A, *d_B, *d_C;
	
	//Allocate device memory for A,B,C
	cudaMalloc((void**)&d_A, size);
	cudaMalloc((void**)&d_B, size);
	cudaMalloc((void**)&d_C, size);

	//copy A,B to device memory
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
	
	//call kernal function that the calculates the product and stores it in C
	dim3 dimBlock(TILEWIDTH,TILEWIDTH,1);
	dim3 dimGrid(ceil(n/(float)TILEWIDTH),ceil(n/(float)TILEWIDTH),1);
	vecMulMatrixKernel<<<dimGrid,dimBlock >>>(d_A,d_B,d_C,n);		

	//copy C from devce memory
	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
	
	//free device memories
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

//Kernal function that runs in each thread


int main(){
	int n=10;
	int i,j;
	float A[n][n],C[n][n],B[n][n];
	
	for(i=0;i<n;i++){
		for(j=0;j<n;j++){
			A[i][j]=i+j;
			B[i][j]=i*j;
		}
	}
	
	vecMulMatrix(&A[0][0],&B[0][0],&C[0][0],n);
	for(i=0;i<n;i++){
		for(j=0;j<n;j++){
			printf("%.3f ",A[i][j]);
		}
		printf("\n");
	}
	printf("---\n");
	for(i=0;i<n;i++){
		for(j=0;j<n;j++){
			printf("%.3f ",B[i][j]);
		}
		printf("\n");
	}
	printf("---\n");
	for(i=0;i<n;i++){
		for(j=0;j<n;j++){
			printf("%.3f ",C[i][j]);
		}
		printf("\n");
	}
	
	return 0;
}




