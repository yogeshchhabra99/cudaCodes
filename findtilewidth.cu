//For a 2d matrix

#include<cuda.h>
#include<stdio.h>
#include<math.h>
#define TILEWIDTH 

int min2Power(int x){
	int res=1;
	while(res<x){
		res*=2;	
	}
	return res/2;
}

__host__
void tileWidth(){
	int i,deviceCount;
	cudaGetDeviceCount(&deviceCount);
	int maxThreadsPerBlock;
	int sharedMemory;
	for(i=0;i<deviceCount;i++){
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp,i);
		maxThreadsPerBlock=devProp.maxThreadsPerBlock;
		sharedMemory=devProp.sharedMemPerBlock;
	}
	// assuming 2-d shared mem array of size Tilewidth*TileWidth
	//my tile width will also become the block size thats why it should not be more than num of threads a block can have
	//TileWidth*TileWidth<max_threads_per_block
	//each thread loads 2 floats i.e. 8 bytes
	//8*num_of_threads<sharedmemory
	//TileWidth*tileWidth<sharedmem/8
	int tileWidth = min((int)sqrt(maxThreadsPerBlock), (int)min2Power(sqrt(sharedMemory/8)));
	tileWidth = max(tileWidth,16);
	printf("%d\n",tileWidth);
}



int main(){
	tileWidth();
	return 0;
}




