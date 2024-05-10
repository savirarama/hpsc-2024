#include <cstdio>
#include <cstdlib>

__global__ void zeros_bucket(int *bucket){
    bucket[threadIdx.x]=0;
}

__global__ void reduce_bucket(int *bucket, int *key){
    atomicAdd(&bucket[key[threadIdx.x]],1);
}

__global__ void sort_bucket(int *bucket, int *key, int *offset) {
    int i=threadIdx.x;
    for (int j=offset[i]; j<offset[i] + bucket[i]; j++) {
        key[j]=i;
    }
}

int main() {
    int n = 50;
    int range = 5;
    
    int *key;
    cudaMallocManaged(&key, n * sizeof(int));
    for (int i=0; i<n; i++) {
        key[i] = rand() % range;
        printf("%d ", key[i]);
    }
    printf("\n");

    int *bucket;
    cudaMallocManaged(&bucket, range * sizeof(int));
    int *offset;
    cudaMallocManaged(&offset, range * sizeof(int));

    zeros_bucket<<<1,range>>>(bucket);
    cudaDeviceSynchronize();

    reduce_bucket<<<1,n>>>(bucket, key);
    cudaDeviceSynchronize();

    for (int i=1; i<range; i++) 
        offset[i] = offset[i-1] + bucket[i-1];
        
    sort_bucket<<<1,range>>>(bucket, key, offset);
    cudaDeviceSynchronize();

    for (int i=0; i<n; i++) {
        printf("%d ", key[i]);
    }
    printf("\n");

    cudaFree(key);
    cudaFree(bucket);
    cudaFree(offset);

    return 0;
}
