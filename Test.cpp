#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include "caffe/util/math_functions.hpp"
using namespace std;
using namespace caffe;

void init_mat(float *A, const int M, const int N) {
    for (unsigned int i = 0; i < M; ++i) {
         for (unsigned int j = 0; j < N; ++j) {
             A[i*N+j] = rand() % 1000;
	 }
    }
}

void usage(string prog) {
    printf("Usage: %s M N K", prog.c_str());
    exit(1);
}

int main(int argc, char* argv[]) {
    if (argc != 4) usage(argv[0]);
    const int M = atoi(argv[1]);
    const int N = atoi(argv[2]);
    const int K = atoi(argv[3]);

    float *h_A = (float*)malloc(M*K*sizeof(float));
    float *h_B = (float*)malloc(K*N*sizeof(float));
    float *h_C = (float*)malloc(M*N*sizeof(float));

    //init_mat(h_A,M,K);
    //init_mat(h_B,K,N);
 
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M*K*sizeof(float));
    cudaMalloc(&d_B, K*N*sizeof(float));
    cudaMalloc(&d_C, M*N*sizeof(float));

    cudaMemcpy(d_A,h_A,M*K*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,K*N*sizeof(float),cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    struct timespec start, end;
    long elapsed;

    cublasOperation_t cuTransA = CUBLAS_OP_N;
    cublasOperation_t cuTransB = CUBLAS_OP_T;
    const int lda = (cuTransA == CUBLAS_OP_N) ? K : M;
    const int ldb = (cuTransB == CUBLAS_OP_N) ? N : K;

    const float alpha = (float)1.;
    const float beta = (float)0.;
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_REALTIME, &start);
    CUBLAS_CHECK(cublasSgemm(handle, cuTransB, cuTransA,
			    N, M, K, &alpha, d_B, ldb, d_A, lda, &beta, d_C, N));
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_REALTIME, &end);
    elapsed = (end.tv_sec*1e9+end.tv_nsec)-(start.tv_sec*1e9+start.tv_nsec);
    printf("(%d,%d,%d) %10ld ns\n", M, N, K, elapsed);

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
}
