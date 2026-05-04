#ifndef _CORE_SPMM_CUH
#define _CORE_SPMM_CUH

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(f)                                                                                             \
    {                                                                                                             \
        cudaError_t err = (f);                                                                                    \
        if (err != cudaSuccess)                                                                                   \
        {                                                                                                         \
            fprintf(stderr, "CUDA error at [%s : %d] %d %s\n", __FILE__, __LINE__, err, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                                   \
        }                                                                                                         \
    }

#define WARP_SIZE 32
#define VERBOSE false

#define VERBOSE_PRINT(...)   \
    if (VERBOSE == true)     \
    {                        \
        printf(__VA_ARGS__); \
    }

#define VERBOSE_PUTS(...)  \
    if (VERBOSE == true)   \
    {                      \
        puts(__VA_ARGS__); \
    }

typedef struct CSR
{
    int nRow;
    int nCol;
    int nnz;
    int *row_offset;
    int *col_idx;
    half *value;
} CSR;

__global__ void warm_up_gpu_kernel();

void destroy_csr(CSR *mat);

// Computes C = A_csr * B.
// A_csr: host-side CSR matrix with FP16 values and zero-based column indices.
// B: host-side row-major dense matrix of shape [A_csr->nCol, N], FP16.
// C: host-side row-major dense matrix of shape [A_csr->nRow, N], FP32.
void our_spmm_balanced(CSR *A_csr, half *B, float *C, int N, int n_iter, double *elapsed = nullptr);

#endif
