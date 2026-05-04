#include "cusparse_baseline.cuh"

#include <cusparse.h>

#define CHECK_CUSPARSE(x)                                                                   \
    {                                                                                       \
        cusparseStatus_t status = (x);                                                      \
        if (status != CUSPARSE_STATUS_SUCCESS)                                              \
        {                                                                                   \
            fprintf(stderr, "cuSPARSE error %s at %s:%d\n", cusparseGetErrorString(status), \
                    __FILE__, __LINE__);                                                    \
            exit(EXIT_FAILURE);                                                             \
        }                                                                                   \
    }

void cusparse_spmm_csr(CSR *A_csr, half *B, float *C, int N, int n_iter, double *elapsed)
{
    const int M = A_csr->nRow;
    const int K = A_csr->nCol;

    cusparseHandle_t handle = nullptr;
    cusparseSpMatDescr_t matA = nullptr;
    cusparseDnMatDescr_t matB = nullptr;
    cusparseDnMatDescr_t matC = nullptr;
    void *dBuffer = nullptr;
    size_t bufferSize = 0;

    int *drow_offset = nullptr;
    int *dcol_idx = nullptr;
    half *dA_val = nullptr;
    half *dB = nullptr;
    float *dC = nullptr;

    cudaEvent_t start, end;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&end));

    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&dA_val), sizeof(half) * A_csr->nnz));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&dcol_idx), sizeof(int) * A_csr->nnz));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&drow_offset), sizeof(int) * (M + 1)));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&dB), sizeof(half) * K * N));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&dC), sizeof(float) * M * N));

    CHECK_CUDA(cudaMemcpy(dA_val, A_csr->value, sizeof(half) * A_csr->nnz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dcol_idx, A_csr->col_idx, sizeof(int) * A_csr->nnz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(drow_offset, A_csr->row_offset, sizeof(int) * (M + 1), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, B, sizeof(half) * K * N, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dC, C, sizeof(float) * M * N, cudaMemcpyHostToDevice));

    CHECK_CUSPARSE(cusparseCreate(&handle));
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, M, K, A_csr->nnz,
                                     drow_offset, dcol_idx, dA_val,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F));
    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, K, N, N, dB, CUDA_R_16F, CUSPARSE_ORDER_ROW));
    CHECK_CUSPARSE(cusparseCreateDnMat(&matC, M, N, N, dC, CUDA_R_32F, CUSPARSE_ORDER_ROW));

    float alpha = 1.0f;
    float beta = 0.0f;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, matA, matB, &beta, matC,
                                           CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize));
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    for (int i = 0; i < 5; i++)
    {
        CHECK_CUSPARSE(cusparseSpMM(handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA, matB, &beta, matC,
                                    CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    float measured_time = 0.0f;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < n_iter; i++)
    {
        CHECK_CUSPARSE(cusparseSpMM(handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA, matB, &beta, matC,
                                    CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));
    }
    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaEventSynchronize(end));
    CHECK_CUDA(cudaEventElapsedTime(&measured_time, start, end));

    CHECK_CUDA(cudaMemcpy(C, dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
    if (elapsed != nullptr)
    {
        *elapsed = static_cast<double>(measured_time) / static_cast<double>(n_iter);
    }

    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matC));
    CHECK_CUSPARSE(cusparseDestroy(handle));
    CHECK_CUDA(cudaFree(dBuffer));
    CHECK_CUDA(cudaFree(dA_val));
    CHECK_CUDA(cudaFree(drow_offset));
    CHECK_CUDA(cudaFree(dcol_idx));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(end));
}
