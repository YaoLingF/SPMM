#include "core_spmm.cuh"
#include "cusparse_baseline.cuh"
#include "mtx_to_csr.cuh"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <random>
#include <stdio.h>
#include <string>

namespace
{
void prepare_dense_b(half **B, int K, int N)
{
    *B = static_cast<half *>(malloc(sizeof(half) * K * N));
    std::mt19937 gen(1);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < K * N; i++)
    {
        (*B)[i] = __float2half(dist(gen));
    }
}

bool compare_result(const float *gold, const float *test, int count, float tol)
{
    bool ok = true;
    int printed = 0;
    float max_abs = 0.0f;
    float max_rel = 0.0f;

    for (int i = 0; i < count; i++)
    {
        float abs_err = fabsf(gold[i] - test[i]);
        float denom = fmaxf(fabsf(gold[i]), 1.0f);
        float rel_err = abs_err / denom;
        max_abs = fmaxf(max_abs, abs_err);
        max_rel = fmaxf(max_rel, rel_err);

        if (abs_err > tol && rel_err > tol)
        {
            ok = false;
            if (printed < 10)
            {
                printf("mismatch[%d]: cusparse=%f our=%f abs=%f rel=%f\n",
                       i, gold[i], test[i], abs_err, rel_err);
                printed++;
            }
        }
    }

    printf("max_abs=%f max_rel=%f\n", max_abs, max_rel);
    return ok;
}
} // namespace

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printf("Usage: %s matrix.mtx [N=256] [n_iter=10]\n", argv[0]);
        return 1;
    }

    int N = argc > 2 ? std::stoi(argv[2]) : 256;
    int n_iter = argc > 3 ? std::stoi(argv[3]) : 10;
    if (N <= 0)
    {
        fprintf(stderr, "N must be positive\n");
        return 1;
    }
    if (n_iter <= 0)
    {
        fprintf(stderr, "n_iter must be positive\n");
        return 1;
    }

    try
    {
        int device = 0;
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDevice(&device));
        CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
        printf("gpu: %s sm_%d%d\n", prop.name, prop.major, prop.minor);

        CSR *A = load_mtx_to_csr(argv[1]);
        printf("matrix: M=%d K=%d nnz=%d N=%d iter=%d\n", A->nRow, A->nCol, A->nnz, N, n_iter);

        half *B = nullptr;
        prepare_dense_b(&B, A->nCol, N);

        float *C_cusparse = static_cast<float *>(malloc(sizeof(float) * A->nRow * N));
        float *C_our = static_cast<float *>(malloc(sizeof(float) * A->nRow * N));
        memset(C_cusparse, 0, sizeof(float) * A->nRow * N);
        memset(C_our, 0, sizeof(float) * A->nRow * N);

        double t_cusparse = 0.0;
        double t_our = 0.0;
        cusparse_spmm_csr(A, B, C_cusparse, N, n_iter, &t_cusparse);
        our_spmm_balanced(A, B, C_our, N, n_iter, &t_our);

        bool ok = compare_result(C_cusparse, C_our, A->nRow * N, 1e-3f);
        printf("cuSPARSE: %.6f ms\n", t_cusparse);
        printf("our     : %.6f ms\n", t_our);
        printf("validation: %s\n", ok ? "passed" : "failed");

        free(B);
        free(C_cusparse);
        free(C_our);
        destroy_csr(A);
        return ok ? 0 : 2;
    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
