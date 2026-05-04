#ifndef _CUSPARSE_BASELINE_CUH
#define _CUSPARSE_BASELINE_CUH

#include "core_spmm.cuh"

// Computes C = A_csr * B with cuSPARSE SpMM.
// Uses the same host-side CSR, B, C layout as our_spmm_balanced.
void cusparse_spmm_csr(CSR *A_csr, half *B, float *C, int N, int n_iter, double *elapsed = nullptr);

#endif
