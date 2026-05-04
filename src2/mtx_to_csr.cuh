#ifndef _MTX_TO_CSR_CUH
#define _MTX_TO_CSR_CUH

#include "core_spmm.cuh"

// Loads a MatrixMarket coordinate matrix into host-side CSR.
// Supports real, integer, and pattern matrices. MatrixMarket indices are
// treated as one-based. Symmetric matrices are expanded to full CSR.
CSR *load_mtx_to_csr(const char *filename);

#endif
