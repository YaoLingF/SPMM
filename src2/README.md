# src2: CSR-only Our SpMM Core

This folder keeps only the custom Tensor Core SpMM path from `src/our_spmm.cu`.
It removes file parsing, dense GEMM validation, cuSPARSE comparison, cuBLAS,
cuSPARSELt, MKL, and argument parsing.

## Files

- `core_spmm.cuh`: minimal CUDA helpers, the host-side `CSR` struct, and the public API.
- `our_spmm.cu`: workload manager, CUDA kernels, and `our_spmm_balanced`.
- `mtx_to_csr.cu/.cuh`: MatrixMarket `.mtx` coordinate input to host-side CSR.
- `cusparse_baseline.cu/.cuh`: cuSPARSE CSR SpMM baseline with the same host-side inputs.
- `bench_mtx.cu`: loads `.mtx`, runs cuSPARSE and `our_spmm_balanced`, compares results.
- `Makefile`: builds `libour_spmm.a` and `bench_mtx`.

## API

```cpp
void our_spmm_balanced(CSR *A_csr, half *B, float *C, int N, int n_iter, double *elapsed = nullptr);
```

The function computes:

```text
C[A_csr->nRow, N] = A_csr[A_csr->nRow, A_csr->nCol] * B[A_csr->nCol, N]
```

Requirements:

- `A_csr` is host-side CSR.
- `A_csr->row_offset`, `A_csr->col_idx`, and `A_csr->value` are host pointers.
- CSR column indices are zero-based.
- `A_csr->value` and `B` are FP16.
- `C` is FP32 and row-major.
- `N` should be a multiple of 64 for the current kernel assumptions.

## Build

```bash
cd src2
make
```

Override the GPU target if needed:

```bash
make ARCH=sm_80
```

## MatrixMarket Benchmark

```bash
./bench_mtx matrix.mtx [N=256] [n_iter=10]
```

`N` must be a multiple of 64 for the current custom kernel. MatrixMarket
coordinate matrices with `real`, `integer`, or `pattern` fields are supported.
Symmetric matrices are expanded to full CSR.
