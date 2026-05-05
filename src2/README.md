# src2: CSR-only Hybrid SpMM Core

This folder keeps the custom SpMM path from `src/our_spmm.cu` and adds an
HR-SpMM-style hybrid split:

- rows or row segments with full 64-nnz chunks are computed by the Tensor Core path;
- rows with fewer than 64 nnz, plus the residue of long rows, are computed by a CUDA Core path;
- long-row residue results are accumulated with the Tensor Core output.
- preprocessing builds explicit `LongTask(row, nnz_start)` and
  `ResidueTask(row, nnz_start, nnz_len)` lists, so kernels dispatch fixed work
  items directly instead of inferring row partitions inside the kernel.

It removes the original dense GEMM validation path, cuBLAS, cuSPARSELt, MKL,
and argument parsing. A small MatrixMarket loader and cuSPARSE benchmark are
kept for validation.

## Files

- `core_spmm.cuh`: minimal CUDA helpers, the host-side `CSR` struct, and the public API.
- `our_spmm.cu`: task-list preprocessing, Tensor Core kernel, CUDA Core residue kernel, and `our_spmm_balanced`.
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

By default the Makefile builds a fat binary for `sm_80`, `sm_86`, and `sm_89`,
plus `compute_89` PTX. Override `GENCODE` only if you want a smaller binary, for
example:

```bash
make GENCODE="-gencode arch=compute_89,code=sm_89"
```

If Tensor Core kernels fail with `an illegal instruction was encountered`, run:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
cuobjdump --list-elf ./bench_mtx
```

Then rebuild with a `GENCODE` entry matching the printed compute capability.

## MatrixMarket Benchmark

```bash
./bench_mtx matrix.mtx [N=256] [n_iter=10]
```

`N` must be a multiple of 64 for the current custom kernel. MatrixMarket
coordinate matrices with `real`, `integer`, or `pattern` fields are supported.
Symmetric matrices are expanded to full CSR.
