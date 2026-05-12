#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <climits>

namespace {

constexpr int kTileN = 64;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;
constexpr int kTensorBlockThreads = kWarpSize * kWarpsPerBlock;
constexpr int kResidueBlockThreads = 128;

__global__ void __launch_bounds__(kTensorBlockThreads)
kernel_tensor_src2_long_rows(const int *__restrict__ rowptr,
                             const int *__restrict__ col_idx,
                             const half *__restrict__ spmat_val,
                             const half *__restrict__ x,
                             float *__restrict__ out,
                             int rows,
                             int feat_dim) {
  extern __shared__ uint32_t shmem[];

  int target_row = blockIdx.x;
  int col_start = blockIdx.y * kTileN;
  if (target_row >= rows || col_start >= feat_dim) {
    return;
  }

  int row_start = rowptr[target_row];
  int row_end = rowptr[target_row + 1];
  int row_nnz = row_end - row_start;
  int full_nnz = (row_nnz / kTileN) * kTileN;
  if (full_nnz == 0) {
    return;
  }

  int laneid;
  int warpid = threadIdx.x / warpSize;
  int num_warps = blockDim.x / warpSize;
  asm("mov.s32 %0, %laneid;" : "=r"(laneid));

  half *A_val_scratch = reinterpret_cast<half *>(shmem);
  float *output_buffer =
      reinterpret_cast<float *>(&A_val_scratch[num_warps * 8]);
  half *B_scratch = reinterpret_cast<half *>(&output_buffer[kTileN]);

  for (int i = 4 * threadIdx.x; i < kTileN; i += 4 * blockDim.x) {
    reinterpret_cast<float4 *>(&output_buffer[i])[0] = float4{0, 0, 0, 0};
  }
  __syncthreads();

  uint32_t reg_a[2];
  uint32_t reg_b[1];
  float reg_c[16];
  half2 *a = reinterpret_cast<half2 *>(reg_a);
  half2 *b = reinterpret_cast<half2 *>(reg_b);

  int B_scratch_leading_dim = kTileN;
  int fetching_col_lane = 8 * (laneid % 8);
  int register_choice = (laneid / 4) % 2;
  int first_transpose_idx =
      fetching_col_lane * B_scratch_leading_dim +
      (8 * warpid + 2 * (laneid / 8));

  int2 local_fetch;
  local_fetch.x = warpid * 8 + 2 * (laneid / 8);
  local_fetch.y = local_fetch.x + 1;

  for (int seg_off = 0; seg_off < full_nnz; seg_off += kTileN) {
    int segment_start = row_start + seg_off;
    int global_fetch_x = segment_start + local_fetch.x;
    int global_fetch_y = segment_start + local_fetch.y;
    int b_col = col_start + fetching_col_lane;

    half2 Aval_tmp;
    Aval_tmp.x = spmat_val[global_fetch_x];
    Aval_tmp.y = spmat_val[global_fetch_y];

    uint4 fetch_buffer1 = uint4{0, 0, 0, 0};
    uint4 fetch_buffer2 = uint4{0, 0, 0, 0};
    if ((feat_dim % 8 == 0) && b_col + 7 < feat_dim) {
      int b_offset_x = col_idx[global_fetch_x] * feat_dim + b_col;
      int b_offset_y = col_idx[global_fetch_y] * feat_dim + b_col;
      fetch_buffer1 = reinterpret_cast<const uint4 *>(&x[b_offset_x])[0];
      fetch_buffer2 = reinterpret_cast<const uint4 *>(&x[b_offset_y])[0];
    } else {
      half *fb1 = reinterpret_cast<half *>(&fetch_buffer1);
      half *fb2 = reinterpret_cast<half *>(&fetch_buffer2);
      int b_row_x = col_idx[global_fetch_x];
      int b_row_y = col_idx[global_fetch_y];
#pragma unroll
      for (int t = 0; t < 8; t++) {
        int col = b_col + t;
        if (col < feat_dim) {
          fb1[t] = x[b_row_x * feat_dim + col];
          fb2[t] = x[b_row_y * feat_dim + col];
        }
      }
    }

    int transpose_idx = first_transpose_idx;
#pragma unroll
    for (int st = 0; st < 8; st++) {
      half2 tmp;
      tmp.x = reinterpret_cast<half *>(&fetch_buffer1)[st];
      tmp.y = reinterpret_cast<half *>(&fetch_buffer2)[st];
      reinterpret_cast<half2 *>(&B_scratch[transpose_idx])[0] = tmp;
      transpose_idx += B_scratch_leading_dim;
    }
    reinterpret_cast<half2 *>(&A_val_scratch[local_fetch.x])[0] = Aval_tmp;

    reg_a[0] = 0;
    reg_a[1] = 0;
    reg_b[0] = 0;
    __syncthreads();

    b[0] = reinterpret_cast<half2 *>(&A_val_scratch[2 * laneid])[0];
#pragma unroll
    for (int idx = 0; idx < 4; idx++) {
      reinterpret_cast<float4 *>(&reg_c[4 * idx])[0] = float4{0, 0, 0, 0};
    }
#pragma unroll
    for (int idx = 0; idx < 4; idx++) {
      int offset =
          B_scratch_leading_dim *
              (16 * idx + warpid * (16 / num_warps)) +
          2 * laneid;
      a[0] = reinterpret_cast<half2 *>(&B_scratch[offset])[0];
      a[1] = reinterpret_cast<half2 *>(&B_scratch[offset + B_scratch_leading_dim])[0];
      asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
                   "{ %0, %1, %2, %3},"
                   "{ %4, %5 },"
                   "{ %6 },"
                   "{ %0, %1, %2, %3};\n"
                   : "+f"(reg_c[4 * idx]), "+f"(reg_c[4 * idx + 1]),
                     "+f"(reg_c[4 * idx + 2]), "+f"(reg_c[4 * idx + 3])
                   : "r"(reg_a[0]), "r"(reg_a[1]), "r"(reg_b[0]));
    }

    float2 val;
#pragma unroll
    for (int idx = 0; idx < 4; idx++) {
      val.x = reg_c[4 * idx + register_choice];
      val.y = reg_c[4 * idx + register_choice + 2];

      const unsigned full_mask = 0xffffffffu;
      val.x += __shfl_down_sync(full_mask, val.x, 18);
      val.y += __shfl_down_sync(full_mask, val.y, 18);
      val.x += __shfl_down_sync(full_mask, val.x, 9);
      val.y += __shfl_down_sync(full_mask, val.y, 9);
      val.x += __shfl_down_sync(full_mask, val.x, 4);
      val.y += __shfl_down_sync(full_mask, val.y, 4);

      int out_col = col_start + 16 * idx + 2 * warpid;
      int out_idx = 16 * idx + 2 * warpid;
      if (laneid == 0 && out_col < feat_dim) {
        output_buffer[out_idx] += val.x;
        if (out_col + 1 < feat_dim) {
          output_buffer[out_idx + 1] += val.y;
        }
      }
    }
    __syncthreads();
  }

  float *output_row = &out[target_row * feat_dim + col_start];
  for (int i = 4 * threadIdx.x; i < kTileN; i += 4 * blockDim.x) {
    if ((feat_dim % 4 == 0) && col_start + i + 3 < feat_dim) {
      reinterpret_cast<float4 *>(&output_row[i])[0] =
          reinterpret_cast<float4 *>(&output_buffer[i])[0];
    } else {
#pragma unroll
      for (int t = 0; t < 4; t++) {
        if (col_start + i + t < feat_dim) {
          output_row[i + t] = output_buffer[i + t];
        }
      }
    }
  }
}

__global__ void kernel_cuda_src2_residue_rows(const int *__restrict__ rowptr,
                                              const int *__restrict__ col_idx,
                                              const half *__restrict__ spmat_val,
                                              const half *__restrict__ x,
                                              float *__restrict__ out,
                                              int rows,
                                              int feat_dim) {
  int target_row = blockIdx.x;
  int col_start = blockIdx.y * kTileN;
  if (target_row >= rows || col_start >= feat_dim) {
    return;
  }

  int row_start = rowptr[target_row];
  int row_end = rowptr[target_row + 1];
  int row_nnz = row_end - row_start;
  if (row_nnz == 0) {
    return;
  }

  int full_nnz = (row_nnz / kTileN) * kTileN;
  int residue_start = row_start;
  int residue_len = row_nnz;
  bool accumulate = false;
  if (full_nnz > 0) {
    residue_start = row_start + full_nnz;
    residue_len = row_nnz - full_nnz;
    accumulate = true;
  }
  if (residue_len == 0) {
    return;
  }

  int col_end = min(col_start + kTileN, feat_dim);
  for (int feat = col_start + threadIdx.x; feat < col_end; feat += blockDim.x) {
    float sum = 0.0f;
    for (int e = residue_start; e < residue_start + residue_len; e++) {
      int src = col_idx[e];
      sum += __half2float(spmat_val[e]) * __half2float(x[src * feat_dim + feat]);
    }

    float *dst = &out[target_row * feat_dim + feat];
    if (accumulate) {
      *dst += sum;
    } else {
      *dst = sum;
    }
  }
}

}  // namespace

torch::Tensor our_spmm_forward_cuda(torch::Tensor rowptr,
                                    torch::Tensor col,
                                    torch::Tensor val,
                                    torch::Tensor x) {
  c10::cuda::CUDAGuard device_guard(x.device());

  TORCH_CHECK(rowptr.dim() == 1, "rowptr must be 1D");
  TORCH_CHECK(col.dim() == 1, "col must be 1D");
  TORCH_CHECK(val.dim() == 1, "val must be 1D");
  TORCH_CHECK(x.dim() == 2, "x must be 2D [K, F]");
  TORCH_CHECK(rowptr.scalar_type() == torch::kInt32,
              "src2 kernel expects int32 rowptr");
  TORCH_CHECK(col.scalar_type() == torch::kInt32,
              "src2 kernel expects int32 col");
  TORCH_CHECK(val.scalar_type() == torch::kFloat16,
              "src2 kernel expects FP16 CSR values");
  TORCH_CHECK(x.scalar_type() == torch::kFloat16,
              "src2 kernel expects FP16 dense features");
  TORCH_CHECK(rowptr.is_contiguous(), "rowptr must be contiguous");
  TORCH_CHECK(col.is_contiguous(), "col must be contiguous");
  TORCH_CHECK(val.is_contiguous(), "val must be contiguous");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

  int64_t rows64 = rowptr.numel() - 1;
  int64_t cols64 = x.size(0);
  int64_t feat64 = x.size(1);
  TORCH_CHECK(rows64 >= 0, "rowptr must have at least one element");
  TORCH_CHECK(rows64 <= INT_MAX, "too many rows for src2 kernel");
  TORCH_CHECK(cols64 <= INT_MAX, "too many columns for src2 kernel");
  TORCH_CHECK(feat64 <= INT_MAX, "too many features for src2 kernel");
  TORCH_CHECK(val.numel() == col.numel(), "val and col length mismatch");

  int rows = static_cast<int>(rows64);
  int feat_dim = static_cast<int>(feat64);
  auto out = torch::empty({rows64, feat64}, x.options().dtype(torch::kFloat32));
  if (rows == 0 || feat_dim == 0) {
    return out;
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  cudaMemsetAsync(out.data_ptr<float>(), 0, out.numel() * sizeof(float), stream);

  const int *rowptr_ptr = rowptr.data_ptr<int>();
  const int *col_ptr = col.data_ptr<int>();
  const half *val_ptr =
      reinterpret_cast<const half *>(val.data_ptr<c10::Half>());
  const half *x_ptr = reinterpret_cast<const half *>(x.data_ptr<c10::Half>());
  float *out_ptr = out.data_ptr<float>();

  dim3 grid(rows, (feat_dim + kTileN - 1) / kTileN);
  dim3 tensor_block(kTensorBlockThreads);
  size_t tensor_shm_size = kWarpsPerBlock * 8 * sizeof(half) +
                           kTileN * sizeof(float) +
                           kWarpsPerBlock * kTileN * 8 * sizeof(half);
  kernel_tensor_src2_long_rows<<<grid, tensor_block, tensor_shm_size, stream>>>(
      rowptr_ptr, col_ptr, val_ptr, x_ptr, out_ptr, rows, feat_dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  dim3 residue_block(kResidueBlockThreads);
  kernel_cuda_src2_residue_rows<<<grid, residue_block, 0, stream>>>(
      rowptr_ptr, col_ptr, val_ptr, x_ptr, out_ptr, rows, feat_dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return out;
}
