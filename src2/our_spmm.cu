#include "core_spmm.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

struct LongTask
{
    int row;
    int nnz_start;
};

struct ResidueTask
{
    int row;
    int nnz_start;
    int nnz_len;
};

static __global__ void kernel_cuda_residue_spmm(ResidueTask *tasks, int num_tasks, half *spmat_val, const int *row_offset,
                                                int *col_idx, half *B, float *C, const int M, const int N, const int K)
{
    int task_idx = blockIdx.x;
    int col_start = blockIdx.y * 64;
    if (task_idx >= num_tasks || col_start >= N)
        return;

    ResidueTask task = tasks[task_idx];
    int target_row = task.row;
    int row_start = row_offset[target_row];
    int row_end = row_offset[target_row + 1];
    int row_nnz = row_end - row_start;
    if (row_nnz == 0 || task.nnz_len == 0)
        return;

    int col_end = min(col_start + 64, N);

    for (int col = col_start + threadIdx.x; col < col_end; col += blockDim.x)
    {
        float sum = 0.0f;
        for (int idx = task.nnz_start; idx < task.nnz_start + task.nnz_len; idx++)
        {
            int b_offset = col_idx[idx] * N + col;
            sum += __half2float(spmat_val[idx]) * __half2float(B[b_offset]);
        }

        float *out = &C[target_row * N + col];
        if (row_nnz >= 64)
        {
            atomicAdd(out, sum);
        }
        else
        {
            *out = sum;
        }
    }
}

static __global__ void
    __launch_bounds__(256)
        kernel_tensor_long_tasks(LongTask *tasks, int num_tasks, half *spmat_val, int *col_idx, half *B, float *C,
                                 const int M, const int N, const int K)
{
    extern __shared__ uint32_t shmem[];

    int task_idx = blockIdx.x;
    int col_start = blockIdx.y * 64;
    if (task_idx >= num_tasks || col_start >= N)
        return;

    LongTask task = tasks[task_idx];
    int target_row = task.row;
    int segment_start = task.nnz_start;

    int laneid;
    int warpid = threadIdx.x / warpSize;
    int num_warps = blockDim.x / warpSize;
    asm("mov.s32 %0, %laneid;" : "=r"(laneid));

    half *A_val_scratch = (half *)shmem;
    float *output_buffer = (float *)&A_val_scratch[num_warps * 8];
    half *B_scratch = (half *)&output_buffer[num_warps * 64];
    for (int i = 4 * threadIdx.x; i < 64 * num_warps; i += 4 * blockDim.x)
    {
        reinterpret_cast<float4 *>(&output_buffer[i])[0] = float4{0, 0, 0, 0};
    }
    for (int i = 8 * threadIdx.x; i < num_warps * 8; i += 8 * blockDim.x)
    {
        reinterpret_cast<uint4 *>(&A_val_scratch[i])[0] = uint4{0, 0, 0, 0};
    }
    for (int i = 8 * threadIdx.x; i < 64 * 8 * num_warps; i += 8 * blockDim.x)
    {
        reinterpret_cast<uint4 *>(&B_scratch[i])[0] = uint4{0, 0, 0, 0};
    }

    uint32_t reg_a[2];
    uint32_t reg_b[1];
    float reg_c[16];

    half2 *a = reinterpret_cast<half2 *>(reg_a);
    half2 *b = reinterpret_cast<half2 *>(reg_b);

    int B_scratch_leading_dim = 64;
    int fetching_col_lane = 8 * (laneid % 8);
    int register_choice = (laneid / 4) % 2;
    int first_transpose_idx = fetching_col_lane * B_scratch_leading_dim + (8 * warpid + 2 * (laneid / 8));

    int2 local_fetch;
    local_fetch.x = warpid * 8 + 2 * (laneid / 8);
    local_fetch.y = local_fetch.x + 1;

    half2 Aval_tmp = half2{0, 0};
    uint4 fetch_buffer1 = uint4{0, 0, 0, 0};
    uint4 fetch_buffer2 = uint4{0, 0, 0, 0};

    int global_fetch_x = segment_start + local_fetch.x;
    int global_fetch_y = segment_start + local_fetch.y;
    if (col_start + fetching_col_lane < N)
    {
        int b_offset_x = col_idx[global_fetch_x] * N + col_start + fetching_col_lane;
        int b_offset_y = col_idx[global_fetch_y] * N + col_start + fetching_col_lane;
        Aval_tmp.x = spmat_val[global_fetch_x];
        Aval_tmp.y = spmat_val[global_fetch_y];
        if (b_offset_x < K * N)
            fetch_buffer1 = reinterpret_cast<uint4 *>(&B[b_offset_x])[0];
        if (b_offset_y < K * N)
            fetch_buffer2 = reinterpret_cast<uint4 *>(&B[b_offset_y])[0];
    }

    __syncthreads();
    int transpose_idx = first_transpose_idx;
    for (int st = 0; st < 8; st++)
    {
        half2 tmp;
        tmp.x = reinterpret_cast<half *>(&fetch_buffer1)[st];
        tmp.y = reinterpret_cast<half *>(&fetch_buffer2)[st];
        reinterpret_cast<half2 *>(&B_scratch[transpose_idx])[0] = tmp;
        transpose_idx += B_scratch_leading_dim;
    }
    reinterpret_cast<half2 *>(&A_val_scratch[local_fetch.x])[0] = Aval_tmp;

    a[0] = a[1] = b[0] = half2{0, 0};
    __syncthreads();
    b[0] = reinterpret_cast<half2 *>(&A_val_scratch[2 * laneid])[0];

    for (int idx = 0; idx < 4; idx++)
    {
        reinterpret_cast<float4 *>(&reg_c[4 * idx])[0] = float4{0, 0, 0, 0};
    }
    for (int idx = 0; idx < 4; idx++)
    {
        int offset = B_scratch_leading_dim * (16 * idx + warpid * (16 / num_warps)) + 2 * laneid;
        a[0] = reinterpret_cast<half2 *>(&B_scratch[offset])[0];
        a[1] = reinterpret_cast<half2 *>(&B_scratch[offset + B_scratch_leading_dim])[0];
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
                     "{ %0, %1, %2, %3},"
                     "{ %4, %5 },"
                     "{ %6 },"
                     "{ %0, %1, %2, %3};\n"
                     : "+f"(reg_c[4 * idx]), "+f"(reg_c[4 * idx + 1]), "+f"(reg_c[4 * idx + 2]), "+f"(reg_c[4 * idx + 3])
                     : "r"(reg_a[0]), "r"(reg_a[1]),
                       "r"(reg_b[0]));
    }

    float2 val;
    for (int idx = 0; idx < 4; idx++)
    {
        val.x = reg_c[4 * idx + register_choice];
        val.y = reg_c[4 * idx + register_choice + 2];

        val.x += __shfl_down_sync(0xffff0000, val.x, 18);
        val.y += __shfl_down_sync(0xffff0000, val.y, 18);
        val.x += __shfl_down_sync(0xff000000, val.x, 9);
        val.y += __shfl_down_sync(0xff000000, val.y, 9);
        val.x += __shfl_down_sync(0xf0000000, val.x, 4);
        val.y += __shfl_down_sync(0xf0000000, val.y, 4);

        int out_col = col_start + 16 * idx + 2 * warpid;
        if (laneid == 0 && out_col < N)
        {
            reinterpret_cast<float2 *>(&output_buffer[16 * idx + 2 * warpid])[0] = val;
        }
        __syncthreads();
    }

    float *output_row = &C[target_row * N + col_start];
    for (int i = 4 * threadIdx.x; i < 64; i += 4 * blockDim.x)
    {
        if (col_start + i < N)
        {
            atomicAdd(&output_row[i], output_buffer[i]);
        }
        if (col_start + i + 1 < N)
        {
            atomicAdd(&output_row[i + 1], output_buffer[i + 1]);
        }
        if (col_start + i + 2 < N)
        {
            atomicAdd(&output_row[i + 2], output_buffer[i + 2]);
        }
        if (col_start + i + 3 < N)
        {
            atomicAdd(&output_row[i + 3], output_buffer[i + 3]);
        }
    }
}

static void build_hybrid_tasks(CSR *A_csr, std::vector<LongTask> *long_tasks, std::vector<ResidueTask> *residue_tasks)
{
    for (int row = 0; row < A_csr->nRow; row++)
    {
        int row_start = A_csr->row_offset[row];
        int row_end = A_csr->row_offset[row + 1];
        int row_nnz = row_end - row_start;
        int full_segments = row_nnz / 64;
        int residue = row_nnz % 64;

        for (int segment = 0; segment < full_segments; segment++)
        {
            long_tasks->push_back(LongTask{row, row_start + segment * 64});
        }
        if (residue > 0)
        {
            residue_tasks->push_back(ResidueTask{row, row_start + full_segments * 64, residue});
        }
    }
}

void our_spmm_balanced(CSR *A_csr, half *B, float *C, int N, int n_iter, double *elapsed)
{
    const int M = A_csr->nRow;
    const int K = A_csr->nCol;

    warm_up_gpu_kernel<<<1, 32>>>();
    cudaDeviceSynchronize();
    float measured_time = 0.0;
    float alloc_time, h2d_time, d2h_time, conversion_time;
    alloc_time = h2d_time = d2h_time = conversion_time = 0.0;

    int *dcol_idx, *drow_offset;
    half *dA_val;
    half *dB;
    float *dC;
    size_t total_mem = 0;

    cudaEvent_t start, end;
    VERBOSE_PUTS("Mtrix Conversion...");
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&end));
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventElapsedTime(&conversion_time, start, end));
    VERBOSE_PRINT("Done %fms taken\n", conversion_time);

    VERBOSE_PUTS("Preparing Resources...");
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDA(cudaMalloc((void **)&dA_val, sizeof(A_csr->value[0]) * A_csr->nnz));
    CHECK_CUDA(cudaMalloc((void **)&dcol_idx, sizeof(A_csr->col_idx[0]) * A_csr->nnz));
    CHECK_CUDA(cudaMalloc((void **)&drow_offset, sizeof(A_csr->row_offset[0]) * (A_csr->nRow + 1)));

    CHECK_CUDA(cudaMalloc((void **)&dB, sizeof(B[0]) * K * N));
    total_mem += sizeof(B[0]) * K * N;
    CHECK_CUDA(cudaMalloc((void **)&dC, sizeof(C[0]) * M * N));
    total_mem += sizeof(C[0]) * M * N;
    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventElapsedTime(&alloc_time, start, end));
    VERBOSE_PRINT("Done %fms taken\n", alloc_time);

    VERBOSE_PUTS("H2D Memcpy...");
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDA(cudaMemcpy(dA_val, A_csr->value, sizeof(A_csr->value[0]) * A_csr->nnz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dcol_idx, A_csr->col_idx, sizeof(A_csr->col_idx[0]) * A_csr->nnz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(drow_offset, A_csr->row_offset, sizeof(A_csr->row_offset[0]) * (A_csr->nRow + 1), cudaMemcpyHostToDevice));

    // CHECK_CUDA(cudaMemcpy(dA, A, sizeof(A[0]) * M * K, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, B, sizeof(B[0]) * K * N, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dC, C, sizeof(C[0]) * M * N, cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventElapsedTime(&h2d_time, start, end));
    VERBOSE_PRINT("Done %fms taken\n", h2d_time);

    VERBOSE_PUTS("Performs our SpMM...");

    const int tile_size = 64;

    std::vector<LongTask> long_tasks;
    std::vector<ResidueTask> residue_tasks;
    int num_warp_per_tb = 8;
    int blockdim = WARP_SIZE * num_warp_per_tb;
    size_t tensor_shm_size = num_warp_per_tb * 8 * sizeof(half) +
                             num_warp_per_tb * 64 * sizeof(float) +
                             num_warp_per_tb * 64 * 8 * sizeof(half);
    LongTask *d_long_tasks = NULL;
    ResidueTask *d_residue_tasks = NULL;

    auto preprocess_start = std::chrono::high_resolution_clock::now();
    build_hybrid_tasks(A_csr, &long_tasks, &residue_tasks);
    auto preprocess_end = std::chrono::high_resolution_clock::now();

    float elapsed_preprocess = std::chrono::duration<float, std::milli>(preprocess_end - preprocess_start).count();
    VERBOSE_PRINT("preprocess %fms taken\n", elapsed_preprocess);
    VERBOSE_PRINT("long_tasks: %zu residue_tasks: %zu blockdim: %d tensor_shm: %fKB\n", long_tasks.size(), residue_tasks.size(), blockdim, (float)tensor_shm_size / 1024.0f);

    if (!long_tasks.empty())
    {
        CHECK_CUDA(cudaMalloc((void **)&d_long_tasks, sizeof(LongTask) * long_tasks.size()));
        CHECK_CUDA(cudaMemcpy(d_long_tasks, long_tasks.data(), sizeof(LongTask) * long_tasks.size(), cudaMemcpyHostToDevice));
    }
    if (!residue_tasks.empty())
    {
        CHECK_CUDA(cudaMalloc((void **)&d_residue_tasks, sizeof(ResidueTask) * residue_tasks.size()));
        CHECK_CUDA(cudaMemcpy(d_residue_tasks, residue_tasks.data(), sizeof(ResidueTask) * residue_tasks.size(), cudaMemcpyHostToDevice));
    }

    const char *disable_tensor_env = getenv("OUR_SPMM_DISABLE_TENSOR");
    const char *disable_residue_env = getenv("OUR_SPMM_DISABLE_RESIDUE");
    bool run_tensor = (disable_tensor_env == NULL || disable_tensor_env[0] == '0');
    bool run_residue = (disable_residue_env == NULL || disable_residue_env[0] == '0');

    dim3 tensor_grid((unsigned int)long_tasks.size(), (N + tile_size - 1) / tile_size);
    dim3 cuda_residue_grid((unsigned int)residue_tasks.size(), (N + tile_size - 1) / tile_size);
    const int cuda_residue_block = 128;
    for (int i = 0; i < 5; i++)
    {
        CHECK_CUDA(cudaMemset(dC, 0, sizeof(float) * M * N));
        if (run_tensor && !long_tasks.empty())
        {
            kernel_tensor_long_tasks<<<tensor_grid, blockdim, tensor_shm_size>>>(d_long_tasks, (int)long_tasks.size(), dA_val, dcol_idx, dB, dC, M, N, K);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
        }
        if (run_residue && !residue_tasks.empty())
        {
            kernel_cuda_residue_spmm<<<cuda_residue_grid, cuda_residue_block>>>(d_residue_tasks, (int)residue_tasks.size(), dA_val, drow_offset, dcol_idx, dB, dC, M, N, K);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
        }
    }
    measured_time = 0.0f;
    for (int i = 0; i < n_iter; i++)
    {
        float iter_time = 0.0f;
        CHECK_CUDA(cudaMemset(dC, 0, sizeof(float) * M * N));
        CHECK_CUDA(cudaEventRecord(start));
        if (run_tensor && !long_tasks.empty())
        {
            kernel_tensor_long_tasks<<<tensor_grid, blockdim, tensor_shm_size>>>(d_long_tasks, (int)long_tasks.size(), dA_val, dcol_idx, dB, dC, M, N, K);
            CHECK_CUDA(cudaGetLastError());
        }
        if (run_residue && !residue_tasks.empty())
        {
            kernel_cuda_residue_spmm<<<cuda_residue_grid, cuda_residue_block>>>(d_residue_tasks, (int)residue_tasks.size(), dA_val, drow_offset, dcol_idx, dB, dC, M, N, K);
            CHECK_CUDA(cudaGetLastError());
        }
        CHECK_CUDA(cudaEventRecord(end));
        CHECK_CUDA(cudaEventSynchronize(end));
        CHECK_CUDA(cudaEventElapsedTime(&iter_time, start, end));
        measured_time += iter_time;
    }
    VERBOSE_PRINT("Done %fms taken\n", measured_time);

    VERBOSE_PUTS("D2H  Memcpy...");
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDA(cudaMemcpy(C, dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventElapsedTime(&d2h_time, start, end));
    VERBOSE_PRINT("Done %fms taken\n", d2h_time);

    measured_time = measured_time / (double)n_iter;
    if (elapsed != NULL)
    {
        *elapsed = (double)measured_time;
    }
    CHECK_CUDA(cudaFree(dcol_idx));
    CHECK_CUDA(cudaFree(drow_offset));
    CHECK_CUDA(cudaFree(dA_val));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    if (d_long_tasks)
        CHECK_CUDA(cudaFree(d_long_tasks));
    if (d_residue_tasks)
        CHECK_CUDA(cudaFree(d_residue_tasks));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(end));
}

__global__ void warm_up_gpu_kernel()
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia = 0.0f;
    float ib = 0.0f;
    ib += ia + tid;
}

void destroy_csr(CSR *mat)
{
    if (mat == nullptr)
        return;
    free(mat->row_offset);
    free(mat->col_idx);
    free(mat->value);
    free(mat);
}
