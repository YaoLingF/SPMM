NVCC ?= nvcc
GENCODE ?= -gencode arch=compute_80,code=sm_80 \
           -gencode arch=compute_86,code=sm_86 \
           -gencode arch=compute_89,code=sm_89 \
           -gencode arch=compute_89,code=compute_89
NVFLAGS ?= -O3 $(GENCODE)

TARGET = libour_spmm.a
BENCH = bench_mtx
CORE_OBJS = our_spmm.o
BENCH_OBJS = bench_mtx.o mtx_to_csr.o cusparse_baseline.o
CUSPARSE_LIBS = -lcusparse

all: $(TARGET) $(BENCH)

$(TARGET): $(CORE_OBJS)
	ar rcs $@ $^

$(BENCH): $(BENCH_OBJS) $(TARGET)
	$(NVCC) $(NVFLAGS) $^ -o $@ $(CUSPARSE_LIBS)

our_spmm.o: our_spmm.cu core_spmm.cuh
	$(NVCC) $(NVFLAGS) -c $< -o $@

mtx_to_csr.o: mtx_to_csr.cu mtx_to_csr.cuh core_spmm.cuh
	$(NVCC) $(NVFLAGS) -c $< -o $@

cusparse_baseline.o: cusparse_baseline.cu cusparse_baseline.cuh core_spmm.cuh
	$(NVCC) $(NVFLAGS) -c $< -o $@

bench_mtx.o: bench_mtx.cu core_spmm.cuh mtx_to_csr.cuh cusparse_baseline.cuh
	$(NVCC) $(NVFLAGS) -c $< -o $@

clean:
	rm -f $(CORE_OBJS) $(BENCH_OBJS) $(TARGET) $(BENCH)
