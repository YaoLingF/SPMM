# 1. 头文件路径
INCLUDE = -I/root/miniconda3/include \
          -I/root/miniconda3/lib/python3.12/site-packages/cusparselt/include

# 2. 库路径
LDFLAGS = -L/root/miniconda3/lib/python3.12/site-packages/cusparselt/lib \
          -L/root/miniconda3/lib

# 3. 编译选项
CXXFLAGS = -Xcompiler=-fopenmp
NVFLAG = -arch=sm_86

# 4. 链接库（⭐重点：加上 MKL）
LIBS = -lcublas -lcusparse -lcusparseLt -lcudart \
       -lmkl_rt -lpthread -lm -ldl

# 5. 最终链接
test: test.o miscalleneous.o gemm_cublas.o spmm_cusparse.o our_spmm.o file_input.o
	nvcc -g -o test $(INCLUDE) $(LDFLAGS) $(CXXFLAGS) $(NVFLAG) \
	test.o miscalleneous.o spmm_cusparse.o our_spmm.o file_input.o gemm_cublas.o \
	$(LIBS)

# 6. 编译规则（⚠️这里不要放 LIBS）
%.o : %.cu
	nvcc $(INCLUDE) $(CXXFLAGS) $(NVFLAG) $(DEFS) -o $@ -c $<

clean:
	rm -f *.o test