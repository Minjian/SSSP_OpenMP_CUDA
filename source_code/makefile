all: serial_and_openmp_sssp.c serial_and_cuda_sssp.cu
	gcc -o serial_and_openmp_sssp -fopenmp serial_and_openmp_sssp.c
	nvcc -o serial_and_cuda_sssp -lnvgraph serial_and_cuda_sssp.cu

clean: serial_and_openmp_sssp serial_and_cuda_sssp
	rm -fr serial_and_openmp_sssp serial_and_cuda_sssp
