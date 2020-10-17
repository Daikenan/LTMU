all: roi_align_kernel.cu roi_align_kernel.h
	nvcc -c -o roi_align.cu.o roi_align_kernel.cu -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_52

clean:
	rm roi_align.cu.o
