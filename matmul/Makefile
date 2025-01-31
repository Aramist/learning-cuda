matmul: matmul.cu rand_lib.o npy_lib.h
	nvcc -o matmul matmul.cu rand_lib.o -I. -lm -lc
rand_lib.o: rand_lib.c rand_lib.h
	nvcc -c -o rand_lib.o rand_lib.c -I. -lc
test: matmul
	./matmul
	python verify.py
	rm L.npy R.npy result.npy
clean:
	rm matmul test_npy
