prog: wordle.c
	gcc wordle.c -lm -o wordle
	
cuda: wordle_cuda.cu
	nvcc wordle_cuda.cu -o wordle_cuda

clean:
	rm -f wordle wordle_cuda