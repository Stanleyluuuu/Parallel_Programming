#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

unsigned long ggg(int x, int omp_threads, unsigned long r){
	unsigned long result = 0;
	for (unsigned long i = x; i < r; i+=omp_threads){
		unsigned long y = ceil(sqrtl((r + i) * (r - i)));
		result += y;
	}

	return result;
}

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	int omp_threads, omp_thread, a;
	unsigned long r = atoll(argv[1]);
	unsigned long k = atoll(argv[2]);
	unsigned long pixels = 0;
	unsigned long result;

	#pragma omp parallel
	{
		omp_threads = omp_get_num_threads();
		omp_thread = omp_get_thread_num();
		result = ggg(omp_thread, omp_threads, r);
		result %= k;
		pixels += result;
	}
	printf("%llu\n", (4 * pixels) % k);
}
// g++ lab2_omp.cc -o lab2_omp -fopenmp -lm
// srun -c4 -n1 ./lab2_omp