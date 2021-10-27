#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

unsigned long ggg(int x, int omp_threads, unsigned long r, int rank, int size){
	unsigned long result = 0;
	unsigned long i = x + omp_threads * rank;
	unsigned long cross = omp_threads * size;
	for (i; i < r; i+=cross){
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
	MPI_Init(&argc, &argv);
	int rank, size, omp_threads, omp_thread;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

	unsigned long r = atoll(argv[1]);
	unsigned long k = atoll(argv[2]);
	unsigned long pixels = 0;
	unsigned long message = 0;
	unsigned long result;

#pragma omp parallel private(omp_thread)
	{
		omp_thread = omp_get_thread_num();
		omp_threads = omp_get_num_threads();
		result = ggg(omp_thread, omp_threads, r, rank, size);
		message += result;
	}
	if(rank != 0){
		MPI_Send(&message, 1, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD);
	}
	else {
		message %= k;
		pixels += message;
		for(int x = 1; x<size; x++){
			MPI_Recv(&message, 1, MPI_UNSIGNED_LONG, x, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			message %= k;
			pixels += message;
		}
		printf("%llu\n", (4 * pixels) % k);

	}
	MPI_Finalize();
}

// mpicxx lab2_hybrid.cc -o lab2_hybrid -fopenmp
// srun -c3 -N2 -n6 ./lab2_hybrid