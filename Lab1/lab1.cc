#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}

	// // Initialize MPI
	MPI_Init(&argc, &argv);
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
	// // Calculating the time this code cost
	// double starttime, endtime;
	// starttime = MPI_Wtime();

	// Assign the variables that will be used
	unsigned long r = atoll(argv[1]);
	unsigned long k = atoll(argv[2]);
	unsigned long pixels = 0;
	unsigned long message = 0;

	// Calculate the pixels
	for (unsigned long x = rank; x < r; x+=size) {
		unsigned long y = ceil(sqrtl((r + x) * (r - x)));
		message += y;
	}

	// Use MPI_Send and Recv to gather the answers calculated from different thread
	// If rank isn't 0, send the answer to rank 0
	if(rank != 0){
		MPI_Send(&message, 1, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD);
	}
	// If rank is 0, recvive the answers from all other threads
	else {
		pixels += message;
		for(int x = 1; x<size; x++){
			MPI_Recv(&message, 1, MPI_UNSIGNED_LONG, x, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			message %= k;
			pixels += message;
		}
		printf("%llu\n", (4 * pixels) % k);
		// endtime = MPI_Wtime();
		// printf("That took %f seconds\n", endtime-starttime);
	}
	MPI_Finalize();
}
