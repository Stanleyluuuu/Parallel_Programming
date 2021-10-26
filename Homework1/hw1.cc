#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <string>
#include <iostream>
#include <algorithm>
#include <utility>
#include <cstring>

using namespace std;

float* merge_two_array(float *arr1, float *arr2, int n1, int n2) {
	int i = 0, j = 0, k = 0, leng = n1 + n2;
	static float* arr3 = new float[leng];
    while (i<n1 && j <n2)
	{
        if (arr1[i] < arr2[j])
            arr3[k++] = arr1[i++];
        else
            arr3[k++] = arr2[j++];
    }
	while (i < n1)
        arr3[k++] = arr1[i++];
	while (j < n2)
        arr3[k++] = arr2[j++];

	return arr3;
}

int main(int argc, char** argv) {
	// Assign the variables that will be used
	int rank, size, leng, mes_leng, leng_left, leng_right, remain, offset, n;
	int numbers = atoll(argv[1]);
	// double starttime, endtime;
	bool even { true };
	int s = 0;

	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// starttime = MPI_Wtime();

	// Calculate how many numbers will each thread gets
	leng = numbers / size;
	mes_leng = numbers / size;
	// Calculate how many remaining numbers are there
	remain = numbers - (leng * size);
	// If there is remaining numbers, distribute them to the threads first n threads then calculate the offset
	if (remain > 0){
		mes_leng += 1;
		// For first n threads, get one more number
		if (rank < remain){
			leng += 1;
			offset = sizeof(float) * rank * leng;
		}
		// For ohter threads, remain then calculate the offset
		else{
			offset = sizeof(float) * ((remain) * (leng + 1) + (rank - remain) * (leng));
		}
	}
	// If there isn't remaining numbers, just calculate the offset
	else{
		offset = sizeof(float) * rank * leng;
	}

	leng_right = leng;
	leng_left = leng;
	if (rank == (remain - 1))
		leng_right = leng - 1;
	
	if (rank == remain)
		leng_left = leng + 1;

	if (numbers > size)
		n = size;
	else
		n = numbers;
	// create an array with corresponding length to store the data
	float* data = new float[leng];
	float* message = new float[mes_leng];
	float* cache = new float[leng];
	float* cache1 = new float[mes_leng];
	float* arr;

	MPI_File f;
	MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &f);
	// Read the data from file f, and get the number correspond to each offset
	MPI_File_read_at(f, offset, data, leng, MPI_FLOAT, MPI_STATUS_IGNORE);
	MPI_File_close(&f);
	if (size > 1){
		sort(data, data+leng);
		while (s <= n){
			// Start sorting
			// Incase the number of threads is larger than number of datas
			if (rank < n){
				// Even mode
				if (even == true){
					if (rank % 2 == 1){
						MPI_Send(data, leng, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD);
						MPI_Recv(message, leng, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						swap(data, message);
					}
					else if (rank % 2 == 0 && rank != (n - 1)){
						MPI_Recv(message, leng_right, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						arr = merge_two_array(data, message, leng, leng_right);
						memcpy(cache, arr, leng * sizeof(float)); 
						memcpy(cache1, &arr[leng], leng_right * sizeof(float));
						swap(data, cache);
						swap(message, cache1);
						MPI_Send(message, leng_right, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
					}
					s += 1;
					even = { false };
				}
				// Odd mode
				else{
					if (rank % 2 == 1 && rank != (n - 1)){
						MPI_Send(data, leng, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
						MPI_Recv(message, leng, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						swap(data, message);
					}
					else if (rank % 2 == 0 && rank != 0){
						MPI_Recv(message, leng_left, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						arr = merge_two_array (data, message, leng, leng_left);
						memcpy(cache1, arr, leng_left * sizeof(float)); 
						memcpy(cache, &arr[leng_left], leng * sizeof(float));
						swap(data, cache);
						swap(message, cache1);
						MPI_Send(message, leng_left, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD);
					}
					s += 1;
					even = { true };
				}
			}
			else{
				s = size;
			}
		}
	}
	else{
		sort(data, data+leng);
	}
	MPI_File w;
	MPI_File_open(MPI_COMM_WORLD, argv[3], MPI_MODE_RDWR | MPI_MODE_CREATE, MPI_INFO_NULL, &w);
	if (rank < n)
		MPI_File_write_at(w, offset, data, leng, MPI_FLOAT, MPI_STATUS_IGNORE);
	MPI_File_close(&w);

	// if (rank == 0){
	// 	endtime = MPI_Wtime();
	// 	printf("That took %f seconds\n", endtime-starttime);
	// }
	
	MPI_Finalize();
}