#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <mpi.h>
#include <omp.h>

using namespace std;

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        int gg = (height - 1 - y);
        for (int x = 0; x < width; ++x) {
            int p = buffer[gg * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    MPI_Init(&argc, &argv);
    int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);
    double h_cal = (upper - lower) / height;
    double w_cal = (right - left) / width;

    /* allocate memory for image */
    int* image = (int*)malloc(width * height * sizeof(int));
    int* results = (int*)malloc(width * height * sizeof(int));
    assert(image);
    assert(results);

    int start, end, leng, remain, ans;
    double x0, y0;
    start = rank * (height / size);
    end = (rank + 1) * (height / size);
    remain = height % size;
    ans = height / size;
    if (remain > 0){
        if (rank < (remain)){
            start += rank;
            end += (rank + 1);
        }
        else{
            start += remain;
            end += remain;
        }
    }
    
    if (rank < remain)
        leng = ans + 1;
    else
        leng = ans;
    
#pragma omp parallel private(x0, y0)
{
    int omp_thread, omp_threads;
    omp_thread = omp_get_thread_num();
    omp_threads = omp_get_num_threads();
    for (int j = start; j < end; j++){
        y0 = j * h_cal + lower;
        int index = j * width;
        #pragma omp for schedule(dynamic, omp_threads)
        for (int i = 0; i < width; i++){
            x0 = i * w_cal + left;
            double repeats = 0;
            double x = 0;
            double y = 0;
            double length_squared = 0;
            while (repeats < iters && length_squared < 4){
                double temp = x * x - y * y + x0;
                y = 2 * x * y + y0;
                x = temp;
                length_squared = x * x + y * y;
                ++repeats;
            }
            image[index + i] = repeats;
        }
        
    }
}
    if (rank != 0){
        MPI_Send(image, width * height, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    else if (rank == 0){
        for(int x = 1; x<size; x++){
            int k = 0, startt = 0, endd = 0;
            startt = ans * x * width;
            endd = ans * (x + 1) * width;
            if (x < height % size){
                startt = (ans + 1) * x * width;
                endd = (ans + 1) * (x + 1) * width;
            }
            else{
                startt = (ans + 1) * remain * width + ans * (x - remain) * width;
                endd = (ans + 1) * remain * width + ans * (x - remain + 1) * width;
            }
			MPI_Recv(&results[0], width * height, MPI_INT, x, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = startt; i < endd; i++){                
                image[i] = results[i];
            }
        }
    }
    /* draw and cleanup */
    MPI_Finalize();
    if (rank == 0)
        write_png(filename, iters, width, height, image);
    free(image);
    free(results);
}
// mpicxx -pthread hw2b.cc -o hw2b -lpng -lm -fopenmp