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
#include <pthread.h>
#include <math.h>
#include <vector>
#include <iostream>

using namespace std;

struct information{
    int iters;
    vector<double> x0;
    vector<double> y0;
    vector<double> results;
};

void *myThread(void *arg){
	struct information *data = (struct information *)arg;
    int iters = data -> iters;
    vector<double> x0 = data -> x0;
    vector<double> y0 = data -> y0;
    vector<double> results = data -> results;
    int index;

    for (int j = 0; j < y0.size(); j++){
        index = j * x0.size();
        for (int i = 0; i < x0.size(); i++){
            double repeats = 0;
            double x = 0;
            double y = 0;
            double length_squared = 0;
            while (repeats < iters && length_squared < 4){
                double temp = x * x - y * y + x0[i];
                y = 2 * x * y + y0[j];
                x = temp;
                length_squared = x * x + y * y;
                ++repeats;
            }
            results[index + i] = repeats;
        }
        
    }
    data -> results = results;

    return 0;
}

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
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    unsigned long ncpus = CPU_COUNT(&cpu_set);

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

    pthread_t threads[ncpus];
	struct information data[ncpus];

    /* allocate memory for image */
    int* image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    int leng[ncpus];
    
    for (int n = 0; n < ncpus; n++){
        leng[n] = height / ncpus;
        if (n < height % ncpus)
            leng[n] = height / ncpus + 1;
    }

    /* mandelbrot set */
    int start = 0, end = 0;
    double h_cal = (upper - lower) / height;
    double w_cal = (right - left) / width;
    for (int t = 0; t < ncpus; t++){
        vector<double> y0;
        vector<double> x0;
        end = start + leng[t];
        for (int c = start; c < end; c++)
            y0.push_back(c * h_cal + lower);
        for (int b = 0; b < width; b++)
            x0.push_back(b * w_cal + left);
        start += leng[t];
        vector<double> results(leng[t] * width);
        data[t].iters = iters;
        data[t].x0 = x0;
        data[t].y0 = y0;
        data[t].results = results;
        pthread_create(&threads[t], NULL, myThread, (void*)&data[t]);
    }

    start = 0;
    for (int t = 0; t < ncpus; t++){
        pthread_join(threads[t], NULL);
        for (int i = 0; i < data[t].results.size(); i++){
            image[start + i] = data[t].results[i];
        }
        start += data[t].results.size();
    }

    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);
}
