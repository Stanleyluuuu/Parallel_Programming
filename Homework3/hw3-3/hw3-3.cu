#include <cstddef>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <omp.h>
#include <iostream>

#define thread_num 32
#define thread4xbig 64

const int INF = ((1 << 30) - 1);
void input(char* inFileName);
void output(char* outFileName);

void block_FW(int B);
int ceil(int a, int b);

__global__ void cal_p1(int* d_Dist, int expand, int round){
    __shared__ int Pivot_dist[thread4xbig][thread4xbig];
    const int factor = thread4xbig / thread_num;
    int buffer;
    int ty = threadIdx.y * 2, tx = threadIdx.x * 2;
    int i = thread4xbig * round + ty, j = thread4xbig * round + tx;

    for (int a = 0; a < factor; a++){
        for (int b = 0; b < factor; b++){
            Pivot_dist[ty + b][tx + a] = d_Dist[(i + b) * expand + (j + a)];
        }
    }
    __syncthreads();

    for(int k = 0; k < thread4xbig; k++){
        for (int a = 0; a < factor; a++){
            for (int b = 0; b < factor; b++){
                buffer = Pivot_dist[ty + b][k] + Pivot_dist[k][tx + a];
                Pivot_dist[ty + b][tx + a] = min(Pivot_dist[ty + b][tx + a], buffer);
            }
        }
    }

    for (int a = 0; a < factor; a++){
        for (int b = 0; b < factor; b++)
            d_Dist[(i + b) * expand + (j + a)] = Pivot_dist[ty + b][tx + a];
    }
}

__global__ void cal_p2(int* d_Dist, int expand, int round){
    if(blockIdx.x == round) return;
    __shared__ int Pivot_dist[thread4xbig][thread4xbig];
    __shared__ int Now_dist[thread4xbig][thread4xbig];
    const int factor = thread4xbig / thread_num;
    int buffer;
    int ty = threadIdx.y * 2, tx = threadIdx.x * 2;
    int i = thread4xbig * round + ty, j = thread4xbig * round + tx;
    int shortDist[factor][factor];
    
    for (int a = 0; a < factor; a++){
        for (int b = 0; b < factor; b++){
            Pivot_dist[ty + b][tx + a] = d_Dist[(i + b) * expand + (j + a)];
        }
    }

    // Row
    if(blockIdx.y == 0){
        i = thread4xbig * round + ty;
        j = thread4xbig * blockIdx.x + tx;
    }
    // Column
    else{
        i = thread4xbig * blockIdx.x + ty;
        j = thread4xbig * round + tx;
    }

    for (int a = 0; a < factor; a++){
        for (int b = 0; b < factor; b++){
            Now_dist[ty + b][tx + a] = d_Dist[(i + b) * expand + (j + a)];
        }
    }

    for (int a = 0; a < factor; a++){
        for (int b = 0; b < factor; b++){
            shortDist[b][a] = Now_dist[ty + b][tx + a];
        }
    }
    __syncthreads();
    // Row
    for(int k = 0; k < thread4xbig; k++){
        if(blockIdx.y == 0){
            for (int a = 0; a < factor; a++){
                for (int b = 0; b < factor; b++){
                    buffer = Pivot_dist[ty + b][k] + Now_dist[k][tx + a];
                    shortDist[b][a] = min(shortDist[b][a], buffer);
                }
            }
        }
    // Column
        else{
            for (int a = 0; a < factor; a++){
                for (int b = 0; b < factor; b++){
                    buffer = Now_dist[ty + b][k] + Pivot_dist[k][tx + a];
                    shortDist[b][a] = min(shortDist[b][a], buffer);
                }
            }
        }
    }

    for (int a = 0; a < factor; a++){
        for (int b = 0; b < factor; b++)
            d_Dist[(i + b) * expand + (j + a)] = shortDist[b][a];
    }
}

__global__ void cal_p3(int* d_Dist, int expand, int round, int blockOffset){
    __shared__ int pi_row[thread4xbig][thread4xbig];
    __shared__ int pi_col[thread4xbig][thread4xbig];
    // __shared__ int pi_now[thread4xbig][thread4xbig];
    const int factor = thread4xbig / thread_num;
    int pi_now[factor][factor];
    int ty = threadIdx.y * 2, tx = threadIdx.x * 2;
    int i, j, buffer;
    
    i = thread4xbig * round + ty;
    j = thread4xbig * blockIdx.x + tx;

    for (int a = 0; a < factor; a++){
        for (int b = 0; b < factor; b++){
            pi_row[ty + b][tx + a] = d_Dist[(i + b) * expand + j + a];
        }
    }
    
    i = thread4xbig * (blockOffset + blockIdx.y) + ty;
    j = thread4xbig * round + tx;

    for (int a = 0; a < factor; a++){
        for (int b = 0; b < factor; b++){
            pi_col[ty + b][tx + a] = d_Dist[(i + b) * expand + j + a];
        }
    }

    i = thread4xbig * (blockOffset +blockIdx.y) + ty;
    j = thread4xbig * blockIdx.x + tx;

    for (int a = 0; a < factor; a++){
        for (int b = 0; b < factor; b++){
            // pi_now[ty + b][tx + a] = d_Dist[(i + b) * expand + j + a];
            pi_now[b][a] = d_Dist[(i + b) * expand + j + a];
        }
    }
    __syncthreads();

    for(int k = 0; k < thread4xbig; k++){
        for (int a = 0; a < factor; a++){
            for (int b = 0; b < factor; b++){
                buffer = pi_col[ty + b][k] + pi_row[k][tx + a];
                // pi_now[ty + b][tx + a] = min(pi_now[ty + b][tx + a], buffer);
                pi_now[b][a] = min(pi_now[b][a], buffer);
            }
        }
    }

    for (int a = 0; a < factor; a++){
        for (int b = 0; b < factor; b++)
            // d_Dist[(i + b) * expand + (j + a)] = pi_now[ty + b][tx + a];
            d_Dist[(i + b) * expand + (j + a)] = pi_now[b][a];
    }
}

int n, m, expand;
static int* Dist = nullptr;
static int* Dist_ori = nullptr;

int main(int argc, char* argv[]) {
    clock_t t1, t2;
    t1 = clock();
    input(argv[1]);
    int B = thread4xbig;
    block_FW(B);
    output(argv[2]);
    t2 = clock();
    printf ("Total %d clicks (%f seconds).\n",t2 - t1,((float)t2-t1)/CLOCKS_PER_SEC);
    return 0;
}

void input(char* infile) {
    int padding = 0;
    FILE* file = fopen(infile, "rb");
    fread(&expand, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    if (expand % thread4xbig != 0)
        padding = thread4xbig - (expand % thread4xbig);
    n = expand;
    expand += padding;

    if (expand / thread4xbig % 2 != 0){
        expand += thread4xbig;
        padding += thread4xbig;
    }

    Dist = (int*)malloc(expand * expand * sizeof(int));
    Dist_ori = (int*)malloc(n * n * sizeof(int));

    for (int i = 0; i < expand; ++i) {
        for (int j = 0; j < expand; ++j) {
            if (i == j)
                Dist[i * expand + j] = 0;
            else 
                Dist[i * expand + j] = INF;
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0] * expand + pair[1]] = pair[2];
    }
    fclose(file);
}

void reshape(int* Dist, int* Dist_ori){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++)
            Dist_ori[i * n + j] = Dist[i * expand + j];
    }
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    reshape(Dist, Dist_ori);
    fwrite(Dist_ori, sizeof(int), n * n, outfile);
    fclose(outfile);
}

int ceil(int a, int b) {return (a + b - 1) / b;}

void comunicate(int* Dist_sou, int* Dist_dst, int r){
    for(int i = 0; i < thread4xbig; i++)
        cudaMemcpyPeer(Dist_dst + (((r + 1) * thread4xbig + i) * expand), 0, Dist_sou + (((r + 1) * thread4xbig + i) * expand), 1, expand * sizeof(int));
}

void block_FW(int B) { 
    int* d0_Dist;
    int* d1_Dist;

    int round = ceil(expand, B);
    size_t offset = expand * expand / 2;
    size_t half_dataSize = offset * sizeof(int);

    cudaSetDevice(0);
    cudaMalloc((void**)&d0_Dist, expand * expand * sizeof(int));
    cudaMemcpy(d0_Dist, Dist, expand * expand * sizeof(int), cudaMemcpyHostToDevice);
    cudaSetDevice(1);
    cudaMalloc((void**)&d1_Dist, expand * expand * sizeof(int));
    cudaMemcpy(d1_Dist, Dist, expand * expand * sizeof(int), cudaMemcpyHostToDevice);

    dim3 grid_p1(1, 1);
    dim3 block_p1(thread_num, thread_num);

    dim3 grid_p2(expand / thread4xbig, 2);
    dim3 block_p2(thread_num, thread_num);
    
    dim3 grid_p3(expand / thread4xbig, expand / thread4xbig / 2);
    dim3 block_p3(thread_num, thread_num);

    int blockOffset = expand / thread4xbig / 2;

    for (int r = 0; r < round; ++r) {
#pragma omp parallel
    {   
        int omp_thread;
        omp_thread = omp_get_thread_num();
        if(omp_thread == 0){
            cudaSetDevice(0);
            cal_p1<<< grid_p1, block_p1 >>>(d0_Dist, expand, r);
            cal_p2<<< grid_p2, block_p2 >>>(d0_Dist, expand, r);
        }
        if(omp_thread == 1){
            cudaSetDevice(1);
            cal_p1<<< grid_p1, block_p1 >>>(d1_Dist, expand, r);
            cal_p2<<< grid_p2, block_p2 >>>(d1_Dist, expand, r);
        }
#pragma omp barrier
        if(omp_thread == 0){
            cudaSetDevice(0);
            cal_p3<<< grid_p3, block_p3 >>>(d0_Dist, expand, r, 0);
            if(r < round / 2 - 1)
                comunicate(d0_Dist, d1_Dist, r);
        }
        if(omp_thread == 1){
            cudaSetDevice(1);
            cal_p3<<< grid_p3, block_p3 >>>(d1_Dist, expand, r, blockOffset);
            if(r >= round / 2 - 1 && r < round - 1)
                comunicate(d1_Dist, d0_Dist, r);
        }
    }
    }
    cudaSetDevice(0);
    cudaMemcpy(Dist, d0_Dist, half_dataSize, cudaMemcpyDeviceToHost);
    cudaFree(d0_Dist);
    cudaSetDevice(1);
    cudaMemcpy(Dist + offset, d1_Dist + offset, half_dataSize, cudaMemcpyDeviceToHost);
    cudaFree(d1_Dist);
}
// nvcc -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 -Xcompiler -fopenmp hw3-3.cu -o hw3-3
// srun -N1 -n1 --gres=gpu:1 ./hw3-3 cases/c01.1 c01.1.out
// srun -p prof -N1 -n1 --gres=gpu:2 nvprof --metrics gld_throughput,gst_throughput,achieved_occupancy,sm_efficiency,shared_load_throughput,shared_store_throughput,inst_integer ./hw3-3 cases/c04.1 c04.1.out