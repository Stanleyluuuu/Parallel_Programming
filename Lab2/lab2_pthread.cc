#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>
#include <string>
#include <unistd.h>
#include <iostream>

using namespace std;

struct information{
	unsigned long r;
	unsigned long k;
	unsigned long result;
	int x;
	unsigned long ncpus;
};

void *myThread(void *arg){
	struct information *data = (struct information *)arg;
	unsigned long x = data -> x;
	unsigned long r = data -> r;
	unsigned long k = data -> k;
	unsigned long result = data -> result;
	unsigned long ncpus = data -> ncpus;
	for (x; x < r; x += ncpus){
		unsigned long y = ceil(sqrtl((r + x) * (r - x)));
		// unsigned long y = ceil(sqrt2((r + x) * (r - x)));
		result += y;
	}
	
   	data-> result = result;
	return 0;
}

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	cpu_set_t cpuset;
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
	unsigned long ncpus = CPU_COUNT(&cpuset);
	unsigned long r = atoll(argv[1]);
	unsigned long k = atoll(argv[2]);
	unsigned long pixels = 0;

	pthread_t threads[ncpus];
	struct information data[ncpus];

	for (int t = 0; t < ncpus; t++){
		data[t].r = r;
		data[t].x = t;
		data[t].result = 0;
		data[t].ncpus = ncpus;
		pthread_create(&threads[t], NULL, myThread, (void*)&data[t]);
	}
	for (int t = 0; t < ncpus; t++){
		pthread_join(threads[t], NULL);
		pixels += data[t].result;
		pixels %= k;
	}
	printf("%llu\n", pixels * 4 % k);
}

// g++ lab2_pthread.cc -o lab2_pthread -pthread -lm
// srun -c4 -n1 ./lab2_pthread r k