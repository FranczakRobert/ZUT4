
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CPU_TIME
#define KERNEL_TIME
#define KERNEL_TRANSFER_TIME


struct WIERZCHOLEK {
    float x,y;
};

__global__ void obracanie(WIERZCHOLEK *W, float alfa, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        float x = W[i].x * cos(alfa) - W[i].y * sin(alfa);
        float y = W[i].x * sin(alfa) + W[i].y * cos(alfa);
        W[i].x = x;
        W[i].y = y;
    }
}

float randomFloat() {
    return (float)(rand()) / (float)(RAND_MAX);
}

void init_random_float(WIERZCHOLEK *Figura, int N) {
    for(int i = 0; i < N; i++) {
        Figura[i].x = randomFloat();
        Figura[i].y = randomFloat();
    }
}

void compare_cpu_with_gpu(struct WIERZCHOLEK *cpu, struct WIERZCHOLEK *gpu) {
	for(int i =0; i < 10; i++) {
		if(cpu[i].x != gpu[i].x) {
			printf(" ROZNICA w X: [GPU] -> %f --- [CPU] -> %f \n",cpu[i].x, gpu[i].x);
		}
		if(cpu[i].y != gpu[i].y) {
			printf(" ROZNICA w Y: [GPU] -> %f --- [CPU] -> %f \n",cpu[i].y, gpu[i].y);
		}

	}
}

struct timeval start_timer() {
    struct timeval begin;
    gettimeofday(&begin, 0);
    return begin;
}

double stop_timer(struct timeval begin, char* str) {
    struct timeval end;
    gettimeofday(&end, 0);
    long seconds = end.tv_sec - begin.tv_sec;
    long microseconds = end.tv_usec - begin.tv_usec;
    printf("\t[%s]\n", str);
    printf("\tTime in s [%lf] \n\n", seconds + microseconds*1e-6);
	return seconds + microseconds*1e-6;
}

int main(void) {

	// struct WIERZCHOLEK test_naruszenia_pamieci [100000000];
    float alfa = 3.1415;
    int ile_cudow;
    struct WIERZCHOLEK *d_W;

    int rozmiary[] = {1000, 10000, 100000, 1000000, 10000000};

    for (int N : rozmiary) {
        printf("size: %d\n", N);
		
        struct WIERZCHOLEK *Figura = (WIERZCHOLEK*) malloc(sizeof(WIERZCHOLEK) * N);
        struct WIERZCHOLEK *cpu_tab = (WIERZCHOLEK*) malloc(sizeof(WIERZCHOLEK) * N);

        srand(time(0));
        init_random_float(Figura, N);

        // CPU Calculation
        #ifdef CPU_TIME
        struct timeval start_cpu_time = start_timer();
        #endif

        for (int i = 0; i < N; i++) {
            float x = Figura[i].x * cos(alfa) - Figura[i].y * sin(alfa);
            float y = Figura[i].x * sin(alfa) + Figura[i].y * cos(alfa);
            Figura[i].x = x;
            Figura[i].y = y;
        }

        #ifdef CPU_TIME
        double stop_timer_CPU = stop_timer(start_cpu_time, "Czas dla CPU");
        #endif

        memcpy(cpu_tab, Figura, sizeof(WIERZCHOLEK) * N);

        cudaGetDeviceCount(&ile_cudow);
        if (ile_cudow == 0) {
            perror("Brak CUDY");
            return 1;
        }

        // GPU Calculation
        #ifdef KERNEL_TRANSFER_TIME
        struct timeval start_kernel_transfer_time = start_timer();
        #endif

        cudaMalloc(&d_W, sizeof(WIERZCHOLEK) * N);
        cudaMemcpy(d_W, Figura, sizeof(WIERZCHOLEK) * N, cudaMemcpyHostToDevice);

        int watki_na_blok = 1024;
        int bloki_na_siatke = (N + watki_na_blok - 1) / watki_na_blok;

        #ifdef KERNEL_TIME
        struct timeval start_kernel_time = start_timer();
        #endif

        obracanie<<<bloki_na_siatke, watki_na_blok>>>(d_W, alfa, N);

        #ifdef KERNEL_TIME
        cudaDeviceSynchronize();
        double stop_timer_KERNEL = stop_timer(start_kernel_time, "Czas dla Kernela");
        #endif

        cudaMemcpy(Figura, d_W, sizeof(WIERZCHOLEK) * N, cudaMemcpyDeviceToHost);

        #ifdef KERNEL_TRANSFER_TIME
        double stop_timer_KERNEL_TRANS = stop_timer(start_kernel_transfer_time, "Czas dla Kernela z transferem");
        #endif

        // compare_cpu_with_gpu(cpu_tab, Figura);

        double transfer_time = stop_timer_KERNEL_TRANS - stop_timer_KERNEL;
        double transfer_cost_percentage = (transfer_time / stop_timer_KERNEL_TRANS) * 100;

        printf("Procentowy koszt transferu do i z wektora  %d elementow: %.2f%%\n", N, transfer_cost_percentage);


        cudaFree(d_W);
        free(Figura);
        free(cpu_tab);
    }

    return 0;
}