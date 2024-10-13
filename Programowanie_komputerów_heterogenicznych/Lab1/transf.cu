
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "timer.h"

struct WIERZCHOLEK {
    float x,y;
};

extern struct time_data time_data_arr[3];

__global__ void obracanie(WIERZCHOLEK *W, float alfa, int N);
float randomFloat();
void init_random_float(WIERZCHOLEK *main_tab, int N);
void compare_cpu_with_gpu(struct WIERZCHOLEK *cpu, struct WIERZCHOLEK *gpu);

int main(void) {

	// struct WIERZCHOLEK test_naruszenia_pamieci [100000000];
    float alfa = 3.1415;
    int ile_cudow;
    struct WIERZCHOLEK *d_W;

    int rozmiary[] = {1000, 10000, 100000, 1000000, 10000000};

    for (int N : rozmiary) {
        printf("size: %d\n", N);
		
        struct WIERZCHOLEK *main_tab = (WIERZCHOLEK*) malloc(sizeof(WIERZCHOLEK) * N);
        struct WIERZCHOLEK *cpu_tab = (WIERZCHOLEK*) malloc(sizeof(WIERZCHOLEK) * N);

        srand(time(0));
        init_random_float(main_tab, N);

        time_data_arr[CPU_TIME].start = start_timer();

        for (int i = 0; i < N; i++) {
            float x = main_tab[i].x * cos(alfa) - main_tab[i].y * sin(alfa);
            float y = main_tab[i].x * sin(alfa) + main_tab[i].y * cos(alfa);
            main_tab[i].x = x;
            main_tab[i].y = y;
        }

        stop_timer(time_data_arr,CPU_TIME, "Czas dla CPU");

        memcpy(cpu_tab, main_tab, sizeof(WIERZCHOLEK) * N);

        cudaGetDeviceCount(&ile_cudow);
        if (ile_cudow == 0) {
            perror("Brak CUDY");
            return 1;
        }

        time_data_arr[KERNEL_TRANSFER_TIME].start = start_timer();

        cudaMalloc(&d_W, sizeof(WIERZCHOLEK) * N);
        cudaMemcpy(d_W, main_tab, sizeof(WIERZCHOLEK) * N, cudaMemcpyHostToDevice);

        int watki_na_blok = 1024;
        int bloki_na_siatke = (N + watki_na_blok - 1) / watki_na_blok;

        time_data_arr[KERNEL_TIME].start = start_timer();

        obracanie<<<bloki_na_siatke, watki_na_blok>>>(d_W, alfa, N);

        cudaDeviceSynchronize();

        stop_timer(time_data_arr,KERNEL_TIME,"Czas dla Kernela");

        cudaMemcpy(main_tab, d_W, sizeof(WIERZCHOLEK) * N, cudaMemcpyDeviceToHost);

        stop_timer(time_data_arr,KERNEL_TRANSFER_TIME,"Czas dla Kernela z transferem");

        // compare_cpu_with_gpu(cpu_tab, main_tab);
        double percente_result = count_percentage(time_data_arr[KERNEL_TRANSFER_TIME].stop, time_data_arr[KERNEL_TIME].stop);

        printf("Procentowy koszt transferu do i z wektora  %d elementow: %.2f%%\n", N, percente_result);


        cudaFree(d_W);
        free(main_tab);
        free(cpu_tab);
    }

    return 0;
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

void init_random_float(WIERZCHOLEK *main_tab, int N) {
    for(int i = 0; i < N; i++) {
        main_tab[i].x = randomFloat();
        main_tab[i].y = randomFloat();
    }
}

float randomFloat() {
    return (float)(rand()) / (float)(RAND_MAX);
}

__global__ void obracanie(WIERZCHOLEK *W, float alfa, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        float x = W[i].x * cos(alfa) - W[i].y * sin(alfa);
        float y = W[i].x * sin(alfa) + W[i].y * cos(alfa);
        W[i].x = x;
        W[i].y = y;
    }
}