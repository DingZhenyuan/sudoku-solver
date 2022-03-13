#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <curand_kernel.h>

#define ROWS 9
#define COLS 9
#define B_ROWS 3
#define B_COLS 3
#define NUM_ITERATION 10000
#define INIT_TEMPERATURE 0.4
#define MIN_TEMPERATURE 0.001
#define INIT_TOLERANCE 1
#define DELTA_T 0.2

char inputFilePath[50] = "H:\\Class Resources\\ECE277\\quiz_2\\quiz_2\\test1.in";
char outputFilePath[50];

// ------------------------------ host functions ------------------------------//
// assign output filepath
void assignOutputFilePath() {
    for (int len = 0; len < strlen(inputFilePath) - 2; len++)
		outputFilePath[len] = inputFilePath[len];
	strcat(outputFilePath, "out");
}

// set devices
int initDevice() {
    int deviceCount;
    cudaGetDevice(&deviceCount);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceCount);

    return deviceProp.multiProcessorCount > 10 ? 10 : deviceProp.multiProcessorCount;
}

//Display the board
void printBoard(int *board) {
    std::cout << "\n-------------------------" << std::endl;
    for (int i = 0; i < ROWS; i++) {
        std::cout << "| ";
        for (int j = 0; j < COLS; j += B_COLS) {
            std::cout << board[i + COLS * j] << " " << board[i + COLS * (j + 1)] << " " << board[i + COLS * (j + 2)] << " | ";
        }
        if ((i + 1) % B_ROWS == 0) {
            std::cout << "\n-------------------------" << std::endl;
        } else {
            std::cout << std::endl;
        }
    }
}

void writeBoard(int *board) {
	FILE *fout = fopen(outputFilePath, "w");

	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLS; j++) {
            fprintf(fout, "%1d", board[i + COLS * j]);
        }
		if (i < ROWS - 1) {
            fprintf(fout, "\n");
        }
	}
	fclose(fout);
}

// read the partial board from file 
// compute the mask. 0 -> mutable value
//                  1-> non-mutable
void initMask(int *mask) {
    FILE *fin = fopen(inputFilePath, "r");

    for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLS; j++) {
            int value;
			fscanf(fin, "%1d", &value);
            mask[i + COLS * j] = value == 0 ? 0 : 1;
		}
	}
    fclose(fin);

    std::cout << "Mask:";
    printBoard(mask);
}

// initialize the board
// 1. read the partial board
// 2. place the values in all the empty slots such that 3x3 subgrid clause is satisfied
void initBoard(int *board) {
	FILE *fin = fopen(inputFilePath, "r");

	//Read the partial board from file
	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLS; j++) {
            int value;
			fscanf(fin, "%1d", &value);
			board[i + COLS * j] = value;
		}
	}
	fclose(fin);

    std::cout << "Original:";
	printBoard(board);
}

// place values in all the empty slots
// satisfied each 3x3 subgrid
void fillBoard(int *board) {
    for (int bi = 0; bi < B_ROWS; bi++) {
		for (int bj = 0; bj < B_COLS; bj++) {
            int nums[ROWS] = {0};
			for (int i = 0; i < B_ROWS; i++) {
				for (int j = 0; j < B_COLS; j++) {
					int x = bi * B_ROWS + i;
					int y = bj * B_COLS + j;
                    int id = x + COLS * y;
					if (board[id] != 0) {
						nums[board[id] - 1] = 1;
					}
				}
			}
			int idx = -1;
            int remainNum[ROWS];
			for (int k = 0; k < ROWS; k++) {
				if (nums[k] == 0) {
                    idx++;
					remainNum[idx] = k + 1;
				}
			}
            idx = 0;
			for (int i = 0; i < B_ROWS; i++) {
				for (int j = 0; j < B_COLS; j++) {
					int x = bi * B_ROWS + i;
					int y = bj * B_COLS + j;
                    int id = x + COLS * y;
					if (board[id] == 0) {
						board[id] = remainNum[idx];
                        idx++;
					}
				}
			}
		}
	}

    std::cout << "Full Filled";
    printBoard(board);
}


// host function
// returns the number of unique elements in a row or column
// flag: 1-row, 2-col
int getUniqueCountRow(int *board, int i, int flag) {
    int nums[ROWS] = { 1,2,3,4,5,6,7,8,9 };
	int count = 0;
	for (int j = 0; j < COLS; j++) {
        int idx = flag == 1 ? board[i + COLS * j] - 1 : board[j + COLS * i] - 1;
		if (idx == -1) {
			return -1;
		}
		if (nums[idx] != 0) {
			count++;
			nums[idx] = 0;
		}
	}
	return count;
}

// host function
// return energy by adding all the unique numbers of all rows and columns
int getEnergy(int *board) {
    int energy = 0;
	for (int i = 0; i < ROWS; i++) {
        energy += getUniqueCountRow(board, i, 1) + getUniqueCountRow(board, i, 2);
	}
	return 162 - energy;
}

// host function
// swap two numbers of board
void swap(int *board, int x1, int x2) {
    int temp = board[x1];
    board[x1] = board[x2];
    board[x2] = temp;
}

// host function
// get a mutable number
void getMutableNum(int *mask, int bx, int by, int &x, int &y) {
    do {
		x = rand() % B_ROWS;
		y = rand() % B_COLS;
	} while (mask[(bx + x) + COLS * (by + y)] == 1);
}

// ------------------------------ device functions ------------------------------//

__constant__ int dMask[81];

// kernel for initializing random number generators
__global__ void initRandom(curandState *state) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(1337, idx, 0, &state[idx]);
}


// device function
// return the number of unique elements in a row or column
__device__ int dGetUniqueCountRow(int board[][COLS], int i, int flag) {
    int nums[ROWS] = { 1,2,3,4,5,6,7,8,9 };
	int count = 0;
	for (int j = 0; j < COLS; j++) {
        int idx = flag == 1 ? board[i][j] - 1 : board[j][i] - 1;
		if (idx == -1) {
            return -1;
        }
		if (nums[idx] != 0) {
			count++;
			nums[idx] = 0;
		}
	}
	return count;
}

// device function
// return energy by adding all the unique numbers of all rows and columns
__device__ int dGetEnergy(int board[][COLS]) {
	int energy = 0;
	for (int i = 0; i < ROWS; i++) {
        energy += dGetUniqueCountRow(board, i, 1) + dGetUniqueCountRow(board, i, 2);
    }
	return 162 - energy;
}

// device function
// swap two numbers of board
__device__ void dSwap(int board[][COLS], int x1, int y1, int x2, int y2) {
    int temp = board[x1][y1];
    board[x1][y1] = board[x2][y2];
    board[x2][y2] = temp;
}

// device function
// get mutable number of board randomly
__device__ void dGetMutableNum(curandState *state, int bIdx, int bx, int by, int &x, int &y) {
    do {
		x = (int)B_ROWS*curand_uniform(&state[bIdx]);
		y = (int)B_COLS*curand_uniform(&state[bIdx]);
	} while (dMask[(bx + x) + COLS * (by + y)] == 1);
}

// device function
// exchange two number randomly
// calculate the benefit and possiblity to determine whether take this operation
__global__ void randomExchange(int* board, curandState *state, int currEnergy, float temperature, int *b1, 
int *b2, int *b3, int *b4, int *b5, int *b6, int *b7, int *b8, int *b9, int *b10, int *dEnergyBlock) {
    // shared memory
    __shared__ int sBoard[ROWS][COLS];

    int tIdx = threadIdx.x * blockDim.x + threadIdx.y;
	int bIdx = blockIdx.x * blockDim.x + blockIdx.y;
    sBoard[threadIdx.x][threadIdx.y] = board[threadIdx.x + COLS * threadIdx.y];

    if (tIdx != 0) {
		return;
	}

    // int temp;
	int energy;
	for (int iter = 0; iter < NUM_ITERATION; iter++) {
		//Select a Random sub block in the board
		int bx = (ROWS / B_ROWS) * (int)(B_ROWS*curand_uniform(&state[bIdx]));
		int by = (COLS / B_COLS) * (int)(B_COLS*curand_uniform(&state[bIdx]));

		//Select two unmasked points
        int x1, y1;
        dGetMutableNum(state, bIdx, bx, by, x1, y1);
        int x2, y2;
        dGetMutableNum(state, bIdx, bx, by, x2, y2);

        dSwap(sBoard, bx + x1, by + y1, bx + x2, by + y2);

		//Compute the energy of this new state
		energy = dGetEnergy(sBoard);
		if (energy < currEnergy) {
            currEnergy = energy;
        } else {
			//Accept the state
			if (exp((float)(currEnergy - energy) / temperature) > curand_uniform(&state[bIdx])) {
                currEnergy = energy;
            } else {
                dSwap(sBoard, bx + x1, by + y1, bx + x2, by + y2);
			}
		}
		//If reached the lowest point break
		if (energy == 0) {
            break;
        }
	}

	//Write the result back to memory
	for (int i = 0; i < ROWS; i++) {
		for (int j = 0; j < COLS; j++) {
            switch (bIdx) {
                case 0:
                    b1[i + COLS * j] = sBoard[i][j];
                    break;
                case 1:
                    b2[i + COLS * j] = sBoard[i][j];
                    break;
                case 2:
                    b3[i + COLS * j] = sBoard[i][j];
                    break;
                case 3:
                    b4[i + COLS * j] = sBoard[i][j];
                    break;
                case 4:
                    b5[i + COLS * j] = sBoard[i][j];
                    break;
                case 5:
                    b6[i + COLS * j] = sBoard[i][j];
                    break;
                case 6:
                    b7[i + COLS * j] = sBoard[i][j];
                    break;
                case 7:
                    b8[i + COLS * j] = sBoard[i][j];
                    break;
                case 8:
                    b9[i + COLS * j] = sBoard[i][j];
                    break;
                case 9:
                    b10[i + COLS * j] = sBoard[i][j];
                    break;
            }
		}
	}
	// write the energy back to memory for the current state
	dEnergyBlock[bIdx] = currEnergy;
    
}

// ------------------------------ main functions ------------------------------//
int main(int arg, char* argv[]) {
    // assign output filepath
    assignOutputFilePath();

    // set device
    int deviceCount = initDevice();

	float temperature = INIT_TEMPERATURE;
	float temp_min = MIN_TEMPERATURE;

	int size = sizeof(int) * 81;

	//host allocate memory
    int *board;
	int *mask;
	int *energyBlock;
	cudaHostAlloc((void**)&board, size, cudaHostAllocDefault);
	cudaHostAlloc((void**)&mask, size, cudaHostAllocDefault);
	cudaHostAlloc((void**)&energyBlock, sizeof(int)*deviceCount, cudaHostAllocDefault);

    // initialize the board and mask
    initMask(mask);
	initBoard(board);
    fillBoard(board);

	//Initial Energy of board
    int currEnergy = getEnergy(board);

    std::cout << "current energy: " << currEnergy << std::endl;

	// device allocate memory
	int *dBoard;
	cudaMalloc((void**)&dBoard, size);
	cudaMalloc((void**)&dMask, size);

    int *dBlock1, *dBlock2, *dBlock3, *dBlock4, *dBlock5, *dBlock6, *dBlock7, *dBlock8, *dBlock9, *dBlock10;
	cudaMalloc((void**)&dBlock1, size);
	cudaMalloc((void**)&dBlock2, size);
	cudaMalloc((void**)&dBlock3, size);
	cudaMalloc((void**)&dBlock4, size);
	cudaMalloc((void**)&dBlock5, size);
	cudaMalloc((void**)&dBlock6, size);
	cudaMalloc((void**)&dBlock7, size);
	cudaMalloc((void**)&dBlock8, size);
	cudaMalloc((void**)&dBlock9, size);
	cudaMalloc((void**)&dBlock10, size);

    int *dEnergyBlock;
	cudaMalloc((void**)&dEnergyBlock, sizeof(int)*deviceCount);

	//copy board and mask to device
	cudaMemcpy(dBoard, board, size, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(dMask, mask, size);

	//define grid and block
	dim3 dimGrid(1, deviceCount);
	dim3 dimBlock(COLS, ROWS);

    std::cout << "solution:";

	//define and launch initRandom kernel
	curandState *dState;
	cudaMalloc(&dState, dimBlock.x* dimBlock.y * dimGrid.x * dimGrid.y);
	initRandom << <dimGrid.x * dimGrid.y, dimBlock.x* dimBlock.y >> > (dState);
	cudaPeekAtLastError();
    // synchronize
	cudaDeviceSynchronize();

	int tolerance = INIT_TOLERANCE;
	int min, minIdx;
	int e;

	int preEnergy = currEnergy;

	//simulated Annealing algorithm
	do {
		randomExchange << < dimGrid, dimBlock >> > (dBoard, dState, currEnergy, temperature, dBlock1, dBlock2, 
        dBlock3, dBlock4, dBlock5, dBlock6, dBlock7, dBlock8, dBlock9, dBlock10, dEnergyBlock);
		cudaDeviceSynchronize();

		cudaMemcpy(energyBlock, dEnergyBlock, sizeof(int)*deviceCount, cudaMemcpyDeviceToHost);

        int min = 100;
        int minIdx = 10;
		for (int i = 0; i < deviceCount; i++) {
			if (energyBlock[i] < min) {
				min = energyBlock[i];
				minIdx = i;
			}
		}

        switch (minIdx) {
            case 0:
                cudaMemcpy(dBoard, dBlock1, size, cudaMemcpyDeviceToDevice);
                currEnergy = min;
                break;
            case 1:
                cudaMemcpy(dBoard, dBlock2, size, cudaMemcpyDeviceToDevice);
                currEnergy = min;
                break;
            case 2:
                cudaMemcpy(dBoard, dBlock3, size, cudaMemcpyDeviceToDevice);
                currEnergy = min;
                break;
            case 3:
                cudaMemcpy(dBoard, dBlock4, size, cudaMemcpyDeviceToDevice);
                currEnergy = min;
                break;
            case 4:
                cudaMemcpy(dBoard, dBlock5, size, cudaMemcpyDeviceToDevice);
                currEnergy = min;
                break;
            case 5:
                cudaMemcpy(dBoard, dBlock6, size, cudaMemcpyDeviceToDevice);
                currEnergy = min;
                break;
            case 6:
                cudaMemcpy(dBoard, dBlock7, size, cudaMemcpyDeviceToDevice);
                currEnergy = min;
                break;
            case 7:
                cudaMemcpy(dBoard, dBlock8, size, cudaMemcpyDeviceToDevice);
                currEnergy = min;
                break;
            case 8:
                cudaMemcpy(dBoard, dBlock9, size, cudaMemcpyDeviceToDevice);
                currEnergy = min;
                break;
            case 9:
                cudaMemcpy(dBoard, dBlock10, size, cudaMemcpyDeviceToDevice);
                currEnergy = min;
                break;
        }
		if (currEnergy == 0) {
			break;
		}

		if (currEnergy == preEnergy) {
            tolerance--;
        } else {
            tolerance = INIT_TOLERANCE;
        }

		// random restart if energy is stuck
		if (tolerance < 0) {
            printf("randomizing\n");

			cudaMemcpy(board, dBoard, size, cudaMemcpyDeviceToHost);

			int bx = rand() % B_ROWS * B_ROWS;
			int by = rand() % B_COLS * B_COLS;

			for (int suf = 0; suf < rand() % 10; suf++) {
                int x1, y1;
                getMutableNum(mask, bx, by, x1, y1);
                int x2, y2;
                getMutableNum(mask, bx, by, x2, y2);

                swap(board, (bx + x1) + COLS * (by + y1), (bx + x2) + COLS * (by + y2));
			}
			cudaMemcpy(dBoard, board, size, cudaMemcpyHostToDevice);
            currEnergy = getEnergy(board);
            printf("Energy after randomizing %d \n",currEnergy);
			tolerance = INIT_TOLERANCE;
			temperature += DELTA_T;
		}

		preEnergy = currEnergy;
		if (currEnergy == 0) {
			break;
		}
		temperature *= 0.8;

		printf("Energy after temp %f is %d \n", temperature, currEnergy);

	} while (temperature > temp_min);


	cudaMemcpy(board, dBoard, size, cudaMemcpyDeviceToHost);

	printBoard(board);

	writeBoard(board);

	// currEnergy = h_compute_energy(board);
    currEnergy = getEnergy(board);

	printf("Current energy %d \n", currEnergy);

	return 0;
}
