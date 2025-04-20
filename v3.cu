%%writefile v3.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h> // CUDA header

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 3
#define BATCH_SIZE 64 // Note: BATCH_SIZE is defined but not used
#define NUM_CLASSES 10  // Digits 0-9

// --- Optimization Constants ---
#define THREADS_PER_BLOCK 256
#define NUM_STREAMS 2 // For double buffering

// --- CUDA Error Checking Macro ---
#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), _FILE, __LINE_); \
        exit(EXIT_FAILURE); \
    }

// --- Host Helper Functions (Mostly Unchanged) ---

// Timer function
double get_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

// Allocate memory for a matrix (HOST - standard malloc)
double* allocateMatrix(int rows, int cols) {
    double* mat = (double*)malloc(rows * cols * sizeof(double));
    if (mat == NULL) {
        fprintf(stderr, "Host memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    return mat;
}

// Free allocated matrix memory (HOST - standard free)
void freeMatrix(double* mat) {
    free(mat);
}

// Allocate Pinned Host Memory (for async transfers)
double* allocatePinnedHostBuffer(int size) {
    double* ptr;
    CHECK_CUDA_ERROR(cudaHostAlloc((void**)&ptr, size * sizeof(double), cudaHostAllocDefault));
    return ptr;
}

// Free Pinned Host Memory
void freePinnedHostBuffer(double* ptr) {
    if (ptr) {
        CHECK_CUDA_ERROR(cudaFreeHost(ptr));
    }
}


// Softmax (remains on CPU)
void softmax(double* x, int size) {
    double max_val = x[0];
    for(int i = 1; i < size; ++i) {
        if (x[i] > max_val) max_val = x[i];
    }
    double sum = 0;
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i] - max_val); // Subtract max for numerical stability
        sum += x[i];
    }
    if (sum == 0) sum = 1e-9; // Avoid division by zero
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// Neural network structure (Host pointers) - Unchanged
typedef struct {
    double* W1; // HIDDEN_SIZE * INPUT_SIZE
    double* W2; // OUTPUT_SIZE * HIDDEN_SIZE
    double* b1; // HIDDEN_SIZE
    double* b2; // OUTPUT_SIZE
} NeuralNetwork;

// Initialize neural network (HOST) - Unchanged
NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    net->W1 = allocateMatrix(HIDDEN_SIZE, INPUT_SIZE);
    net->W2 = allocateMatrix(OUTPUT_SIZE, HIDDEN_SIZE);
    net->b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    net->b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));
    if (!net->b1 || !net->b2) { fprintf(stderr, "Host bias allocation failed\n"); exit(EXIT_FAILURE); }
    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE; i++) { for (int j = 0; j < INPUT_SIZE; j++) { net->W1[i * INPUT_SIZE + j] = ((double)rand() / RAND_MAX - 0.5) * 0.1; } }
    for (int i = 0; i < OUTPUT_SIZE; i++) { for (int j = 0; j < HIDDEN_SIZE; j++) { net->W2[i * HIDDEN_SIZE + j] = ((double)rand() / RAND_MAX - 0.5) * 0.1; } }
    return net;
}

// Free network memory (HOST) - Unchanged
void freeNetwork(NeuralNetwork* net) { if (net) { freeMatrix(net->W1); freeMatrix(net->W2); free(net->b1); free(net->b2); free(net); } }

// --- CUDA Device Memory Management (Unchanged) ---

double* allocateMatrix_gpu(int rows, int cols) { double* mat_gpu; CHECK_CUDA_ERROR(cudaMalloc((void**)&mat_gpu, rows * cols * sizeof(double))); return mat_gpu; }
void freeMatrix_gpu(double* mat_gpu) { if (mat_gpu) { CHECK_CUDA_ERROR(cudaFree(mat_gpu)); } }
void copyToDevice(double* host_data, double* device_data, int size) { CHECK_CUDA_ERROR(cudaMemcpy(device_data, host_data, size * sizeof(double), cudaMemcpyHostToDevice)); }
void copyToHost(double* device_data, double* host_data, int size) { CHECK_CUDA_ERROR(cudaMemcpy(host_data, device_data, size * sizeof(double), cudaMemcpyDeviceToHost)); }


// --- CUDA Kernels (with Shared Memory) ---

// Kernel for Matrix-Vector Multiplication: C = A * B + Bias (Using Shared Memory for B)
// A is matrix (rows x cols), B is vector (cols x 1), C is vector (rows x 1)
// Assumes 'cols' fits reasonably within shared memory limits. Max size of B is INPUT_SIZE (784).
_global_ void matrixVectorMulBiasKernelShared(double* A, double* B, double* C, double* Bias, int rows, int cols) {
    // Allocate shared memory for vector B
    // Size needs to be 'cols'. Ensure this fits! (784 * 8 bytes = ~6KB, usually okay)
    _shared_ double s_B[INPUT_SIZE]; // Statically allocate max possible size needed

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x; // e.g., THREADS_PER_BLOCK

    // Load vector B into shared memory cooperatively
    // Each thread loads multiple elements if cols > block_size
    for (int i = tid; i < cols; i += block_size) {
        if (i < cols) { // Boundary check
           s_B[i] = B[i];
        }
    }
    __syncthreads(); // Wait for ALL threads in the block to finish loading s_B

    // Perform matrix-vector multiplication using shared memory
    if (row < rows) {
        double sum = 0.0;
        for (int k = 0; k < cols; k++) {
            // Read from shared memory - potentially much faster than global memory
            sum += A[row * cols + k] * s_B[k];
        }
        if (Bias != NULL) {
             C[row] = sum + Bias[row];
        } else {
             C[row] = sum;
        }
    }
}

// Kernel for ReLU activation: X = max(0, X) - No shared memory needed
_global_ void reluKernel(double* X, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        X[idx] = (X[idx] > 0.0) ? X[idx] : 0.0;
    }
}

// Kernel for calculating output gradient: d_output = output - target - No shared memory needed
_global_ void outputGradientKernel(double* output, double* target, double* d_output, int size) {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (idx < size) {
         d_output[idx] = output[idx] - target[idx];
     }
}


// Kernel for calculating hidden gradient part 1: d_hidden_temp = W2^T * d_output (Using Shared Memory for d_output)
// W2 is (OUTPUT_SIZE x HIDDEN_SIZE), d_output is (OUTPUT_SIZE x 1), d_hidden_temp is (HIDDEN_SIZE x 1)
_global_ void hiddenGradientPart1KernelShared(double* W2, double* d_output, double* d_hidden_temp, int hidden_size, int output_size) {
    // Shared memory for d_output vector (size OUTPUT_SIZE = 10, very small)
    _shared_ double s_d_output[OUTPUT_SIZE];

    int row = blockIdx.x * blockDim.x + threadIdx.x; // row corresponds to hidden unit index
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Load d_output into shared memory (only first 'output_size' threads needed)
    if (tid < output_size) {
        s_d_output[tid] = d_output[tid];
    }
    __syncthreads(); // Wait for s_d_output to be loaded

    // Perform transposed matrix-vector multiplication using shared memory
    if (row < hidden_size) {
        double sum = 0.0;
        for (int k = 0; k < output_size; k++) {
            // Access W2 transposed: W2[k][row] -> W2[k * hidden_size + row]
            // Read d_output from shared memory
            sum += W2[k * hidden_size + row] * s_d_output[k];
        }
        d_hidden_temp[row] = sum;
    }
}

// Kernel for calculating hidden gradient part 2 (applying ReLU derivative) - No shared memory needed
_global_ void hiddenGradientPart2Kernel(double* d_hidden_temp, double* hidden_activations, double* d_hidden, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_hidden[idx] = d_hidden_temp[idx] * (hidden_activations[idx] > 0.0 ? 1.0 : 0.0);
    }
}

// Kernel for updating weights: W -= learning_rate * d_layer * activation_prev^T - No shared memory needed
_global_ void updateWeightsKernel(double* W, double* d_layer, double* activation_prev, double learning_rate, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        W[idx] -= learning_rate * d_layer[row] * activation_prev[col];
    }
}

// Kernel for updating biases: B -= learning_rate * d_layer - No shared memory needed
_global_ void updateBiasesKernel(double* B, double* d_layer, double learning_rate, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        B[idx] -= learning_rate * d_layer[idx];
    }
}


// --- GPU Forward and Backward Pass (Using Shared Memory Kernels) ---

// Forward pass on GPU (using shared memory kernels)
void forward_gpu(double* d_W1, double* d_b1, double* d_W2, double* d_b2,
                 double* d_input, double* d_hidden, double* d_output_pre_softmax,
                 cudaStream_t stream) // Added stream argument
{
    dim3 gridDimHidden((HIDDEN_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDimOutput((OUTPUT_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    // Calculate hidden layer: hidden = relu(W1 * input + b1)
    matrixVectorMulBiasKernelShared<<<gridDimHidden, blockDim, 0, stream>>>(d_W1, d_input, d_hidden, d_b1, HIDDEN_SIZE, INPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    reluKernel<<<gridDimHidden, blockDim, 0, stream>>>(d_hidden, HIDDEN_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Calculate output layer (pre-softmax): output_pre_softmax = W2 * hidden + b2
    matrixVectorMulBiasKernelShared<<<gridDimOutput, blockDim, 0, stream>>>(d_W2, d_hidden, d_output_pre_softmax, d_b2, OUTPUT_SIZE, HIDDEN_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// Backward pass on GPU (using shared memory kernels)
void backward_gpu(double* d_W1, double* d_b1, double* d_W2, double* d_b2,
                  double* d_input, double* d_hidden, double* d_output_post_softmax, // After softmax
                  double* d_target,
                  double* d_d_output, double* d_d_hidden, double* d_d_hidden_temp, // Device gradient buffers
                  cudaStream_t stream) // Added stream argument
{
    dim3 gridDimOutput((OUTPUT_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDimHidden((HIDDEN_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    // 1. Compute output layer gradient: d_output = output - target
    outputGradientKernel<<<gridDimOutput, blockDim, 0, stream>>>(d_output_post_softmax, d_target, d_d_output, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // 2. Compute hidden layer gradient: d_hidden = (W2^T * d_output) * relu_derivative(hidden)
    // Part 1: d_hidden_temp = W2^T * d_output (Shared Memory)
    hiddenGradientPart1KernelShared<<<gridDimHidden, blockDim, 0, stream>>>(d_W2, d_d_output, d_d_hidden_temp, HIDDEN_SIZE, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    // Part 2: d_hidden = d_hidden_temp * relu_derivative(hidden)
    hiddenGradientPart2Kernel<<<gridDimHidden, blockDim, 0, stream>>>(d_d_hidden_temp, d_hidden, d_d_hidden, HIDDEN_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // 3. Update weights W2: W2 -= LR * d_output * hidden^T
    dim3 blockDimUpdateW2(16, 16);
    dim3 gridDimUpdateW2((HIDDEN_SIZE + blockDimUpdateW2.x - 1) / blockDimUpdateW2.x,
                         (OUTPUT_SIZE + blockDimUpdateW2.y - 1) / blockDimUpdateW2.y);
    updateWeightsKernel<<<gridDimUpdateW2, blockDimUpdateW2, 0, stream>>>(d_W2, d_d_output, d_hidden, LEARNING_RATE, OUTPUT_SIZE, HIDDEN_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // 4. Update weights W1: W1 -= LR * d_hidden * input^T
    dim3 blockDimUpdateW1(16, 16);
    dim3 gridDimUpdateW1((INPUT_SIZE + blockDimUpdateW1.x - 1) / blockDimUpdateW1.x,
                         (HIDDEN_SIZE + blockDimUpdateW1.y - 1) / blockDimUpdateW1.y);
    updateWeightsKernel<<<gridDimUpdateW1, blockDimUpdateW1, 0, stream>>>(d_W1, d_d_hidden, d_input, LEARNING_RATE, HIDDEN_SIZE, INPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // 5. Update biases b2: b2 -= LR * d_output
    updateBiasesKernel<<<gridDimOutput, blockDim, 0, stream>>>(d_b2, d_d_output, LEARNING_RATE, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // 6. Update biases b1: b1 -= LR * d_hidden
    updateBiasesKernel<<<gridDimHidden, blockDim, 0, stream>>>(d_b1, d_d_hidden, LEARNING_RATE, HIDDEN_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
}


// --- MNIST Loading Functions (Unchanged) ---
double* loadMNISTImages(const char* filename, int numImages) { FILE* file = fopen(filename, "rb"); if (!file) { printf("Error opening %s\n", filename); exit(1); } fseek(file, 16, SEEK_SET); double* images = allocateMatrix(numImages, INPUT_SIZE); for (int i = 0; i < numImages; i++) { for (int j = 0; j < INPUT_SIZE; j++) { unsigned char pixel; if (fread(&pixel, sizeof(unsigned char), 1, file) != 1) { fprintf(stderr, "Error: Failed to read pixel\n"); fclose(file); exit(EXIT_FAILURE); } images[i * INPUT_SIZE + j] = pixel / 255.0; } } fclose(file); return images; }
double* loadMNISTLabels(const char* filename, int numLabels) { FILE* file = fopen(filename, "rb"); if (!file) { printf("Error opening %s\n", filename); exit(1); } fseek(file, 8, SEEK_SET); double* labels = allocateMatrix(numLabels, OUTPUT_SIZE); for (int i = 0; i < numLabels; i++) { unsigned char label_val; if (fread(&label_val, sizeof(unsigned char), 1, file) != 1) { fprintf(stderr, "Error: Failed to read label\n"); fclose(file); exit(EXIT_FAILURE); } for (int j = 0; j < OUTPUT_SIZE; j++) { labels[i * OUTPUT_SIZE + j] = (j == label_val) ? 1.0 : 0.0; } } fclose(file); return labels; }


// --- Modified Training and Evaluation (with Streams and Pinned Memory) ---

// Train network using GPU with Streams and Shared Memory Kernels
void train_gpu(NeuralNetwork* net, double* images, double* labels, int numImages) {
    clock_t total_start = clock();

    // --- Create CUDA Streams ---
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CHECK_CUDA_ERROR(cudaStreamCreate(&streams[i]));
    }

    // --- Allocate GPU Memory (Network parameters - only 1 set needed) ---
    double* d_W1 = allocateMatrix_gpu(HIDDEN_SIZE, INPUT_SIZE);
    double* d_W2 = allocateMatrix_gpu(OUTPUT_SIZE, HIDDEN_SIZE);
    double* d_b1 = allocateMatrix_gpu(1, HIDDEN_SIZE);
    double* d_b2 = allocateMatrix_gpu(1, OUTPUT_SIZE);

    // --- Allocate GPU Memory (Per-sample activations/gradients - only 1 set needed) ---
    double* d_hidden = allocateMatrix_gpu(1, HIDDEN_SIZE);
    double* d_d_output = allocateMatrix_gpu(1, OUTPUT_SIZE);
    double* d_d_hidden = allocateMatrix_gpu(1, HIDDEN_SIZE);
    double* d_d_hidden_temp = allocateMatrix_gpu(1, HIDDEN_SIZE); // Temporary buffer

    // --- Allocate Double Buffers on GPU (for input/target/output) ---
    double* d_input[NUM_STREAMS];
    double* d_target[NUM_STREAMS];
    double* d_output_pre_softmax[NUM_STREAMS];
    double* d_output_post_softmax[NUM_STREAMS]; // After softmax (copied back from host)
    for (int i = 0; i < NUM_STREAMS; ++i) {
        d_input[i] = allocateMatrix_gpu(1, INPUT_SIZE);
        d_target[i] = allocateMatrix_gpu(1, OUTPUT_SIZE);
        d_output_pre_softmax[i] = allocateMatrix_gpu(1, OUTPUT_SIZE);
        d_output_post_softmax[i] = allocateMatrix_gpu(1, OUTPUT_SIZE);
    }

    // --- Allocate Pinned Host Memory (Double Buffers) ---
    double* h_input_pinned[NUM_STREAMS];
    double* h_target_pinned[NUM_STREAMS];
    double* h_output_pinned[NUM_STREAMS]; // For receiving pre-softmax output
    for (int i = 0; i < NUM_STREAMS; ++i) {
        h_input_pinned[i] = allocatePinnedHostBuffer(INPUT_SIZE);
        h_target_pinned[i] = allocatePinnedHostBuffer(OUTPUT_SIZE);
        h_output_pinned[i] = allocatePinnedHostBuffer(OUTPUT_SIZE);
    }

    // --- Copy Initial Network Parameters to GPU (Synchronously) ---
    copyToDevice(net->W1, d_W1, HIDDEN_SIZE * INPUT_SIZE);
    copyToDevice(net->W2, d_W2, OUTPUT_SIZE * HIDDEN_SIZE);
    copyToDevice(net->b1, d_b1, HIDDEN_SIZE);
    copyToDevice(net->b2, d_b2, OUTPUT_SIZE);

    printf("Starting GPU Training (Streams + Shared Memory)...\n");

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double epoch_loss = 0.0;
        int epoch_correct = 0;

        // --- Pre-load first sample's data into pinned buffer 0 ---
        memcpy(h_input_pinned[0], &images[0], INPUT_SIZE * sizeof(double));
        memcpy(h_target_pinned[0], &labels[0], OUTPUT_SIZE * sizeof(double));
        // --- Start async copy for the first sample ---
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_input[0], h_input_pinned[0], INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice, streams[0]));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_target[0], h_target_pinned[0], OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice, streams[0]));


        for (int i = 0; i < numImages; i++) {
            int current_stream_idx = i % NUM_STREAMS;
            int next_stream_idx = (i + 1) % NUM_STREAMS;
            cudaStream_t current_stream = streams[current_stream_idx];

            // --- If not the last iteration, prepare and start copy for the NEXT sample ---
            if (i + 1 < numImages) {
                 cudaStream_t next_stream = streams[next_stream_idx];
                 // Copy data for sample i+1 into the next pinned buffer
                 memcpy(h_input_pinned[next_stream_idx], &images[(i + 1) * INPUT_SIZE], INPUT_SIZE * sizeof(double));
                 memcpy(h_target_pinned[next_stream_idx], &labels[(i + 1) * OUTPUT_SIZE], OUTPUT_SIZE * sizeof(double));
                 // Start async HtoD copy for sample i+1 in the next stream
                 CHECK_CUDA_ERROR(cudaMemcpyAsync(d_input[next_stream_idx], h_input_pinned[next_stream_idx], INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice, next_stream));
                 CHECK_CUDA_ERROR(cudaMemcpyAsync(d_target[next_stream_idx], h_target_pinned[next_stream_idx], OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice, next_stream));
            }

            // --- Launch Forward Pass Kernels for the CURRENT sample in the current stream ---
            // Kernels implicitly wait for HtoD copy in the same stream to finish
            forward_gpu(d_W1, d_b1, d_W2, d_b2,
                        d_input[current_stream_idx], d_hidden, d_output_pre_softmax[current_stream_idx],
                        current_stream);

            // --- Start Asynchronous DtoH Copy of pre-softmax output for CURRENT sample ---
            CHECK_CUDA_ERROR(cudaMemcpyAsync(h_output_pinned[current_stream_idx], d_output_pre_softmax[current_stream_idx], OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost, current_stream));

            // --- Synchronize the CURRENT stream ---
            // We MUST wait here for the DtoH copy to complete before CPU softmax
            CHECK_CUDA_ERROR(cudaStreamSynchronize(current_stream));

            // --- CPU Operations (Softmax, Loss, Accuracy) ---
            // Now h_output_pinned[current_stream_idx] contains the result
            softmax(h_output_pinned[current_stream_idx], OUTPUT_SIZE); // Apply softmax on host

            // Compute Loss & Accuracy
            double sample_loss = 0.0;
            double* current_label_host = &labels[i * OUTPUT_SIZE]; // Pointer to original host label
            for (int k = 0; k < OUTPUT_SIZE; k++) {
                 sample_loss -= current_label_host[k] * log(h_output_pinned[current_stream_idx][k] + 1e-9);
            }
            epoch_loss += sample_loss;

            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (h_output_pinned[current_stream_idx][j] > h_output_pinned[current_stream_idx][pred]) pred = j;
                if (current_label_host[j] > 0.9) actual = j;
            }
            if (pred == actual) epoch_correct++;

            // --- Copy Post-Softmax Output back to GPU (Asynchronously) ---
            // Needed for backward pass gradient calculation
            CHECK_CUDA_ERROR(cudaMemcpyAsync(d_output_post_softmax[current_stream_idx], h_output_pinned[current_stream_idx], OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice, current_stream));

            // --- Launch Backward Pass Kernels for the CURRENT sample in the current stream ---
            // Kernels wait for the HtoD copy above to finish
            backward_gpu(d_W1, d_b1, d_W2, d_b2,
                         d_input[current_stream_idx], d_hidden, d_output_post_softmax[current_stream_idx], d_target[current_stream_idx],
                         d_d_output, d_d_hidden, d_d_hidden_temp,
                         current_stream);

            // Note: No explicit sync needed here if the next iteration's HtoD copy
            // in the other stream can overlap with these backward kernels.
            // However, a sync might be needed before the very next operation
            // in the same stream if dependencies exist (which they do implicitly here).
            // The structure ensures operations within a stream are ordered.

        } // End image loop

        // --- Synchronize ALL streams after the epoch ---
        // Ensures all work is done before calculating epoch time and printing stats
        for (int i = 0; i < NUM_STREAMS; ++i) {
            CHECK_CUDA_ERROR(cudaStreamSynchronize(streams[i]));
        }

        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, epoch_loss / numImages, (epoch_correct / (double)numImages) * 100, get_time(epoch_start));

    } // End epoch loop

    printf("Total training time: %.3fs\n", get_time(total_start));

    // --- Copy Final Weights Back to Host (Synchronously) ---
    printf("Copying final weights back to host...\n");
    // Ensure all kernel updates are finished before copying back
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    copyToHost(d_W1, net->W1, HIDDEN_SIZE * INPUT_SIZE);
    copyToHost(d_W2, net->W2, OUTPUT_SIZE * HIDDEN_SIZE);
    copyToHost(d_b1, net->b1, HIDDEN_SIZE);
    copyToHost(d_b2, net->b2, OUTPUT_SIZE);

    // --- Cleanup ---
    printf("Freeing GPU memory and resources...\n");
    freeMatrix_gpu(d_W1);
    freeMatrix_gpu(d_W2);
    freeMatrix_gpu(d_b1);
    freeMatrix_gpu(d_b2);
    freeMatrix_gpu(d_hidden);
    freeMatrix_gpu(d_d_output);
    freeMatrix_gpu(d_d_hidden);
    freeMatrix_gpu(d_d_hidden_temp);
    for (int i = 0; i < NUM_STREAMS; ++i) {
        freeMatrix_gpu(d_input[i]);
        freeMatrix_gpu(d_target[i]);
        freeMatrix_gpu(d_output_pre_softmax[i]);
        freeMatrix_gpu(d_output_post_softmax[i]);
        freePinnedHostBuffer(h_input_pinned[i]);
        freePinnedHostBuffer(h_target_pinned[i]);
        freePinnedHostBuffer(h_output_pinned[i]);
        CHECK_CUDA_ERROR(cudaStreamDestroy(streams[i]));
    }
}

// Evaluate accuracy on test data using GPU (with Streams and Shared Memory)
void evaluate_gpu(NeuralNetwork* net, double* images, double* labels, int numImages) {
    printf("Starting GPU Evaluation (Streams + Shared Memory)...\n");
    int correct = 0;

    // --- Create CUDA Streams ---
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CHECK_CUDA_ERROR(cudaStreamCreate(&streams[i]));
    }

    // --- Allocate GPU Memory (Network parameters) ---
    double* d_W1 = allocateMatrix_gpu(HIDDEN_SIZE, INPUT_SIZE);
    double* d_W2 = allocateMatrix_gpu(OUTPUT_SIZE, HIDDEN_SIZE);
    double* d_b1 = allocateMatrix_gpu(1, HIDDEN_SIZE);
    double* d_b2 = allocateMatrix_gpu(1, OUTPUT_SIZE);

    // --- Allocate GPU Memory (Activations - 1 set needed) ---
    double* d_hidden = allocateMatrix_gpu(1, HIDDEN_SIZE);

    // --- Allocate Double Buffers on GPU (Input/Output) ---
    double* d_input[NUM_STREAMS];
    double* d_output_pre_softmax[NUM_STREAMS];
     for (int i = 0; i < NUM_STREAMS; ++i) {
        d_input[i] = allocateMatrix_gpu(1, INPUT_SIZE);
        d_output_pre_softmax[i] = allocateMatrix_gpu(1, OUTPUT_SIZE);
    }

    // --- Allocate Pinned Host Memory (Double Buffers) ---
    double* h_input_pinned[NUM_STREAMS];
    double* h_output_pinned[NUM_STREAMS]; // For receiving pre-softmax output
    for (int i = 0; i < NUM_STREAMS; ++i) {
        h_input_pinned[i] = allocatePinnedHostBuffer(INPUT_SIZE);
        h_output_pinned[i] = allocatePinnedHostBuffer(OUTPUT_SIZE);
    }

    // --- Copy Network Parameters to GPU (Synchronously) ---
    copyToDevice(net->W1, d_W1, HIDDEN_SIZE * INPUT_SIZE);
    copyToDevice(net->W2, d_W2, OUTPUT_SIZE * HIDDEN_SIZE);
    copyToDevice(net->b1, d_b1, HIDDEN_SIZE);
    copyToDevice(net->b2, d_b2, OUTPUT_SIZE);

    // --- Pre-load first sample ---
    memcpy(h_input_pinned[0], &images[0], INPUT_SIZE * sizeof(double));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_input[0], h_input_pinned[0], INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice, streams[0]));

    for (int i = 0; i < numImages; i++) {
        int current_stream_idx = i % NUM_STREAMS;
        int next_stream_idx = (i + 1) % NUM_STREAMS;
        cudaStream_t current_stream = streams[current_stream_idx];

        // --- If not last, prepare and start copy for NEXT sample ---
        if (i + 1 < numImages) {
            cudaStream_t next_stream = streams[next_stream_idx];
            memcpy(h_input_pinned[next_stream_idx], &images[(i + 1) * INPUT_SIZE], INPUT_SIZE * sizeof(double));
            CHECK_CUDA_ERROR(cudaMemcpyAsync(d_input[next_stream_idx], h_input_pinned[next_stream_idx], INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice, next_stream));
        }

        // --- Launch Forward Pass Kernels for CURRENT sample ---
        forward_gpu(d_W1, d_b1, d_W2, d_b2,
                    d_input[current_stream_idx], d_hidden, d_output_pre_softmax[current_stream_idx],
                    current_stream);

        // --- Start Async DtoH Copy for CURRENT sample ---
        CHECK_CUDA_ERROR(cudaMemcpyAsync(h_output_pinned[current_stream_idx], d_output_pre_softmax[current_stream_idx], OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost, current_stream));

        // --- Synchronize CURRENT stream to get result ---
        CHECK_CUDA_ERROR(cudaStreamSynchronize(current_stream));

        // --- CPU Softmax and Accuracy Calculation ---
        softmax(h_output_pinned[current_stream_idx], OUTPUT_SIZE);

        double* current_label_host = &labels[i * OUTPUT_SIZE];
        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (h_output_pinned[current_stream_idx][j] > h_output_pinned[current_stream_idx][pred]) pred = j;
            if (current_label_host[j] > 0.9) actual = j;
        }
        if (pred == actual) correct++;
    }

    // --- Synchronize ALL streams before final print ---
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CHECK_CUDA_ERROR(cudaStreamSynchronize(streams[i]));
    }

    printf("Test Accuracy: %.2f%%\n", (correct / (double)numImages) * 100);

    // --- Cleanup ---
    freeMatrix_gpu(d_W1);
    freeMatrix_gpu(d_W2);
    freeMatrix_gpu(d_b1);
    freeMatrix_gpu(d_b2);
    freeMatrix_gpu(d_hidden);
     for (int i = 0; i < NUM_STREAMS; ++i) {
        freeMatrix_gpu(d_input[i]);
        freeMatrix_gpu(d_output_pre_softmax[i]);
        freePinnedHostBuffer(h_input_pinned[i]);
        freePinnedHostBuffer(h_output_pinned[i]);
        CHECK_CUDA_ERROR(cudaStreamDestroy(streams[i]));
    }
}


// --- Main Function (Mostly Unchanged, ensure paths are correct) ---
int main(int argc, char **argv) {
    printf("MNIST Neural Network (CUDA + Shared Memory + Streams)\n\n");

    int deviceId = 0;
    if(argc > 1) deviceId = atoi(argv[1]);
    cudaDeviceProp deviceProp;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, deviceId));
    if (INPUT_SIZE * sizeof(double) > deviceProp.sharedMemPerBlock) {
         fprintf(stderr, "Warning: INPUT_SIZE (%d doubles) might exceed available shared memory per block (%zu bytes).\n", INPUT_SIZE, deviceProp.sharedMemPerBlock);
    }
    printf("Using GPU Device %d: %s\n", deviceId, deviceProp.name);
    CHECK_CUDA_ERROR(cudaSetDevice(deviceId));

    printf("Loading MNIST data...\n"); 
    const char* train_images_path = "/kaggle/input/mnist-dataset/train-images.idx3-ubyte";
    const char* train_labels_path = "/kaggle/input/mnist-dataset/train-labels.idx1-ubyte";
    const char* test_images_path = "/kaggle/input/mnist-dataset/t10k-images.idx3-ubyte";
    const char* test_labels_path = "/kaggle/input/mnist-dataset/t10k-labels.idx1-ubyte";
    // Add file existence checks as before...
    FILE *f_check; if ((f_check = fopen(train_images_path, "rb")) == NULL) { fprintf(stderr, "Error: Cannot open training images file: %s\n", train_images_path); return 1; } fclose(f_check); if ((f_check = fopen(train_labels_path, "rb")) == NULL) { fprintf(stderr, "Error: Cannot open training labels file: %s\n", train_labels_path); return 1; } fclose(f_check); if ((f_check = fopen(test_images_path, "rb")) == NULL) { fprintf(stderr, "Error: Cannot open test images file: %s\n", test_images_path); return 1; } fclose(f_check); if ((f_check = fopen(test_labels_path, "rb")) == NULL) { fprintf(stderr, "Error: Cannot open test labels file: %s\n", test_labels_path); return 1; } fclose(f_check);

    double* train_images = loadMNISTImages(train_images_path, 60000);
    double* train_labels = loadMNISTLabels(train_labels_path, 60000);
    double* test_images = loadMNISTImages(test_images_path, 10000);
    double* test_labels = loadMNISTLabels(test_labels_path, 10000);
    printf("Data loaded.\n");

    NeuralNetwork* net = createNetwork();
    printf("Network created.\n");

    train_gpu(net, train_images, train_labels, 60000);
    evaluate_gpu(net, test_images, test_labels, 10000);

    printf("Freeing host memory...\n");
    freeNetwork(net);
    freeMatrix(train_images);
    freeMatrix(train_labels);
    freeMatrix(test_images);
    freeMatrix(test_labels);

    CHECK_CUDA_ERROR(cudaDeviceReset());
    printf("Done.\n");
   return 0;
}