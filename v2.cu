%%writefile v2.cu
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

// Allocate memory for a matrix (HOST)
double* allocateMatrix(int rows, int cols) {
    double* mat = (double*)malloc(rows * cols * sizeof(double));
    if (mat == NULL) {
        fprintf(stderr, "Host memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    return mat;
}

// Free allocated matrix memory (HOST)
void freeMatrix(double* mat) {
    free(mat);
}

// Softmax (remains on CPU for simplicity)
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

// Neural network structure (Host pointers)
typedef struct {
    double* W1; // HIDDEN_SIZE * INPUT_SIZE
    double* W2; // OUTPUT_SIZE * HIDDEN_SIZE
    double* b1; // HIDDEN_SIZE
    double* b2; // OUTPUT_SIZE
} NeuralNetwork;

// Initialize neural network (HOST)
NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    net->W1 = allocateMatrix(HIDDEN_SIZE, INPUT_SIZE);
    net->W2 = allocateMatrix(OUTPUT_SIZE, HIDDEN_SIZE);
    net->b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double)); // Use calloc for biases
    net->b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double)); // Use calloc for biases

    if (!net->b1 || !net->b2) {
         fprintf(stderr, "Host bias allocation failed\n");
         exit(EXIT_FAILURE);
    }

    srand(time(NULL));
    // Initialize W1
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            net->W1[i * INPUT_SIZE + j] = ((double)rand() / RAND_MAX - 0.5) * 0.1; // Small random values centered around 0
        }
    }
    // Initialize W2
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            net->W2[i * HIDDEN_SIZE + j] = ((double)rand() / RAND_MAX - 0.5) * 0.1; // Small random values centered around 0
        }
    }
    // Biases already initialized to 0 by calloc

    return net;
}

// Free network memory (HOST)
void freeNetwork(NeuralNetwork* net) {
    if (net) {
        freeMatrix(net->W1);
        freeMatrix(net->W2);
        free(net->b1);
        free(net->b2);
        free(net);
    }
}

// --- CUDA Device Memory Management ---

// Allocate memory on GPU
double* allocateMatrix_gpu(int rows, int cols) {
    double* mat_gpu;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&mat_gpu, rows * cols * sizeof(double)));
    return mat_gpu;
}

// Free memory on GPU
void freeMatrix_gpu(double* mat_gpu) {
    if (mat_gpu) {
        CHECK_CUDA_ERROR(cudaFree(mat_gpu));
    }
}

// Copy data from Host to Device
void copyToDevice(double* host_data, double* device_data, int size) {
    CHECK_CUDA_ERROR(cudaMemcpy(device_data, host_data, size * sizeof(double), cudaMemcpyHostToDevice));
}

// Copy data from Device to Host
void copyToHost(double* device_data, double* host_data, int size) {
    CHECK_CUDA_ERROR(cudaMemcpy(host_data, device_data, size * sizeof(double), cudaMemcpyDeviceToHost));
}


// --- CUDA Kernels ---

// Kernel for Matrix-Vector Multiplication: C = A * B + Bias (or C = A * B if bias is NULL)
// A is matrix (rows x cols), B is vector (cols x 1), C is vector (rows x 1)
_global_ void matrixVectorMulBiasKernel(double* A, double* B, double* C, double* Bias, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows) {
        double sum = 0.0;
        for (int k = 0; k < cols; k++) {
            sum += A[row * cols + k] * B[k];
        }
        if (Bias != NULL) {
             C[row] = sum + Bias[row];
        } else {
             C[row] = sum;
        }
    }
}

// Kernel for ReLU activation: X = max(0, X)
_global_ void reluKernel(double* X, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        X[idx] = (X[idx] > 0.0) ? X[idx] : 0.0;
    }
}

// Kernel for calculating output gradient: d_output = output - target
_global_ void outputGradientKernel(double* output, double* target, double* d_output, int size) {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (idx < size) {
         d_output[idx] = output[idx] - target[idx];
     }
}


// Kernel for calculating hidden gradient part 1: d_hidden_temp = W2^T * d_output
// W2 is (OUTPUT_SIZE x HIDDEN_SIZE), d_output is (OUTPUT_SIZE x 1), d_hidden_temp is (HIDDEN_SIZE x 1)
_global_ void hiddenGradientPart1Kernel(double* W2, double* d_output, double* d_hidden_temp, int hidden_size, int output_size) {
    int row = blockIdx.x * blockDim.x + threadIdx.x; // row corresponds to hidden unit index

    if (row < hidden_size) {
        double sum = 0.0;
        for (int k = 0; k < output_size; k++) {
            // Access W2 transposed: W2[k][row] -> W2[k * hidden_size + row]
            sum += W2[k * hidden_size + row] * d_output[k];
        }
        d_hidden_temp[row] = sum;
    }
}

// Kernel for calculating hidden gradient part 2 (applying ReLU derivative): d_hidden = d_hidden_temp * (hidden > 0)
_global_ void hiddenGradientPart2Kernel(double* d_hidden_temp, double* hidden_activations, double* d_hidden, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_hidden[idx] = d_hidden_temp[idx] * (hidden_activations[idx] > 0.0 ? 1.0 : 0.0);
    }
}

// Kernel for updating weights: W -= learning_rate * d_layer * activation_prev^T
// W is (rows x cols), d_layer is (rows x 1), activation_prev is (cols x 1)
_global_ void updateWeightsKernel(double* W, double* d_layer, double* activation_prev, double learning_rate, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        W[idx] -= learning_rate * d_layer[row] * activation_prev[col];
    }
}

// Kernel for updating biases: B -= learning_rate * d_layer
_global_ void updateBiasesKernel(double* B, double* d_layer, double learning_rate, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        B[idx] -= learning_rate * d_layer[idx];
    }
}


// --- GPU Forward and Backward Pass ---

// Forward pass on GPU
// Takes device pointers for network parameters and input
// Outputs device pointers for hidden and output activations (pre-softmax)
void forward_gpu(double* d_W1, double* d_b1, double* d_W2, double* d_b2,
                 double* d_input, double* d_hidden, double* d_output_pre_softmax)
{
    int threadsPerBlock = 256;

    // Calculate hidden layer: hidden = relu(W1 * input + b1)
    dim3 gridDimHidden((HIDDEN_SIZE + threadsPerBlock - 1) / threadsPerBlock);
    dim3 blockDimHidden(threadsPerBlock);
    matrixVectorMulBiasKernel<<<gridDimHidden, blockDimHidden>>>(d_W1, d_input, d_hidden, d_b1, HIDDEN_SIZE, INPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for kernel launch errors
    reluKernel<<<gridDimHidden, blockDimHidden>>>(d_hidden, HIDDEN_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Calculate output layer (pre-softmax): output_pre_softmax = W2 * hidden + b2
    dim3 gridDimOutput((OUTPUT_SIZE + threadsPerBlock - 1) / threadsPerBlock);
    dim3 blockDimOutput(threadsPerBlock);
    matrixVectorMulBiasKernel<<<gridDimOutput, blockDimOutput>>>(d_W2, d_hidden, d_output_pre_softmax, d_b2, OUTPUT_SIZE, HIDDEN_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Synchronization might be needed if subsequent operations depend on completion
    // cudaDeviceSynchronize(); // Add if necessary, e.g., before copying back to host
}

// Backward pass on GPU
// Takes device pointers for network, input, activations, and target
// Updates network parameters (d_W1, d_b1, d_W2, d_b2) on the device
void backward_gpu(double* d_W1, double* d_b1, double* d_W2, double* d_b2,
                  double* d_input, double* d_hidden, double* d_output, // Note: d_output here is AFTER softmax
                  double* d_target,
                  double* d_d_output, double* d_d_hidden, double* d_d_hidden_temp) // Device gradient buffers
{
    int threadsPerBlock = 256;

    // 1. Compute output layer gradient: d_output = output - target
    //    (Assuming output is already post-softmax from host calculation)
    dim3 gridDimOutput((OUTPUT_SIZE + threadsPerBlock - 1) / threadsPerBlock);
    dim3 blockDimOutput(threadsPerBlock);
    outputGradientKernel<<<gridDimOutput, blockDimOutput>>>(d_output, d_target, d_d_output, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // 2. Compute hidden layer gradient: d_hidden = (W2^T * d_output) * relu_derivative(hidden)
    dim3 gridDimHidden((HIDDEN_SIZE + threadsPerBlock - 1) / threadsPerBlock);
    dim3 blockDimHidden(threadsPerBlock);
    // Part 1: d_hidden_temp = W2^T * d_output
    hiddenGradientPart1Kernel<<<gridDimHidden, blockDimHidden>>>(d_W2, d_d_output, d_d_hidden_temp, HIDDEN_SIZE, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());
    // Part 2: d_hidden = d_hidden_temp * relu_derivative(hidden)
    hiddenGradientPart2Kernel<<<gridDimHidden, blockDimHidden>>>(d_d_hidden_temp, d_hidden, d_d_hidden, HIDDEN_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // 3. Update weights W2: W2 -= LR * d_output * hidden^T
    dim3 blockDimUpdateW2(16, 16); // Example 2D block
    dim3 gridDimUpdateW2((HIDDEN_SIZE + blockDimUpdateW2.x - 1) / blockDimUpdateW2.x,
                         (OUTPUT_SIZE + blockDimUpdateW2.y - 1) / blockDimUpdateW2.y);
    updateWeightsKernel<<<gridDimUpdateW2, blockDimUpdateW2>>>(d_W2, d_d_output, d_hidden, LEARNING_RATE, OUTPUT_SIZE, HIDDEN_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // 4. Update weights W1: W1 -= LR * d_hidden * input^T
    dim3 blockDimUpdateW1(16, 16); // Example 2D block
    dim3 gridDimUpdateW1((INPUT_SIZE + blockDimUpdateW1.x - 1) / blockDimUpdateW1.x,
                         (HIDDEN_SIZE + blockDimUpdateW1.y - 1) / blockDimUpdateW1.y);
    updateWeightsKernel<<<gridDimUpdateW1, blockDimUpdateW1>>>(d_W1, d_d_hidden, d_input, LEARNING_RATE, HIDDEN_SIZE, INPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // 5. Update biases b2: b2 -= LR * d_output
    updateBiasesKernel<<<gridDimOutput, blockDimOutput>>>(d_b2, d_d_output, LEARNING_RATE, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // 6. Update biases b1: b1 -= LR * d_hidden
    updateBiasesKernel<<<gridDimHidden, blockDimHidden>>>(d_b1, d_d_hidden, LEARNING_RATE, HIDDEN_SIZE);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Synchronization might be needed before the next iteration
    // cudaDeviceSynchronize();
}


// --- MNIST Loading Functions (Unchanged) ---

// Read MNIST dataset into a flat 1D array (returns double*)
double* loadMNISTImages(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 16, SEEK_SET); // Skip header
    double* images = allocateMatrix(numImages, INPUT_SIZE);
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            unsigned char pixel;
            if (fread(&pixel, sizeof(unsigned char), 1, file) != 1) {
                fprintf(stderr, "Error: Failed to read pixel\n"); fclose(file); exit(EXIT_FAILURE);
            }
            images[i * INPUT_SIZE + j] = pixel / 255.0;
        }
    }
    fclose(file);
    return images;
}

// Read MNIST labels into a flat 1D array (one-hot encoded, returns double*)
double* loadMNISTLabels(const char* filename, int numLabels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename); exit(1);
    }
    fseek(file, 8, SEEK_SET); // Skip header
    double* labels = allocateMatrix(numLabels, OUTPUT_SIZE);
    for (int i = 0; i < numLabels; i++) {
        unsigned char label_val;
        if (fread(&label_val, sizeof(unsigned char), 1, file) != 1) {
            fprintf(stderr, "Error: Failed to read label\n"); fclose(file); exit(EXIT_FAILURE);
        }
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            labels[i * OUTPUT_SIZE + j] = (j == label_val) ? 1.0 : 0.0;
        }
    }
    fclose(file);
    return labels;
}


// --- Modified Training and Evaluation ---

// Train network using GPU
void train_gpu(NeuralNetwork* net, double* images, double* labels, int numImages) {
    clock_t total_start = clock();

    // --- Allocate GPU Memory ---
    double* d_W1 = allocateMatrix_gpu(HIDDEN_SIZE, INPUT_SIZE);
    double* d_W2 = allocateMatrix_gpu(OUTPUT_SIZE, HIDDEN_SIZE);
    double* d_b1 = allocateMatrix_gpu(1, HIDDEN_SIZE);
    double* d_b2 = allocateMatrix_gpu(1, OUTPUT_SIZE);

    double* d_input = allocateMatrix_gpu(1, INPUT_SIZE);
    double* d_target = allocateMatrix_gpu(1, OUTPUT_SIZE);
    double* d_hidden = allocateMatrix_gpu(1, HIDDEN_SIZE);
    double* d_output_pre_softmax = allocateMatrix_gpu(1, OUTPUT_SIZE); // Before softmax
    double* d_output_post_softmax = allocateMatrix_gpu(1, OUTPUT_SIZE); // After softmax (for gradient calc)

    // Allocate GPU gradient buffers
    double* d_d_output = allocateMatrix_gpu(1, OUTPUT_SIZE);
    double* d_d_hidden = allocateMatrix_gpu(1, HIDDEN_SIZE);
    double* d_d_hidden_temp = allocateMatrix_gpu(1, HIDDEN_SIZE); // Temporary buffer

    // Host buffer for output layer (to run softmax and calculate loss/accuracy)
    double h_output[OUTPUT_SIZE];

    // --- Copy Initial Network Parameters to GPU ---
    copyToDevice(net->W1, d_W1, HIDDEN_SIZE * INPUT_SIZE);
    copyToDevice(net->W2, d_W2, OUTPUT_SIZE * HIDDEN_SIZE);
    copyToDevice(net->b1, d_b1, HIDDEN_SIZE);
    copyToDevice(net->b2, d_b2, OUTPUT_SIZE);

    printf("Starting GPU Training...\n");

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double epoch_loss = 0.0;
        int epoch_correct = 0;

        for (int i = 0; i < numImages; i++) {
            // Get pointers to current host image and label
            double* current_image = &images[i * INPUT_SIZE];
            double* current_label = &labels[i * OUTPUT_SIZE];

            // --- Copy Input/Target to GPU ---
            copyToDevice(current_image, d_input, INPUT_SIZE);
            copyToDevice(current_label, d_target, OUTPUT_SIZE);

            // --- Forward Pass on GPU ---
            forward_gpu(d_W1, d_b1, d_W2, d_b2, d_input, d_hidden, d_output_pre_softmax);

            // --- Copy Output (pre-softmax) to Host for Softmax & Loss/Accuracy ---
            copyToHost(d_output_pre_softmax, h_output, OUTPUT_SIZE);
            CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Ensure copy is complete

            // --- Softmax on CPU ---
            softmax(h_output, OUTPUT_SIZE);

            // --- Compute Loss & Accuracy on CPU ---
            double sample_loss = 0.0;
            for (int k = 0; k < OUTPUT_SIZE; k++) {
                 sample_loss -= current_label[k] * log(h_output[k] + 1e-9); // Add epsilon
            }
            epoch_loss += sample_loss;

            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (h_output[j] > h_output[pred]) pred = j;
                if (current_label[j] > 0.9) actual = j; // Check for 1.0 in one-hot
            }
            if (pred == actual) epoch_correct++;

            // --- Copy Softmax Output back to GPU (for gradient calculation) ---
            copyToDevice(h_output, d_output_post_softmax, OUTPUT_SIZE);

            // --- Backward Pass on GPU ---
            backward_gpu(d_W1, d_b1, d_W2, d_b2,
                         d_input, d_hidden, d_output_post_softmax, d_target,
                         d_d_output, d_d_hidden, d_d_hidden_temp);

            // Optional: Synchronize after each sample if debugging or precise timing needed
            // CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        } // End image loop

        // Synchronize after epoch to ensure all GPU work is done before timing/printing
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, epoch_loss / numImages, (epoch_correct / (double)numImages) * 100, get_time(epoch_start));

    } // End epoch loop

    printf("Total training time: %.3fs\n", get_time(total_start));

    // --- Copy Final Weights Back to Host (optional, needed for CPU evaluation) ---
    printf("Copying final weights back to host...\n");
    copyToHost(d_W1, net->W1, HIDDEN_SIZE * INPUT_SIZE);
    copyToHost(d_W2, net->W2, OUTPUT_SIZE * HIDDEN_SIZE);
    copyToHost(d_b1, net->b1, HIDDEN_SIZE);
    copyToHost(d_b2, net->b2, OUTPUT_SIZE);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Ensure copies complete

    // --- Free GPU Memory ---
    printf("Freeing GPU memory...\n");
    freeMatrix_gpu(d_W1);
    freeMatrix_gpu(d_W2);
    freeMatrix_gpu(d_b1);
    freeMatrix_gpu(d_b2);
    freeMatrix_gpu(d_input);
    freeMatrix_gpu(d_target);
    freeMatrix_gpu(d_hidden);
    freeMatrix_gpu(d_output_pre_softmax);
    freeMatrix_gpu(d_output_post_softmax);
    freeMatrix_gpu(d_d_output);
    freeMatrix_gpu(d_d_hidden);
    freeMatrix_gpu(d_d_hidden_temp);
}

// Evaluate accuracy on test data using GPU for forward pass
void evaluate_gpu(NeuralNetwork* net, double* images, double* labels, int numImages) {
    printf("Starting GPU Evaluation...\n");
    int correct = 0;

    // --- Allocate GPU Memory (only what's needed for forward pass) ---
    double* d_W1 = allocateMatrix_gpu(HIDDEN_SIZE, INPUT_SIZE);
    double* d_W2 = allocateMatrix_gpu(OUTPUT_SIZE, HIDDEN_SIZE);
    double* d_b1 = allocateMatrix_gpu(1, HIDDEN_SIZE);
    double* d_b2 = allocateMatrix_gpu(1, OUTPUT_SIZE);
    double* d_input = allocateMatrix_gpu(1, INPUT_SIZE);
    double* d_hidden = allocateMatrix_gpu(1, HIDDEN_SIZE);
    double* d_output_pre_softmax = allocateMatrix_gpu(1, OUTPUT_SIZE);

    // Host buffer for output
    double h_output[OUTPUT_SIZE];

    // --- Copy Network Parameters to GPU ---
    // Assumes 'net' contains the trained parameters (either trained on CPU or copied back after GPU training)
    copyToDevice(net->W1, d_W1, HIDDEN_SIZE * INPUT_SIZE);
    copyToDevice(net->W2, d_W2, OUTPUT_SIZE * HIDDEN_SIZE);
    copyToDevice(net->b1, d_b1, HIDDEN_SIZE);
    copyToDevice(net->b2, d_b2, OUTPUT_SIZE);

    for (int i = 0; i < numImages; i++) {
        double* current_image = &images[i * INPUT_SIZE];
        double* current_label = &labels[i * OUTPUT_SIZE];

        // Copy input to GPU
        copyToDevice(current_image, d_input, INPUT_SIZE);

        // Forward pass on GPU
        forward_gpu(d_W1, d_b1, d_W2, d_b2, d_input, d_hidden, d_output_pre_softmax);

        // Copy output (pre-softmax) back to host
        copyToHost(d_output_pre_softmax, h_output, OUTPUT_SIZE);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Ensure copy is complete

        // Softmax on CPU
        softmax(h_output, OUTPUT_SIZE);

        // Calculate prediction on CPU
        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (h_output[j] > h_output[pred]) pred = j;
            if (current_label[j] > 0.9) actual = j;
        }
        if (pred == actual) correct++;
    }

    printf("Test Accuracy: %.2f%%\n", (correct / (double)numImages) * 100);

    // --- Free GPU Memory ---
    freeMatrix_gpu(d_W1);
    freeMatrix_gpu(d_W2);
    freeMatrix_gpu(d_b1);
    freeMatrix_gpu(d_b2);
    freeMatrix_gpu(d_input);
    freeMatrix_gpu(d_hidden);
    freeMatrix_gpu(d_output_pre_softmax);
}


// --- Main Function ---
int main(int argc, char **argv) {
    printf("MNIST Neural Network (CUDA Version)\n\n");

    // --- Select GPU Device ---
    int deviceId = 0; // Default to device 0
    if(argc > 1) {
        deviceId = atoi(argv[1]);
    }
    cudaDeviceProp deviceProp;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, deviceId));
    printf("Using GPU Device %d: %s\n", deviceId, deviceProp.name);
    CHECK_CUDA_ERROR(cudaSetDevice(deviceId));


    // --- Load Data (Host) ---
    printf("Loading MNIST data...\n");
    // IMPORTANT: Update these paths to your actual MNIST file locations 
    const char* train_images_path = "/kaggle/input/mnist-dataset/train-images.idx3-ubyte";
    const char* train_labels_path = "/kaggle/input/mnist-dataset/train-labels.idx1-ubyte";
    const char* test_images_path = "/kaggle/input/mnist-dataset/t10k-images.idx3-ubyte";
    const char* test_labels_path = "/kaggle/input/mnist-dataset/t10k-labels.idx1-ubyte";

    // Check if files exist before loading (basic check)
    FILE *f_check;
    if ((f_check = fopen(train_images_path, "rb")) == NULL) {
        fprintf(stderr, "Error: Cannot open training images file: %s\n", train_images_path);
        fprintf(stderr, "Please ensure MNIST dataset files are in the correct directory or update the paths in the code.\n");
        return 1;
    } fclose(f_check);
     if ((f_check = fopen(train_labels_path, "rb")) == NULL) {
        fprintf(stderr, "Error: Cannot open training labels file: %s\n", train_labels_path);
        return 1;
    } fclose(f_check);
     if ((f_check = fopen(test_images_path, "rb")) == NULL) {
        fprintf(stderr, "Error: Cannot open test images file: %s\n", test_images_path);
        return 1;
    } fclose(f_check);
     if ((f_check = fopen(test_labels_path, "rb")) == NULL) {
        fprintf(stderr, "Error: Cannot open test labels file: %s\n", test_labels_path);
        return 1;
    } fclose(f_check);


    double* train_images = loadMNISTImages(train_images_path, 60000);
    double* train_labels = loadMNISTLabels(train_labels_path, 60000);
    double* test_images = loadMNISTImages(test_images_path, 10000);
    double* test_labels = loadMNISTLabels(test_labels_path, 10000);
    printf("Data loaded.\n");

    // --- Create Network (Host) ---
    NeuralNetwork* net = createNetwork();
    printf("Network created.\n");

    // --- Train on GPU ---
    train_gpu(net, train_images, train_labels, 60000);

    // --- Evaluate on GPU ---
    // The 'net' structure now holds the trained weights copied back from the GPU
    evaluate_gpu(net, test_images, test_labels, 10000);

    // --- Cleanup Host Memory ---
    printf("Freeing host memory...\n");
    freeNetwork(net);
    freeMatrix(train_images);
    freeMatrix(train_labels);
    freeMatrix(test_images);
    freeMatrix(test_labels);

    // --- Reset Device ---
    CHECK_CUDA_ERROR(cudaDeviceReset());
    printf("Done.\n");

  return 0;
}