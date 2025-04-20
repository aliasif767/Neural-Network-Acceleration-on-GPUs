# Neural-Network-Acceleration-on-GPUs
This project explores different implementations of a computational algorithm using CUDA to demonstrate the performance evolution from a sequential baseline to a highly optimized GPU implementation.

🔹 Version 2: Naive GPU Implementation (v2.cu)
This version represents a basic parallelization of the sequential algorithm. Key aspects include:

Kernel Offloading: The core computation is offloaded to the GPU using a single CUDA kernel.

Thread Mapping: Each thread is assigned to process one or more data elements in a 1D grid configuration.

Global Memory Access: All input and output data is accessed directly from global memory without optimization.

No Memory Hierarchy Usage: No usage of shared, constant, or texture memory.

No Load Balancing: Does not consider warp divergence or occupancy; threads may be underutilized.

⚠️ This version does not optimize for GPU memory latency, thread divergence, or occupancy, making it a good benchmark to compare performance improvements in V3.

🔹 Version 3: Optimized GPU Implementation (v3.cu)
This version significantly enhances the performance using multiple CUDA optimization techniques:

✅ a. Launch Configuration
Tuned Block and Grid Sizes: Used cudaOccupancyMaxPotentialBlockSize (or experimentation) to find the optimal number of threads per block to maximize occupancy.

Boundary Checks: Ensured that threads beyond the input size are safely handled to avoid memory access violations.

✅ b. Occupancy Optimization
Increased Occupancy: Selected a thread/block configuration that reduces idle cores and maximizes Streaming Multiprocessor (SM) usage.

Reduced Register Pressure: Optimized kernel code to reduce register usage and allow more active warps.

✅ c. Communication Optimizations
Avoiding Redundant Memory Accesses: Moved frequently reused values to registers or shared memory.

Minimized Global Writes: Threads write back to global memory only once after completing calculations.

✅ d. Memory Optimizations
Shared Memory Usage: Utilized __shared__ memory for intermediate results and reducing redundant global memory accesses.

Memory Coalescing: Ensured memory accesses are coalesced to maximize memory bandwidth.

Loop Unrolling (if applicable): Inlined small loops inside the kernel to reduce control overhead and improve instruction throughput.

Constant Memory (Optional): Used for read-only data where appropriate to reduce bandwidth pressure on global memory.