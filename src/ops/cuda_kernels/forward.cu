#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <iostream>
#include <vector_types.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_pipeline_primitives.h>
#include <cuda_fp16.hpp>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>


template <typename TA, typename TB>
__device__ __always_inline void loop_mul_add(TA* ptr_a, TB* ptr_b, TB weight, int stride_a, int stride_b, int n){
    #pragma unroll
    for(int i=0; i<n; i++){
        *ptr_a = (TA)(*ptr_a + (*ptr_b) * weight);
        ptr_a += stride_a;
        ptr_b += stride_b;
    }
}

template <typename TA, typename TB>
__device__ __always_inline void loop_mul_load(TA* ptr_a, TB* ptr_b, TB weight, int stride_a, int stride_b, int n){
    #pragma unroll
    for(int i=0; i<n; i++){
        *ptr_a = (TA)((*ptr_b) * weight);
        ptr_a += stride_a;
        ptr_b += stride_b;
    }
}

template <typename TA, typename TB>
__device__ __always_inline void loop_load(TA* ptr_a, TB* ptr_b, int stride_a, int stride_b, int n){
#pragma unroll
    for(int i=0; i<n; i++){
        *ptr_a = (TA)((*ptr_b));
        ptr_a += stride_a;
        ptr_b += stride_b;
    }
}

template <typename TA>
__device__ __always_inline void loop_reset(TA* ptr_a, int stride, int n){
#pragma unroll
    for(int i=0; i<n; i++){
        *ptr_a = 0;
        ptr_a += stride;
    }
}

// B, H, W, C, BLOCK_DIM must be multiple of C
template <typename math_t, typename scalar_t, int K, int BLOCK_DIM>
__global__ void dcn_forward_kernel_register(const int H, const int W, const int C, scalar_t* ptr_value, scalar_t* ptr_deformables, scalar_t* ptr_weights, scalar_t* ptr_out){
    int work_id = threadIdx.x;
    int bid = blockIdx.z;
    int gid = blockIdx.y;
    int G = gridDim.y;
    int c_blockid = blockIdx.x;
    int work_load = (H*W/blockDim.x);

    extern __shared__ int shm[];
    math_t* math_buffer = reinterpret_cast<math_t*>(shm);

    math_t register_bufferA[BLOCK_DIM] = {0};
    int base_c = c_blockid*BLOCK_DIM;

#pragma unroll
    for(int i=0; i<work_load; i++){
        int job_id = work_load*work_id + i;
        int offset2 = (bid*H*W*G*C + job_id*C*G + gid*C + base_c);
        for(int j=0; j<BLOCK_DIM; j++){
            if((base_c+j) < C){
                   math_buffer[job_id*BLOCK_DIM +j] = (math_t)*(ptr_value + offset2 +j);
            }
        }
    }
    __syncthreads();

    work_load = (H*W)/blockDim.x;
    int offset = 0;
    for(int i=0; i<work_load; i++){
        int job_id = (work_id*work_load+i);
        int hid = job_id/W;
        int wid = job_id%W;
        loop_reset<math_t>(register_bufferA, 1, BLOCK_DIM);
#pragma unroll
        for(int k=0; k<K; k++){
            // read weights to register
            offset = bid*K*H*W*G + hid*W*K*G + wid*K*G + gid*K +k;
            math_t weight = *(ptr_weights + offset);
            // read deformables to register
            offset = offset*2;
            math_t x = *(ptr_deformables + offset) + wid;
            math_t y = *(ptr_deformables + offset + 1) + hid;
            int floor_x = x;
            int floor_y = y;
            int ceil_x = floor_x + 1;
            int ceil_y = floor_y + 1;

            // reset A buffer and top left
            math_t tl_weight = (ceil_x - x)*(ceil_y - y)*weight;
            if( (0<= floor_y) and (floor_y < H) and (0<= floor_x) and (floor_x < W)){
                loop_mul_add<math_t, math_t>(register_bufferA, (math_t*)&math_buffer[floor_y*W+floor_x], tl_weight, 1, 1, BLOCK_DIM);
            }
            // load top right
            math_t tr_weight = (x - floor_x)*(ceil_y - y)*weight;
            if((0<= floor_y) and (floor_y < H) and (0<= ceil_x) and (ceil_x < W)){
                loop_mul_add<math_t, math_t>(register_bufferA, (math_t*)&math_buffer[floor_y*W + ceil_x], tr_weight, 1, 1, BLOCK_DIM);
            }

            // load bottom left
            math_t bl_weight = (ceil_x - x)*(y - floor_y)*weight;
            if((0<= ceil_y) and (ceil_y < H) and (0<= floor_x) and (floor_x < W) ){
                loop_mul_add<math_t, math_t>(register_bufferA, (math_t*)&math_buffer[ceil_y*W+floor_x], bl_weight, 1, 1, BLOCK_DIM);
            }
            // load bottom right
            math_t br_weight = (x - floor_x)*(y - floor_y)*weight;
            if((0<=ceil_y) and (ceil_y < H) and (0<=ceil_x) and (ceil_x < W)){
                loop_mul_add<math_t, math_t>(register_bufferA, (math_t*)&math_buffer[ceil_y*W+ceil_x], br_weight, 1, 1, BLOCK_DIM);
            }

        }
        int offset2 = (bid*H*W*G*C + job_id*C*G + gid*C + base_c);
#pragma unroll
        for(int j=0; j<BLOCK_DIM; j++){
            if((base_c+j) < C){
                *(ptr_out + offset2 + j) = (scalar_t)register_bufferA[j];
            }
        }  
    }
    __syncthreads();
}





// B, H, W, C, BLOCK_DIM must be multiple of C
template <typename math_t, typename scalar_t, int transfer_length, int K, int BLOCK_DIM>
__global__ void dcn_forward_kernel_pipeline(const int H, const int W, const int C, scalar_t* ptr_value, scalar_t* ptr_deformables, scalar_t* ptr_weights, scalar_t* ptr_out){
    int work_id = threadIdx.x;
    int bid = blockIdx.z;
    int gid = blockIdx.y;
    int G = gridDim.y;
    int c_blockid = blockIdx.x;
    int work_load = (H*W/blockDim.x);

    extern __shared__ int shm[];
    math_t* math_buffer = reinterpret_cast<math_t*>(shm);
    scalar_t* io_buffer = reinterpret_cast<scalar_t*>(shm) + H*W*BLOCK_DIM*sizeof(math_t)/sizeof(scalar_t);
    math_t register_bufferA[BLOCK_DIM] = {0};
    int base_c = c_blockid*BLOCK_DIM;

    int num_transfers = BLOCK_DIM/transfer_length;
#pragma unroll
    for(int i=0; i<work_load; i++){
        int job_id = work_load*work_id + i;
        int offset2 = (bid*H*W*G*C + job_id*C*G + gid*C + base_c)/transfer_length;
        for(int j=0; j<num_transfers; j++){
            if((base_c+j*transfer_length) < C){
                   __pipeline_memcpy_async((long*)(&math_buffer[job_id]) + j, (long *)ptr_value + offset2 + j, sizeof(long));
            }
        }
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    work_load = (H*W)/blockDim.x;
    int offset = 0;
    for(int i=0; i<work_load; i++){
        int job_id = (work_id*work_load+i);
        int hid = job_id/W;
        int wid = job_id%W;
        loop_reset<math_t>(register_bufferA, 1, BLOCK_DIM);
        loop_reset<scalar_t>((scalar_t*)&io_buffer[hid*W+wid], 1, BLOCK_DIM);
#pragma unroll
        for(int k=0; k<K; k++){
            // read weights to register
            offset = bid*K*H*W*G + hid*W*K*G + wid*K*G + gid*K +k;
            math_t weight = *(ptr_weights + offset);
            // read deformables to register
            offset = offset*2;
            math_t x = *(ptr_deformables + offset) + wid;
            math_t y = *(ptr_deformables + offset + 1) + hid;
            int floor_x = x;
            int floor_y = y;
            int ceil_x = floor_x + 1;
            int ceil_y = floor_y + 1;

            // reset A buffer and top left
            math_t tl_weight = (ceil_x - x)*(ceil_y - y)*weight;
            if( (0<= floor_y) and (floor_y < H) and (0<= floor_x) and (floor_x < W)){
                loop_mul_add<math_t, math_t>(register_bufferA, (math_t*)&math_buffer[floor_y*W+floor_x], tl_weight, 1, 1, BLOCK_DIM);
            }
            // load top right
            math_t tr_weight = (x - floor_x)*(ceil_y - y)*weight;
            if((0<= floor_y) and (floor_y < H) and (0<= ceil_x) and (ceil_x < W)){
                loop_mul_add<math_t, math_t>(register_bufferA, (math_t*)&math_buffer[floor_y*W + ceil_x], tr_weight, 1, 1, BLOCK_DIM);
            }

            // load bottom left
            math_t bl_weight = (ceil_x - x)*(y - floor_y)*weight;
            if((0<= ceil_y) and (ceil_y < H) and (0<= floor_x) and (floor_x < W) ){
                loop_mul_add<math_t, math_t>(register_bufferA, (math_t*)&math_buffer[ceil_y*W+floor_x], bl_weight, 1, 1, BLOCK_DIM);
            }
            // load bottom right
            math_t br_weight = (x - floor_x)*(y - floor_y)*weight;
            if((0<=ceil_y) and (ceil_y < H) and (0<=ceil_x) and (ceil_x < W)){
                loop_mul_add<math_t, math_t>(register_bufferA, (math_t*)&math_buffer[ceil_y*W+ceil_x], br_weight, 1, 1, BLOCK_DIM);
            }

        }
        loop_load<scalar_t, math_t>((scalar_t*)&io_buffer[hid*W+wid], register_bufferA, 1, 1, BLOCK_DIM);

    }

    __syncthreads();

#pragma unroll
    for(int i=0; i<work_load; i++){
        int job_id = work_load*work_id + i;
        // int offset1 = job_id*num_transfers;
        int offset2 = (bid*H*W*G*C + job_id*C*G + gid*C + base_c)/transfer_length;
#pragma unroll
        for(int j=0; j<num_transfers; j++){
            if((base_c+j*transfer_length) < C){
                *((long *)ptr_out + offset2 + j) = *((long *)(&io_buffer[job_id]) +j);
            }
        }
    }
}

template<int C_BLOCK_DIM, int THREADS>
void dcn_forward(const int B, const int G, const int C, const int H, const int W, torch::Tensor value, torch::Tensor deformables, torch::Tensor weights, torch::Tensor out) {
    
    int NUM_C_BLOCK = (C+C_BLOCK_DIM-1)/C_BLOCK_DIM;
    dim3 launch_threads_per_block(THREADS);
    dim3 launch_blocks(NUM_C_BLOCK, G, B);
    int shm_size = H*W*C_BLOCK_DIM*sizeof(at::Half);
    switch (value.type().scalarType()) {
        case at::ScalarType::Half:
            return dcn_forward_kernel_register<at::Half, at::Half, 9, C_BLOCK_DIM><<<launch_blocks, launch_threads_per_block, shm_size>>>(
                    H, W, C,
                    value.data_ptr<at::Half>(),
                    deformables.data_ptr<at::Half>(),
                    weights.data_ptr<at::Half>(),
                    out.data_ptr<at::Half>());
        case at::ScalarType::Float:
            return dcn_forward_kernel_register<at::Half, float, 9, C_BLOCK_DIM><<<launch_blocks, launch_threads_per_block, shm_size>>>(
                    H, W, C,
                    value.data_ptr<float>(),
                    deformables.data_ptr<float>(),
                    weights.data_ptr<float>(),
                    out.data_ptr<float>());
        default:
            printf("running error");
    }
}

template<int C_BLOCK_DIM, int THREADS>
void dcn_forward_pipeline(int B, int G, int C, int H, int W, torch::Tensor value, torch::Tensor deformables, torch::Tensor weights, torch::Tensor out) {

    int NUM_C_BLOCK = (C+C_BLOCK_DIM-1)/C_BLOCK_DIM;
    dim3 launch_threads_per_block(THREADS);
    dim3 launch_blocks(NUM_C_BLOCK, G, B);
    int shm_size = 2*H*W*C_BLOCK_DIM*sizeof(at::Half);
    switch (value.type().scalarType()) {
        case at::ScalarType::Half:
            return dcn_forward_kernel_pipeline<at::Half, at::Half, 4, 9, C_BLOCK_DIM><<<launch_blocks, launch_threads_per_block, shm_size>>>(
                    H, W, C,
                    value.data_ptr<at::Half>(),
                    deformables.data_ptr<at::Half>(),
                    weights.data_ptr<at::Half>(),
                    out.data_ptr<at::Half>());
        case at::ScalarType::BFloat16:
            return dcn_forward_kernel_pipeline<at::BFloat16, at::BFloat16, 4, 9, C_BLOCK_DIM><<<launch_blocks, launch_threads_per_block, shm_size>>>(
                    H, W, C,
                    value.data_ptr<at::BFloat16>(),
                    deformables.data_ptr<at::BFloat16>(),
                    weights.data_ptr<at::BFloat16>(),
                    out.data_ptr<at::BFloat16>());
        default:
            printf("running error");
    }
}

// PyBind11 bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//m.def("dcn_forward_c1_f4", &dcn_forward<1, 4>, "CUDA dcn forward");
//m.def("dcn_forward_c2_f4", &dcn_forward<2, 4>, "CUDA dcn forward");
m.def("dcn_forward_l256_c4", &dcn_forward<4, 256>, "CUDA dcn forward");
m.def("dcn_forward_l256_c8", &dcn_forward<8, 256>, "CUDA dcn forward");
m.def("dcn_forward_l256_c16", &dcn_forward<16, 256>, "CUDA dcn forward");
m.def("dcn_forward_pipeline_l256_c4", &dcn_forward_pipeline<4, 256>, "CUDA dcn forward");
m.def("dcn_forward_pipeline_l256_c8", &dcn_forward_pipeline<8, 256>, "CUDA dcn forward");
m.def("dcn_forward_pipeline_l256_c16", &dcn_forward_pipeline<16, 256>, "CUDA dcn forward");
// m.def("dcn_forward_l256_c32", &dcn_forward<256, 32, 256>, "CUDA dcn forward");
// m.def("dcn_forward_l1024_c2", &dcn_forward<1024, 2, 256>, "CUDA dcn forward");
// m.def("dcn_forward_l1024_c4", &dcn_forward<1024, 4, 256>, "CUDA dcn forward");
// m.def("dcn_forward_l1024_c8", &dcn_forward<1024, 8, 256>, "CUDA dcn forward");
// m.def("dcn_forward_l1024_c12", &dcn_forward<1024, 12, 256>, "CUDA dcn forward");
}
