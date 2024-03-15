/*
3D IoU Calculation and Rotated NMS(modified from 2D NMS written by others)
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include <cstdint>
#include <vector>

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

#define CHECK_ERROR(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}

const int THREADS_PER_BLOCK_NMS = sizeof(unsigned long long) * 8;


void boxesoverlapbevLauncher(const int num_a, const float *boxes_a, const int num_b, const float *boxes_b, float *ans_overlap);
void boxesoverlap3dLauncher(const int num_a, const float *boxes_a, const int num_b, const float *boxes_b, float *ans_overlap);
void boxesioubevLauncher(const int num_a, const float *boxes_a, const int num_b, const float *boxes_b, float *ans_iou);
void boxesiou3dLauncher(const int batch_size, const int num_a, const float *boxes_a, const int num_b, const float *boxes_b, float *ans_iou);
void boxesiou3dalignedLauncher(const int num_a, const float *boxes_a, const float *boxes_b, float *ans_iou);
void nms3dLauncher(const float *boxes, unsigned long long * mask, int boxes_num, float nms_overlap_thresh);
void nmsbevLauncher(const float *boxes, unsigned long long * mask, int boxes_num, float nms_overlap_thresh);
void nmsdistLauncher(const float *boxes, unsigned long long * mask, int boxes_num, float nms_overlap_thresh);

int boxes_overlap_bev_gpu(at::Tensor boxes_a, at::Tensor boxes_b, at::Tensor ans_overlap, int device_id){
    // params boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
    // params boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]
    // params ans_overlap: (N, M)

    CHECK_INPUT(boxes_a);
    CHECK_INPUT(boxes_b);
    CHECK_INPUT(ans_overlap);
    cudaSetDevice(device_id);
    int num_a = boxes_a.size(0);
    int num_b = boxes_b.size(0);

    const float * boxes_a_data = boxes_a.data_ptr<float>();
    const float * boxes_b_data = boxes_b.data_ptr<float>();
    float * ans_overlap_data = ans_overlap.data_ptr<float>();

    boxesoverlapbevLauncher(num_a, boxes_a_data, num_b, boxes_b_data, ans_overlap_data);

    return 1;
}

int boxes_overlap_3d_gpu(at::Tensor boxes_a, at::Tensor boxes_b, at::Tensor ans_overlap, int device_id){
    // params boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
    // params boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]
    // params ans_overlap: (N, M)

    CHECK_INPUT(boxes_a);
    CHECK_INPUT(boxes_b);
    CHECK_INPUT(ans_overlap);
    cudaSetDevice(device_id);
    int num_a = boxes_a.size(0);
    int num_b = boxes_b.size(0);

    const float * boxes_a_data = boxes_a.data_ptr<float>();
    const float * boxes_b_data = boxes_b.data_ptr<float>();
    float * ans_overlap_data = ans_overlap.data_ptr<float>();

    boxesoverlap3dLauncher(num_a, boxes_a_data, num_b, boxes_b_data, ans_overlap_data);

    return 1;
}

int boxes_iou_bev_gpu(at::Tensor boxes_a, at::Tensor boxes_b, at::Tensor ans_iou, int device_id){
    // params boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
    // params boxes_b: (M, 7) [x, y, z, dx, dy, dz, heading]
    // params ans_overlap: (N, M)
    CHECK_INPUT(boxes_a);
    CHECK_INPUT(boxes_b);
    CHECK_INPUT(ans_iou);
    cudaSetDevice(device_id);
    int num_a = boxes_a.size(0);
    int num_b = boxes_b.size(0);

    const float * boxes_a_data = boxes_a.data_ptr<float>();
    const float * boxes_b_data = boxes_b.data_ptr<float>();
    float * ans_iou_data = ans_iou.data_ptr<float>();

    boxesioubevLauncher(num_a, boxes_a_data, num_b, boxes_b_data, ans_iou_data);

    return 1;
}

int boxes_iou_3d_gpu(at::Tensor boxes_a, at::Tensor boxes_b, at::Tensor ans_iou, int device_id){
    // params boxes_a: (B, N, 7) [x, y, z, dx, dy, dz, heading]
    // params boxes_b: (B, M, 7) [x, y, z, dx, dy, dz, heading]
    // params ans_overlap: (B, N, M)
    CHECK_INPUT(boxes_a);
    CHECK_INPUT(boxes_b);
    CHECK_INPUT(ans_iou);
    cudaSetDevice(device_id);
    int batch_size = boxes_a.size(0);
    int num_a = boxes_a.size(1);
    int num_b = boxes_b.size(1);

    const float * boxes_a_data = boxes_a.data_ptr<float>();
    const float * boxes_b_data = boxes_b.data_ptr<float>();
    float * ans_iou_data = ans_iou.data_ptr<float>();

    boxesiou3dLauncher(batch_size, num_a, boxes_a_data, num_b, boxes_b_data, ans_iou_data);

    return 1;
}

int boxes_iou_3d_aligned_gpu(at::Tensor boxes_a, at::Tensor boxes_b, at::Tensor ans_iou, int device_id){
    // params boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
    // params boxes_b: (N, 7) [x, y, z, dx, dy, dz, heading]
    // params ans_overlap: (N, )
    CHECK_INPUT(boxes_a);
    CHECK_INPUT(boxes_b);
    CHECK_INPUT(ans_iou);
    cudaSetDevice(device_id);
    int num_a = boxes_a.size(0);

    const float * boxes_a_data = boxes_a.data_ptr<float>();
    const float * boxes_b_data = boxes_b.data_ptr<float>();
    float * ans_iou_data = ans_iou.data_ptr<float>();

    boxesiou3dalignedLauncher(num_a, boxes_a_data, boxes_b_data, ans_iou_data);

    return 1;
}

int nms_bev_gpu(at::Tensor boxes, at::Tensor keep, float nms_overlap_thresh, int device_id){
    // params boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    // params keep: (N)
    CHECK_INPUT(boxes);
    CHECK_CONTIGUOUS(keep);
    cudaSetDevice(device_id);

    int boxes_num = boxes.size(0);
    const float * boxes_data = boxes.data_ptr<float>();
    int64_t * keep_data = keep.data_ptr<int64_t>();

    const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);

    unsigned long long *mask_data = NULL;
    CHECK_ERROR(cudaMalloc((void**)&mask_data, boxes_num * col_blocks * sizeof(unsigned long long)));
    nmsbevLauncher(boxes_data, mask_data, boxes_num, nms_overlap_thresh);

    // unsigned long long mask_cpu[boxes_num * col_blocks];
    // unsigned long long *mask_cpu = new unsigned long long [boxes_num * col_blocks];
    std::vector<unsigned long long> mask_cpu(boxes_num * col_blocks);

//    printf("boxes_num=%d, col_blocks=%d\n", boxes_num, col_blocks);
    CHECK_ERROR(cudaMemcpy(&mask_cpu[0], mask_data, boxes_num * col_blocks * sizeof(unsigned long long),
                           cudaMemcpyDeviceToHost));

    cudaFree(mask_data);

    unsigned long long *remv_cpu = new unsigned long long[col_blocks]();

    int num_to_keep = 0;

    for (int i = 0; i < boxes_num; i++){
        int nblock = i / THREADS_PER_BLOCK_NMS;
        int inblock = i % THREADS_PER_BLOCK_NMS;

        if (!(remv_cpu[nblock] & (1ULL << inblock))){
            keep_data[num_to_keep++] = i;
            unsigned long long *p = &mask_cpu[0] + i * col_blocks;
            for (int j = nblock; j < col_blocks; j++){
                remv_cpu[j] |= p[j];
            }
        }
    }
    delete[] remv_cpu;
    if ( cudaSuccess != cudaGetLastError() ) printf( "Error!\n" );

    return num_to_keep;
}

int nms_3d_gpu(at::Tensor boxes, at::Tensor keep, float nms_overlap_thresh, int device_id){
    // params boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    // params keep: (N)
    CHECK_INPUT(boxes);
    CHECK_CONTIGUOUS(keep);
    cudaSetDevice(device_id);

    int boxes_num = boxes.size(0);
    const float * boxes_data = boxes.data_ptr<float>();
    int64_t * keep_data = keep.data_ptr<int64_t>();

    const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);

    unsigned long long *mask_data = NULL;
    CHECK_ERROR(cudaMalloc((void**)&mask_data, boxes_num * col_blocks * sizeof(unsigned long long)));
    nms3dLauncher(boxes_data, mask_data, boxes_num, nms_overlap_thresh);

    // unsigned long long mask_cpu[boxes_num * col_blocks];
    // unsigned long long *mask_cpu = new unsigned long long [boxes_num * col_blocks];
    std::vector<unsigned long long> mask_cpu(boxes_num * col_blocks);

//    printf("boxes_num=%d, col_blocks=%d\n", boxes_num, col_blocks);
    CHECK_ERROR(cudaMemcpy(&mask_cpu[0], mask_data, boxes_num * col_blocks * sizeof(unsigned long long),
                           cudaMemcpyDeviceToHost));

    cudaFree(mask_data);

    unsigned long long *remv_cpu = new unsigned long long[col_blocks]();

    int num_to_keep = 0;

    for (int i = 0; i < boxes_num; i++){
        int nblock = i / THREADS_PER_BLOCK_NMS;
        int inblock = i % THREADS_PER_BLOCK_NMS;

        if (!(remv_cpu[nblock] & (1ULL << inblock))){
            keep_data[num_to_keep++] = i;
            unsigned long long *p = &mask_cpu[0] + i * col_blocks;
            for (int j = nblock; j < col_blocks; j++){
                remv_cpu[j] |= p[j];
            }
        }
    }
    delete[] remv_cpu;
    if ( cudaSuccess != cudaGetLastError() ) printf( "Error!\n" );

    return num_to_keep;
}


int nms_dist_gpu(at::Tensor boxes, at::Tensor keep, float nms_overlap_thresh, int device_id){
    // params boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    // params keep: (N)
    CHECK_INPUT(boxes);
    CHECK_CONTIGUOUS(keep);
    cudaSetDevice(device_id);

    int boxes_num = boxes.size(0);
    const float * boxes_data = boxes.data_ptr<float>();
    int64_t * keep_data = keep.data_ptr<int64_t>();

    const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);

    unsigned long long *mask_data = NULL;
    CHECK_ERROR(cudaMalloc((void**)&mask_data, boxes_num * col_blocks * sizeof(unsigned long long)));
    nmsdistLauncher(boxes_data, mask_data, boxes_num, nms_overlap_thresh);

    // unsigned long long mask_cpu[boxes_num * col_blocks];
    // unsigned long long *mask_cpu = new unsigned long long [boxes_num * col_blocks];
    std::vector<unsigned long long> mask_cpu(boxes_num * col_blocks);

//    printf("boxes_num=%d, col_blocks=%d\n", boxes_num, col_blocks);
    CHECK_ERROR(cudaMemcpy(&mask_cpu[0], mask_data, boxes_num * col_blocks * sizeof(unsigned long long),
                           cudaMemcpyDeviceToHost));

    cudaFree(mask_data);

    unsigned long long *remv_cpu = new unsigned long long[col_blocks]();

    int num_to_keep = 0;

    for (int i = 0; i < boxes_num; i++){
        int nblock = i / THREADS_PER_BLOCK_NMS;
        int inblock = i % THREADS_PER_BLOCK_NMS;

        if (!(remv_cpu[nblock] & (1ULL << inblock))){
            keep_data[num_to_keep++] = i;
            unsigned long long *p = &mask_cpu[0] + i * col_blocks;
            for (int j = nblock; j < col_blocks; j++){
                remv_cpu[j] |= p[j];
            }
        }
    }
    delete[] remv_cpu;
    if ( cudaSuccess != cudaGetLastError() ) printf( "Error!\n" );

    return num_to_keep;
}