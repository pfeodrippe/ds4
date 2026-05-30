// DS4 LoRA (Low-Rank Adaptation) kernel
// Applies: y += scale * B * A * x
//
// A: [rank x input_dim]  (down-projection)
// B: [output_dim x rank]  (up-projection)
// x: [input_dim]          (input vector)
// y: [output_dim]         (output vector, accumulated in-place)
//
// This is called after the base Q/K/V projection to add the adapter delta.

#include <metal_stdlib>
using namespace metal;

kernel void kernel_lora_matmul(
        device const float * A         [[buffer(0)]],
        device const float * B         [[buffer(1)]],
        device const float * x         [[buffer(2)]],
        device       float * y         [[buffer(3)]],
        constant   int     & rank      [[buffer(4)]],
        constant   int     & input_dim [[buffer(5)]],
        constant   int     & output_dim[[buffer(6)]],
        constant   float   & scale    [[buffer(7)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]]) {
    const int row = (int)(tgpig.x * ntg.x + tpitg.x);

    if (row >= output_dim) return;

    // Compute tmp = A * x (rank-dimensional), then y[row] += scale * B[row] * tmp
    // For small rank (8, 16, 32), we compute A*x inline per output row
    
    float sum = 0.0f;
    const device float *b_row = B + (int64_t)row * rank;

    for (int r = 0; r < rank; r++) {
        // dot(A[r], x)
        float a_dot_x = 0.0f;
        const device float *a_row = A + (int64_t)r * input_dim;
        for (int d = 0; d < input_dim; d++) {
            a_dot_x += a_row[d] * x[d];
        }
        sum += b_row[r] * a_dot_x;
    }

    y[row] += scale * sum;
}
