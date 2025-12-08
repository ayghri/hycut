// #include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>

template <typename scalar_t>
__global__ void hyper2f1_lse_kernel(
    const scalar_t *__restrict__ z,
    const scalar_t *__restrict__ log_binom,
    const scalar_t *__restrict__ log_r,
    long m, long K,
    scalar_t *__restrict__ out)
{

    long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= K)
        return;

    scalar_t zi = z[i];
    scalar_t logz = log(zi);    // ok if -inf at zi=0 (guarded below)
    scalar_t logq = log1p(-zi); // ok if -inf at zi=1

    scalar_t max_log = -CUDART_INF;
    for (long k = 0; k <= m; ++k)
    {
        scalar_t lw = 0;
        if (k > 0)
            lw += k * logz; // uniform branches => no divergence
        if (k < m)
            lw += (m - k) * logq;
        scalar_t logt = log_binom[k] + log_r[k] + lw;
        max_log = fmax(max_log, logt);
    }

    scalar_t acc = 0;
    for (long k = 0; k <= m; ++k)
    {
        scalar_t lw = 0;
        if (k > 0)
            lw += k * logz;
        if (k < m)
            lw += (m - k) * logq;
        scalar_t logt = log_binom[k] + log_r[k] + lw;
        acc += exp(logt - max_log);
    }

    out[i] = exp(max_log) * acc;
}

// ---- C++-callable typed launchers (no at::Tensor here) ----
extern "C" void hyper2f1_launch_float(
    const float *z, const float *log_binom, const float *log_r,
    long m, long K, float *out, cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid((K + block.x - 1) / block.x);
    hyper2f1_lse_kernel<float><<<grid, block, 0, stream>>>(z, log_binom, log_r, m, K, out);
}

extern "C" void hyper2f1_launch_double(
    const double *z, const double *log_binom, const double *log_r,
    long m, long K, double *out, cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid((K + block.x - 1) / block.x);
    hyper2f1_lse_kernel<double><<<grid, block, 0, stream>>>(z, log_binom, log_r, m, K, out);
}