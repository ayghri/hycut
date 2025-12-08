// hyper2f1_bindings.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h> // getCurrentCUDAStream
#include <c10/cuda/CUDAGuard.h>    // OptionalCUDAGuard
#include <cuda_runtime.h>

// Declarations of the typed launchers from the .cu TU
extern "C" void hyper2f1_launch_float(
    const float *z, const float *log_binom, const float *log_r,
    long m, long K, float *out, cudaStream_t stream);
extern "C" void hyper2f1_launch_double(
    const double *z, const double *log_binom, const double *log_r,
    long m, long K, double *out, cudaStream_t stream);

torch::Tensor hyper2f1_forward_cuda(
    torch::Tensor z,         // (K,) CUDA
    torch::Tensor log_binom, // (m+1,) CUDA
    torch::Tensor log_r,     // (m+1,) CUDA
    long m)
{
    TORCH_CHECK(z.is_cuda(), "z must be CUDA");
    TORCH_CHECK(z.dim() == 1, "z must be (K,)");
    TORCH_CHECK(log_binom.is_cuda() && log_r.is_cuda(), "tables must be CUDA");
    TORCH_CHECK(log_binom.numel() == m + 1 && log_r.numel() == m + 1, "tables length m+1");
    TORCH_CHECK(z.scalar_type() == log_binom.scalar_type() && z.scalar_type() == log_r.scalar_type(),
                "dtype mismatch");

    const c10::cuda::OptionalCUDAGuard guard(z.device());
    auto out = torch::empty_like(z);
    const auto K = z.size(0);
    auto stream = at::cuda::getCurrentCUDAStream();

    if (z.scalar_type() == at::kFloat)
    {
        hyper2f1_launch_float(
            z.data_ptr<float>(),
            log_binom.data_ptr<float>(),
            log_r.data_ptr<float>(),
            m, K, out.data_ptr<float>(),
            stream.stream());
    }
    else if (z.scalar_type() == at::kDouble)
    {
        hyper2f1_launch_double(
            z.data_ptr<double>(),
            log_binom.data_ptr<double>(),
            log_r.data_ptr<double>(),
            m, K, out.data_ptr<double>(),
            stream.stream());
    }
    else
    {
        TORCH_CHECK(false, "Only float32/float64 supported");
    }

    // Launch check
    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ",
                static_cast<int>(err), " (", cudaGetErrorString(err), ")");

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &hyper2f1_forward_cuda,
          "2F1(-m,b;c;z) (binomial/log-sum-exp, CUDA)");
}