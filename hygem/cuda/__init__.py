import os
import torch
from torch.utils.cpp_extension import load

# Optional: cache build artifacts inside your repo for reproducible builds
os.environ.setdefault("TORCH_EXTENSIONS_DIR", os.path.join(os.path.dirname(__file__), ".torch_extensions"))

_src_dir = os.path.dirname(__file__)
_ext = load(
    name="hyper2f1_cuda",
    sources=[os.path.join(_src_dir, "hyper2f1_bindings.cpp"),
             os.path.join(_src_dir, "hyper2f1_kernel.cu")],
    verbose=False,
    extra_cuda_cflags=[
        # add your arch list here or set TORCH_CUDA_ARCH_LIST in the env
        # e.g., "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__"
    ],
)

@torch.no_grad()
def _build_tables(m: int, b: float, c: float, device, dtype):
    k = torch.arange(m + 1, device=device, dtype=dtype)
    lg_m1 = torch.lgamma(torch.tensor(m + 1, device=device, dtype=dtype))
    log_binom = lg_m1 - torch.lgamma(k + 1) - torch.lgamma(torch.tensor(m, device=device, dtype=dtype) - k + 1)
    cb = torch.tensor(c - b, device=device, dtype=dtype)
    c_t = torch.tensor(c, device=device, dtype=dtype)
    log_r = (torch.lgamma(cb + k) - torch.lgamma(cb)
             + torch.lgamma(c_t) - torch.lgamma(c_t + k))
    return log_binom, log_r

@torch.no_grad()
def hyper2f1_negint(z: torch.Tensor, a_int: int, b: float, c: float) -> torch.Tensor:
    if a_int > 0:
        raise ValueError("'a_int' must be a non-positive integer")
    if z.dim() != 1 or not z.is_cuda:
        raise ValueError("z must be a 1D CUDA tensor")
    if not (b > 0 and c > 0 and c > b):
        raise ValueError("Require b>0, c>0, c>b")
    if torch.any((z < 0) | (z > 1)):
        raise ValueError("All z must be in [0,1]")
    m = -int(a_int)
    if m == 0:
        return torch.ones_like(z)
    log_binom, log_r = _build_tables(m, b, c, z.device, z.dtype)
    return _ext.forward(z.contiguous(), log_binom.contiguous(), log_r.contiguous(), m)