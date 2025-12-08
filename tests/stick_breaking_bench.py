#!/usr/bin/env python3

import argparse, time, math, statistics, os, sys
from typing import List, Tuple
import torch
import torch.nn.functional as F

try:
    import pandas as pd
except Exception:
    pd = None

# ---------------------- Stick-breaking implementations ----------------------

def stick_breaking_logits_to_probs_log(Y: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Stable log-domain stick-breaking: Y (..., K-1) -> P (..., K)."""
    Km1 = Y.size(dim)
    if Km1 < 1:
        raise ValueError("Y must have size >= 1 along `dim`.")
    log_v  = F.logsigmoid(Y)    # log sigma(Y)
    log1mv = F.logsigmoid(-Y)   # log(1 - sigma(Y)) = log sigma(-Y)
    S = torch.cumsum(log1mv, dim=dim)
    # Build exclusive prefix of S by padding a zero then dropping the last elem
    zeros_shape = list(Y.shape)
    zeros_shape[dim] = 1
    zeros = torch.zeros(zeros_shape, dtype=Y.dtype, device=Y.device)
    # helper slices
    idx = [slice(None)] * Y.ndim
    idx_last = idx.copy()
    idx_last[dim] = slice(Km1-1, Km1)
    idx_up_to = idx.copy()
    idx_up_to[dim] = slice(0, Km1-1)
    prefix = torch.cat([zeros, S[tuple(idx_up_to)]], dim=dim)
    log_p_except_last = log_v + prefix
    log_p_last = S[tuple(idx_last)]
    logP = torch.cat([log_p_except_last, log_p_last], dim=dim)
    return torch.exp(logP)


def stick_breaking_logits_to_probs_naive(Y: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Naive domain: v = sigmoid(Y); P_j = v_j * prod_{i<j}(1-v_i); last = prod_i (1-v_i)."""
    v = torch.sigmoid(Y)
    one_minus_v = 1 - v
    Km1 = Y.size(dim)
    if Km1 < 1:
        raise ValueError("Y must have size >= 1 along `dim`.")
    # Exclusive cumprod of (1-v)
    # prefix[j] = prod_{i<j} (1 - v_i), with prefix[0] = 1
    ones_shape = list(v.shape); ones_shape[dim] = 1
    ones = torch.ones(ones_shape, dtype=v.dtype, device=v.device)
    # cumprod inclusive, then shift-right with leading 1
    cum = torch.cumprod(one_minus_v, dim=dim)
    idx = [slice(None)] * v.ndim
    idx_up_to = idx.copy(); idx_up_to[dim] = slice(0, Km1-1)
    prefix = torch.cat([ones, cum[tuple(idx_up_to)]], dim=dim)
    p_except_last = v * prefix
    # p_last = prod_i (1 - v_i) = last element of cum
    idx_last = idx.copy(); idx_last[dim] = slice(Km1-1, Km1)
    p_last = cum[tuple(idx_last)]
    P = torch.cat([p_except_last, p_last], dim=dim)
    return P


# ---------------------- Utilities ----------------------

def make_sizes(spec: str) -> List[Tuple[int,int]]:
    # spec like "32x64,256x64,1024x256,4096x1024"
    out = []
    for token in spec.split(","):
        token = token.strip().lower().replace(" ", "")
        if not token:
            continue
        if "x" in token:
            b,k = token.split("x")
        elif "," in token:
            b,k = token.split(",")
        else:
            raise ValueError(f"Bad size token: {token}")
        out.append((int(b), int(k)))
    return out


def time_it(fn, sync, iters: int) -> float:
    start = time.perf_counter()
    for _ in range(iters):
        fn()
        sync()
    end = time.perf_counter()
    return (end - start) * 1000.0 / iters  # ms/iter


def peak_cuda_mem_bytes() -> int:
    if not torch.cuda.is_available():
        return 0
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    torch.cuda.synchronize()
    return peak


def benchmark_case(B: int, K: int, device: torch.device, dtype: torch.dtype, iters: int, warmup: int):
    results = []
    dim = -1
    Km1 = K - 1
    sync = (torch.cuda.synchronize if device.type == "cuda" else (lambda: None))

    # Data
    Y = torch.randn(B, Km1, device=device, dtype=dtype, requires_grad=True)
    target = torch.rand(B, K, device=device, dtype=dtype)
    target = target / target.sum(dim=dim, keepdim=True)

    def loss_fn(P):
        # Simple stable loss that exercises backward paths
        return (P - target).square().sum()

    impls = {
        "log_domain": stick_breaking_logits_to_probs_log,
        "naive":       stick_breaking_logits_to_probs_naive,
    }

    for name, impl in impls.items():
        # Warmup
        for _ in range(warmup):
            P = impl(Y)
            l = loss_fn(P)
            l.backward()
            if Y.grad is not None:
                Y.grad.zero_()
            sync()

        # Forward timing (no grad)
        def fwd():
            with torch.no_grad():
                P = impl(Y)

        fwd_ms = time_it(fwd, sync, iters)

        # Forward+Backward timing
        def fwd_bwd():
            P = impl(Y)
            l = loss_fn(P)
            l.backward()
            Y.grad.zero_()

        fwd_bwd_ms = time_it(fwd_bwd, sync, iters)

        # Numerical quality vs log_domain (use log_domain as reference)
        with torch.no_grad():
            Pref = stick_breaking_logits_to_probs_log(Y)
            Pcur = impl(Y)
            diff = (Pref - Pcur).abs()
            linf = float(diff.max().item())
            l2 = float(torch.linalg.vector_norm(diff).item())
            n_nans = int(torch.isnan(Pcur).sum().item())
            n_infs = int(torch.isinf(Pcur).sum().item())

        # CUDA peak memory (approx per pass)
        peak_mem = 0
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            # Run an instance of forward+backward to record peak
            P = impl(Y)
            l = loss_fn(P); l.backward()
            peak_mem = int(torch.cuda.max_memory_allocated())
            # cleanup
            if Y.grad is not None:
                Y.grad.zero_()
            del P, l
            sync()

        results.append({
            "impl": name,
            "B": B,
            "K": K,
            "device": device.type,
            "dtype": str(dtype).replace("torch.", ""),
            "fwd_ms": round(fwd_ms, 4),
            "fwd_bwd_ms": round(fwd_bwd_ms, 4),
            "linf_err_vs_log": linf,
            "l2_err_vs_log": l2,
            "n_nans": n_nans,
            "n_infs": n_infs,
            "peak_mem_bytes": peak_mem
        })

    return results


def main():
    ap = argparse.ArgumentParser(description="Benchmark stick-breaking parameterization implementations.")
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"], help="Device to run on.")
    ap.add_argument("--dtype", default="float32", choices=["float32","float16","bfloat16"], help="Tensor dtype.")
    ap.add_argument("--iters", type=int, default=100, help="Timing iterations per case.")
    ap.add_argument("--warmup", type=int, default=50, help="Warmup iterations per case.")
    ap.add_argument("--sizes", default="32x64,256x64,1024x256,4096x512", help="Comma-separated BxK cases.")
    ap.add_argument("--csv", type=str, default="", help="Optional path to save CSV summary.")
    args = ap.parse_args()

    if args.device == "auto":
        use_cuda = torch.cuda.is_available()
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            print("CUDA not available; falling back to CPU.", file=sys.stderr)
            use_cuda = False
        else:
            use_cuda = True
    else:
        use_cuda = False

    device = torch.device("cuda" if use_cuda else "cpu")
    dtypemap = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    dtype = dtypemap[args.dtype]

    sizes = make_sizes(args.sizes)

    all_rows = []
    for (B, K) in sizes:
        if K < 2:
            print(f"Skipping K={K}; needs at least 2.", file=sys.stderr)
            continue
        rows = benchmark_case(B, K, device, dtype, args.iters, args.warmup)
        all_rows.extend(rows)

    # Pretty print
    def fmt_row(r):
        kb = r["peak_mem_bytes"]/1024.0
        return (f"{r['impl']:>11} | B={r['B']:>6} K={r['K']:>6} | "
                f"{r['device']:>4} {r['dtype']:>8} | "
                f"fwd={r['fwd_ms']:>8.3f} ms | fwd+bwd={r['fwd_bwd_ms']:>8.3f} ms | "
                f"linf={r['linf_err_vs_log']:.3e} | nans={r['n_nans']}, infs={r['n_infs']} | "
                f"peak_mem={kb:,.1f} KiB")

    print("\n=== Stick-breaking Benchmark Results ===\n")
    for r in all_rows:
        print(fmt_row(r))

    # Optional CSV
    if args.csv:
        import pandas as pd  # in case it was lazy-imported
        df = pd.DataFrame(all_rows)
        df.to_csv(args.csv, index=False)
        print(f"\nSaved CSV to {args.csv}")


if __name__ == "__main__":
    main()
