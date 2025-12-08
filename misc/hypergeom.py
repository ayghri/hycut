"""Hypergeometric 2F1 implementation for negative integer a."""

from math import lgamma, exp, fabs, sqrt, copysign, isfinite
from typing import Union
import numpy as np


# _PFAFF_KUMMER_THRESHOLD = (3.0 - sqrt(5.0)) / 2.0  # ~0.381966
_PFAFF_KUMMER_THRESHOLD = 0.3
_EPS = 1e-14


def _scalar_hyper2f1(a_int: int, b: float, c: float, z: float) -> float:
    if a_int > 0:
        raise ValueError("'a_int' must be a non-positive integer")
    if a_int == 0:
        return 1.0
    if not (0.0 <= z <= 1.0):
        raise ValueError("z must be in [0,1]")
    if b <= 0 or c <= 0 or c <= b:
        raise ValueError("Require b>0, c>0, c>b")

    if fabs(z) < _EPS:
        return 1.0

    m = float(-a_int)  # m >= 1
    if fabs(z - 1.0) < _EPS:
        # Closed form at z=1 via gamma functions (ratio form) 2F1(-m,b;c;1)
        log_gamma_c = lgamma(c)
        log_gamma_c_minus_b_plus_m = lgamma(c - b + m)
        log_gamma_c_plus_m = lgamma(c + m)
        log_gamma_c_minus_b = lgamma(c - b)
        return exp(
            log_gamma_c
            + log_gamma_c_minus_b_plus_m
            - log_gamma_c_plus_m
            - log_gamma_c_minus_b
        )

    a_d = float(a_int)
    sum_val = 1.0
    term = 1.0

    if z < _PFAFF_KUMMER_THRESHOLD:
        c_minus_b = c - b
        z_pfaff = z / (z - 1.0)
        for k in range(int(m)):
            k_val = float(k)
            term *= (a_d + k_val) * (c_minus_b + k_val) * z_pfaff
            denom = (c + k_val) * (k_val + 1.0)
            term /= denom
            sum_val += term
        return (1.0 - z) ** m * sum_val
    else:
        C_prime = a_d + b - c + 1.0
        z_kummer = 1.0 - z
        for k in range(int(m)):
            k_val = float(k)
            num_factor = (a_d + k_val) * (b + k_val) * z_kummer
            den_factor = (C_prime + k_val) * (k_val + 1.0)
            if fabs(den_factor) < _EPS * _EPS:
                if fabs(num_factor) < _EPS * _EPS:
                    term = 0.0
                else:
                    term = copysign(float("inf"), num_factor / den_factor)
            else:
                term *= num_factor / den_factor
            sum_val += term
            if (not isfinite(term)) and term != 0.0:
                break

        log_gamma_c_minus_b_plus_m = lgamma(c - b + m)
        log_gamma_c = lgamma(c)
        log_gamma_c_minus_b = lgamma(c - b)
        log_gamma_c_plus_m = lgamma(c + m)
        prefactor = exp(
            log_gamma_c_minus_b_plus_m
            + log_gamma_c
            - log_gamma_c_minus_b
            - log_gamma_c_plus_m
        )
        return prefactor * sum_val


def hypergeometric_2F1_stable(
    a_int: int, b: float, c: float, z: Union[float, int, np.ndarray]
):
    """Compute 2F1(a,b;c;z) for a negative integer a (terminating series).

    Supports scalar z or numpy array z; in the array case returns an array
    of matching shape.
    """
    # Restrict scalars to int/float; other dtypes (str/complex) raise early
    if isinstance(z, (int, float)):
        return _scalar_hyper2f1(a_int, b, c, float(z))
    z_arr = np.asarray(z, dtype=float)
    out = np.empty_like(z_arr)
    it = np.nditer(z_arr, flags=["multi_index"])
    while not it.finished:
        out[it.multi_index] = _scalar_hyper2f1(a_int, b, c, float(it[0]))
        it.iternext()
    return out


# print(hypergeometric_2F1_stable(-2048, 1.0, 1.5, 0.30403469))


x = np.linspace(0,1, 1000)