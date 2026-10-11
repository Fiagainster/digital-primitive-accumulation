"""Average-rank Spearman and exact, two-sided index-permutation tests.

The inferential observations are configuration means, never seed repetitions.
For n=6, all 6! index permutations are included, including the observed order.
Tied observations retain multiplicity. No Monte Carlo or add-one adjustment is used.
"""
from itertools import permutations
import math
import numpy as np
from scipy.stats import rankdata, t as student_t


def spearman_average(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 3:
        raise ValueError("Expected equal one-dimensional samples, n >= 3")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("Non-finite observations")
    rx, ry = rankdata(x, method="average"), rankdata(y, method="average")
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        raise ValueError("Spearman is undefined for a constant sample")
    rho = float(np.corrcoef(rx, ry)[0, 1])
    p = (0.0 if abs(rho) >= 1 else
         float(2 * student_t.sf(abs(rho) * math.sqrt((len(x)-2)/(1-rho*rho)), len(x)-2)))
    return rho, p


def exact_spearman(x, y, tolerance=1e-12):
    if len(x) > 8:
        raise ValueError("Exact enumeration is restricted to n <= 8")
    rho, _ = spearman_average(x, y)
    rx = rankdata(np.asarray(x, float), method="average")
    ry = rankdata(np.asarray(y, float), method="average")
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denominator = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    null = [float(np.dot(rx, ry[list(idx)]) / denominator)
            for idx in permutations(range(len(x)))]
    extreme = sum(abs(v) >= abs(rho) - tolerance for v in null)
    return {"rho": rho, "p_exact_two_sided": extreme / len(null),
            "n_configurations": len(x), "n_index_permutations": len(null),
            "extreme_count": extreme, "absolute_rho_tolerance": tolerance,
            "rank_method": "average", "null_rho": null,
            "tail_rule": "abs(rho_permuted) >= abs(rho_observed) - tolerance",
            "unit": "country-type configuration mean over five seeded repetitions",
            "x": [float(v) for v in x], "y": [float(v) for v in y]}
