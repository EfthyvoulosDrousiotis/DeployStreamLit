import numpy as np
from discretesampling.base.algorithms.smc_components.importance_resampling_version3 import systematic_resampling
# from filtering.base.algorithms.smc_components.resampling import systematic_resampling

def residual_resampling(x, w, mvrs_rng , N=None):

    if N is None:
        N = len(w)

    x_new = []
    w_new = np.zeros_like(w)

    if np.sum(w) != 1:
        w = w / np.sum(w)

    nd = np.floor(N * w).astype(int)  # nd(i) copies of x(i) to the new distribution
    urw = N*w - nd  # unnormalised residual weights
    nw = urw / np.sum(urw)  # normalised weights

    # i = 0
    #
    # for j in range(len(nd)):
    #     for k in range(nd[j]):
    #         i += 1
    #         x_new[i-1] = x[j]
    x_new = []
    for j in range(len(nd)):
        for k in range(nd[j]):
            x_new.append(x[j])

    Nnd = np.sum(nd)  # number of deterministic samples
    Nr = N - Nnd  # number of residual samples

    xr, _, _ = systematic_resampling(x, nw, mvrs_rng, Nr)
    # xr, _ = systematic_resampling(x, np.log(nw), np.log(w), mvrs_rng, parallel=False)

    # x_new[Nnd:] = xr[ :Nr]
    x_new.extend(xr)

    # w_new[:Nnd] = 1.0 / N
    # w_new[Nnd:] = nw[:Nr] * N / Nr
    w_new[:] = 1.0 / N

    log_w_new = np.log(w_new)

    return x_new, log_w_new

