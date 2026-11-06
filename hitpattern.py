"""v1.2 (preregistered 2026-11-01): hit-pattern consistency requirement.
First hit = first sensor layer (mag.LAYERS) outside the TRUE vertex radius, per-layer hit efficiency 0.99,
pair first hit = inner of the two tracks' first hits. A pair is consistent if
R_prev - t*sigma_r < r_meas < R_first + t*sigma_r  (R_prev = previous sensor layer, 0 if none)."""
import numpy as np
import mag

EFF = 0.99
SEED = 23
L = mag.LAYERS


def first_hit_index(r_true, rng):
    """Index of the pair's first-hit layer (len(L) if none)."""
    n = len(r_true)
    idx_geo = np.searchsorted(L, r_true, side='right')        # first layer with R > r_true
    out = []
    for _ in range(2):                                         # two tracks
        hit = rng.uniform(0, 1, (n, len(L))) < EFF
        hit &= (np.arange(len(L))[None, :] >= idx_geo[:, None])
        any_hit = hit.any(1)
        out.append(np.where(any_hit, hit.argmax(1), len(L)))
    return np.minimum(out[0], out[1])


def consistent(r_true, r_meas, sigma_r, t=1.0, seed=SEED):
    rng = np.random.default_rng(seed)
    k = first_hit_index(r_true, rng)
    ok = k < len(L)
    kk = np.minimum(k, len(L) - 1)
    R_first = L[kk]
    R_prev = np.where(kk > 0, L[np.maximum(kk - 1, 0)], 0.0)
    return ok & (r_meas > R_prev - t * sigma_r) & (r_meas < R_first + t * sigma_r)
