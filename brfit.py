"""v1.4 (preregistered 2026-11-02): bremsstrahlung-aware vertex fits on top of the v1.3 mini simulation.

(a) inner-hit fit : each track keeps only its first N measured hits (default N = 6); v1.3 mode-B fit otherwise.
(b) brems-noise fit: single-Gaussian approximation of a Gaussian-sum filter.
    - track model with the MEAN energy loss: after each material surface crossed after the vertex the curvature is
      multiplied by exp(c_k), c_k = t_k / ln 2, t_k = t0 cosh(eta)   (<-ln z> = c_k for Bethe-Heitler)
    - brems NOISE in the covariance: d(ln omega) with variance c_k after each surface, propagated to later hits with
      numerically computed lever arms, added to the v1.3 hit + multiple-scattering covariance.
    The list of surfaces after the vertex, cosh(eta) and the scattering momentum come from truth, as in v1.3.
All fits use the v1.3 mode-B framework (parallel constraint, same seeds, starting points and tie-breaks)."""
import numpy as np
import minitrack as mt

NL = mt.NL
# ordered union of surfaces (radius, sensor index or -1, X0); at equal radius: hit first, then material
SURF = sorted([(r, i, 0.0) for i, r in enumerate(mt.SENS)] + [(r, -1, t) for r, t in zip(mt.MAT_R, mt.MAT_X0)],
              key=lambda s: (s[0], s[1] < 0))
S_R = np.array([s[0] for s in SURF]); S_I = np.array([s[1] for s in SURF]); S_T = np.array([s[2] for s in SURF])


# ------------------------------------------------------------------------------------------------ (a)
def keep_inner(hits, n_keep=6):
    """Keep only the first n_keep measured hits of each track (innermost first)."""
    m = ~np.isnan(hits)
    rank = np.cumsum(m, 1)
    return np.where(m & (rank <= n_keep), hits, np.nan)


# ------------------------------------------------------------------------------------------------ (b)
def track_c(vtx, p3, brems_c=True):
    """(n, S) table of c_k for every surface in SURF: t0*cosh(eta)/ln2 for material surfaces beyond the TRUE vertex
    radius, 0 otherwise; plus the boolean 'active' mask (surface beyond the true vertex)."""
    pt = np.hypot(p3[:, 0], p3[:, 1]); ch = np.linalg.norm(p3, axis=1) / pt
    r0 = np.hypot(vtx[:, 0], vtx[:, 1])
    active = S_R[None, :] > r0[:, None]
    c = (S_T[None, :] * ch[:, None]) / np.log(2) * active * (S_I[None, :] < 0)
    return (c if brems_c else 0 * c), active


def propagate_b(x0, y0, phi0, om0, c, active, kick=None, eps=0.0):
    """Deterministic surface-by-surface propagation with mean energy loss. Returns hit azimuths (n, NL).
    kick = surface index whose outgoing ln(omega) is shifted by eps (for lever arms)."""
    n = len(x0)
    x, y, phi, om = x0.copy(), y0.copy(), phi0.copy(), om0.copy()
    hits = np.full((n, NL), np.nan)
    ok = np.ones(n, bool)
    for k in range(len(S_R)):
        act = active[:, k] & ok
        if not act.any():
            continue
        R = S_R[k]
        s = mt.to_radius(x, y, phi, om, R, 'nearest')
        s = np.where(np.abs(np.hypot(x, y) - R) < 1e-6, 0.0, s)
        bad = act & np.isnan(s)
        ok &= ~bad; act &= ~bad
        sx, sy, sp = mt.advance(x, y, phi, om, np.where(act, s, 0.0))
        x, y, phi = np.where(act, sx, x), np.where(act, sy, y), np.where(act, sp, phi)
        if S_I[k] >= 0:
            hits[:, S_I[k]] = np.where(act, np.arctan2(y, x), np.nan)
        else:
            f = np.exp(c[:, k])
            if kick is not None and k == kick:
                f = f * np.exp(eps)
            om = np.where(act, om * f, om)
    return hits


def make_pred_b(brems_c=True):
    """pred_fn for minitrack.vertex_fit: returns a predictor bound to the per-pair c table."""
    def pred_fn(vtx, p3, q):
        c, active = track_c(vtx, p3, brems_c)
        return lambda x0, y0, phi0, om0: propagate_b(x0, y0, phi0, om0, c, active)
    return pred_fn


def make_cov_b(brems_c=True):
    """cov_fn for minitrack.vertex_fit: v1.3 hit + scattering covariance plus Gaussian brems noise."""
    def cov_fn(vtx, p3, q):
        C = mt.hit_cov(vtx, p3, q)
        if not brems_c:
            return C
        c, active = track_c(vtx, p3, True)
        n = len(vtx)
        qq = np.broadcast_to(np.asarray(q, float), (n,))
        pt = np.hypot(p3[:, 0], p3[:, 1])
        x0, y0 = vtx[:, 0].copy(), vtx[:, 1].copy()
        phi0 = np.arctan2(p3[:, 1], p3[:, 0]); om0 = mt.omega_of(qq, pt)
        base = propagate_b(x0, y0, phi0, om0, c, active)
        eps = 1e-4
        for k in np.where((S_I < 0) & (c.max(0) > 0))[0]:
            h = propagate_b(x0, y0, phi0, om0, c, active, kick=k, eps=eps)
            lev = mt.SENS[None, :] * mt.wrap(h - base) / eps
            lev = np.where(np.isfinite(lev), lev, 0.0)
            C += c[:, k][:, None, None] * lev[:, :, None] * lev[:, None, :]
        return C
    return cov_fn


# ------------------------------------------------------------------------------------------------ driver
def simulate_pairs(vtx, e1, e2, seed, brems=True, chunk=20000):
    """Exactly the v1.3 simulation call sequence (same rng use per chunk) -> identical hits."""
    rng = np.random.default_rng(seed)
    n = len(vtx)
    H1, H2 = np.full((n, NL), np.nan), np.full((n, NL), np.nan)
    for s in range(0, n, chunk):
        sl = slice(s, min(n, s + chunk))
        H1[sl], _ = mt.simulate_track(vtx[sl], e1[sl], -1, rng, brems=brems)
        H2[sl], _ = mt.simulate_track(vtx[sl], e2[sl], +1, rng, brems=brems)
    return H1, H2


def fit_pairs(H1, H2, vtx, e1, e2, method, n_keep=6, chunk=20000):
    """method: 'B' (v1.3 mode B), 'a' (inner hits), 'b' (brems-noise fit), 'b0' (fit b with c = 0)."""
    n = len(vtx)
    res = {k: np.full(n, np.nan) for k in ('r', 'sigma_r', 'chi2', 'ndf', 'x', 'y', 'nbehind')}
    res['ok'] = np.zeros(n, bool)
    kw = dict(parallel=True)
    if method in ('b', 'b0'):
        kw.update(cov_fn=make_cov_b(method == 'b'), pred_fn=make_pred_b(method == 'b'))
    for s in range(0, n, chunk):
        sl = slice(s, min(n, s + chunk))
        h1, h2 = H1[sl], H2[sl]
        if method == 'a':
            h1, h2 = keep_inner(h1, n_keep), keep_inner(h2, n_keep)
        o = mt.vertex_fit(h1, h2, vtx[sl], e1[sl], e2[sl], **kw)
        for k in res:
            if k in o:
                res[k][sl] = o[k]
    return res
