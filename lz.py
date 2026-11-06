"""v1.7 (preregistered, doi:10.5281/zenodo.23995668): longitudinal (r-z) simulation and two-hypothesis r-z fit.

Simulation : z = z_v + cot(theta) * s along the transverse path s of the transverse simulation (same hits as v1.5/v1.6);
             at every material surface after the vertex the polar angle gets a Highland kick with the momentum of the
             simulated track at that surface (bremsstrahlung included); hit smearing sigma_z per layer and geometry.
             Longitudinal random numbers come from a separate generator, drawn once and shared by all geometries.
Fit        : after each transverse hypothesis, s from the fitted vertex to each layer along the fitted track;
             linear least squares in r-z: H_conv,L (z_v, common slope) and H_ALP,L (z_v, c1, c2);
             covariance = sigma_z^2 + polar scattering (true path lever arms, true initial momentum)."""
import numpy as np
import minitrack as mt
import brfit as bf

SENS = mt.SENS
GEOM = {
    'default': np.where(SENS < 200, 0.025, np.inf),
    'stereo': np.where(SENS < 200, 0.025, np.inf),
    'allpixel': np.full(len(SENS), 0.025),
}
for r, sz in ((255, 0.25), (339, 0.25), (608, 0.5), (692, 0.5)):
    GEOM['stereo'][np.isclose(SENS, r)] = sz


def highland(p, t):
    return 0.0136 / p * np.sqrt(t) * (1 + 0.038 * np.log(t))


def sim_pairs(vtx, e1, e2, seed_T, seed_L, brems=True, chunk=20000):
    """Transverse hits exactly as brfit.simulate_pairs (same rng sequence) plus z hits for every geometry.
    Returns H1, H2, Z (dict geom -> (Z1, Z2)), T1, T2 (truth path info for the covariance)."""
    rng = np.random.default_rng(seed_T); rngL = np.random.default_rng(seed_L)
    n = len(vtx); NL = len(SENS)
    H = [np.full((n, NL), np.nan), np.full((n, NL), np.nan)]
    Z = {g: [np.full((n, NL), np.nan), np.full((n, NL), np.nan)] for g in GEOM}
    T = [dict(), dict()]
    for s0 in range(0, n, chunk):
        sl = slice(s0, min(n, s0 + chunk))
        infos = []
        for i, (e, q) in enumerate(((e1, -1), (e2, +1))):
            h, inf = mt.simulate_track(vtx[sl], e[sl], q, rng, brems=brems, return_path=True)
            H[i][sl] = h; infos.append(inf)
        for i, inf in enumerate(infos):
            m = sl.stop - sl.start
            kick = rngL.normal(0, 1, (m, inf['s_mat'].shape[1])); zeta = rngL.normal(0, 1, (m, NL))
            th0 = highland(inf['p_mat'], inf['t_mat'][None, :] * inf['cosh'][:, None])
            dcot = np.nan_to_num(kick * th0 * inf['cosh'][:, None]**2)          # d(cot theta) = d(theta) / sin^2(theta)
            s_k, s_m = inf['s_sens'], inf['s_mat']
            arm = np.clip(s_k[:, :, None] - np.nan_to_num(s_m, nan=1e9)[:, None, :], 0, None)
            ztrue = inf['z0'][:, None] + inf['cot'][:, None] * s_k + (arm * dcot[:, None, :]).sum(2)
            for g, sz in GEOM.items():
                zz = ztrue + zeta * np.where(np.isfinite(sz), sz, 0)[None, :]
                Z[g][i][sl] = np.where(np.isfinite(sz)[None, :] & ~np.isnan(H[i][sl]), zz, np.nan)
            for k in ('s_sens', 's_mat', 'cosh'):
                T[i].setdefault(k, []).append(inf[k])
    for i in range(2):
        T[i] = {k: np.concatenate(v) for k, v in T[i].items()}
        T[i]['t_mat'] = infos[0]['t_mat']
    return H[0], H[1], Z, T[0], T[1]


def zcov(T, p_true, sz):
    """(n, NL, NL) r-z covariance from the truth path: z resolution plus polar scattering at the true momentum."""
    s_k, s_m, ch = T['s_sens'], T['s_mat'], T['cosh']
    th0 = highland(p_true[:, None], T['t_mat'][None, :] * ch[:, None])
    v = (th0 * ch[:, None]**2)**2                                                 # var of d(cot theta)
    arm = np.clip(np.nan_to_num(s_k, nan=0)[:, :, None] - np.nan_to_num(s_m, nan=1e9)[:, None, :], 0, None)
    C = np.einsum('nkm,nlm,nm->nkl', arm, arm, np.nan_to_num(v))
    C[:, np.arange(len(sz)), np.arange(len(sz))] += np.where(np.isfinite(sz), sz, 1.0)**2
    return C


def fitted_s(par, i):
    """s from the fitted vertex to each sensor along fitted track i (i = 0, 1), nearest intersection (as predict())."""
    x, y, ph, om = par[:, 0], par[:, 1], par[:, 2 + 2 * i], par[:, 3 + 2 * i]
    return mt.to_radius(x[:, None], y[:, None], ph[:, None], om[:, None], SENS[None, :], 'nearest')


def fit_L(Z1, Z2, W1, W2, par):
    """Return chi2 for the common-slope (k=1) and free-slope (k=2) r-z fits with s from the transverse fit par."""
    s1, s2 = fitted_s(par, 0), fitted_s(par, 1)
    m1, m2 = ~np.isnan(Z1) & np.isfinite(s1), ~np.isnan(Z2) & np.isfinite(s2)
    ref = np.nanmedian(np.where(m1, Z1, np.nan), axis=1)                         # numerical reference only
    out = {}
    q = []
    for Zt, st, W, m in ((Z1, s1, W1, m1), (Z2, s2, W2, m2)):
        z = np.where(m, Zt - ref[:, None], 0.0); s = np.where(m, st, 0.0); u = m.astype(float)
        Wm = W * (m[:, :, None] & m[:, None, :])
        f = lambda a, b: np.einsum('nk,nkl,nl->n', a, Wm, b)
        q.append(dict(a=f(u, u), b=f(u, s), c=f(s, s), du=f(u, z), ds=f(s, z), zz=f(z, z), nh=m.sum(1)))
    A, Bq = q
    # common slope: params (z_v, c)
    M2 = np.stack([np.stack([A['a'] + Bq['a'], A['b'] + Bq['b']], -1), np.stack([A['b'] + Bq['b'], A['c'] + Bq['c']], -1)], -2)
    r2 = np.stack([A['du'] + Bq['du'], A['ds'] + Bq['ds']], -1)
    # free slopes: params (z_v, c1, c2)
    z0 = np.zeros_like(A['a'])
    M3 = np.stack([np.stack([A['a'] + Bq['a'], A['b'], Bq['b']], -1), np.stack([A['b'], A['c'], z0], -1),
                   np.stack([Bq['b'], z0, Bq['c']], -1)], -2)
    r3 = np.stack([A['du'] + Bq['du'], A['ds'], Bq['ds']], -1)
    zz = A['zz'] + Bq['zz']
    info = (A['nh'] >= 2) & (Bq['nh'] >= 2)
    with np.errstate(all='ignore'):
        th2 = np.linalg.solve(np.where(info[:, None, None], M2, np.eye(2)), np.where(info[:, None], r2, 0)[..., None])[..., 0]
        th3 = np.linalg.solve(np.where(info[:, None, None], M3, np.eye(3)), np.where(info[:, None], r3, 0)[..., None])[..., 0]
    out['chi2_1'] = zz - (th2 * r2).sum(1); out['chi2_2'] = zz - (th3 * r3).sum(1); out['info'] = info
    return out


def dchi2_L(Z1, Z2, T1, T2, e1, e2, parc, para, geom):
    """Delta chi2_L = chi2(H_conv,L with s from the H_conv transverse fit) - chi2(H_ALP,L with s from the H_ALP fit).
    0 for pairs without r-z information (each track needs >= 2 z hits)."""
    sz = GEOM[geom]
    p1, p2 = np.linalg.norm(e1, axis=1), np.linalg.norm(e2, axis=1)
    m1, m2 = ~np.isnan(Z1), ~np.isnan(Z2)
    W1 = mt.weight(zcov(T1, p1, sz), m1); W2 = mt.weight(zcov(T2, p2, sz), m2)
    fc = fit_L(Z1, Z2, W1, W2, parc); fa = fit_L(Z1, Z2, W1, W2, para)
    info = fc['info'] & fa['info']
    d = np.where(info, fc['chi2_1'] - fa['chi2_2'], 0.0)
    return d, info
