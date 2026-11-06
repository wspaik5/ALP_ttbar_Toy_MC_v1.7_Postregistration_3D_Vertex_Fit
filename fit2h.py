"""v1.5 (preregistered 2026-11-03, doi:10.5281/zenodo.23905511): two-hypothesis vertex fit.

H_conv : v1.4 fit (b) exactly (brems-noise covariance + mean-loss track model, parallel-at-vertex constraint).
H_ALP  : same hits, same covariance and track model, both track directions free (6 parameters),
         seeded from the H_conv solution (phi2 = phi1 at the start). Gauss-Newton only accepts downhill steps.
dchi2  : chi2_conv - chi2_ALP.
'ab'   : (a)+(b) combination = fit (b) on the first 6 measured hits of each track, conversion hypothesis only."""
import numpy as np
import minitrack as mt
import brfit as bf

KEYS = ('r', 'sigma_r', 'chi2_conv', 'chi2_alp', 'dchi2', 'r_alp', 'om1_alp', 'om2_alp')
PKEYS = ('par_conv', 'par_alp')   # v1.7: fitted transverse parameters (x, y, phi1, om1, phi2, om2)


def fit_pairs_2h(H1, H2, vtx, e1, e2, brems_c=True, n_keep=None, alp=True, chunk=20000):
    n = len(vtx)
    res = {k: np.full(n, np.nan) for k in KEYS}
    res['ok'] = np.zeros(n, bool); res['ok_alp'] = np.zeros(n, bool)
    for k in PKEYS:
        res[k] = np.full((n, 6), np.nan)
    for s in range(0, n, chunk):
        sl = slice(s, min(n, s + chunk))
        h1, h2 = H1[sl], H2[sl]
        if n_keep is not None:
            h1, h2 = bf.keep_inner(h1, n_keep), bf.keep_inner(h2, n_keep)
        o = mt.vertex_fit(h1, h2, vtx[sl], e1[sl], e2[sl], parallel=True,
                          cov_fn=bf.make_cov_b(brems_c), pred_fn=bf.make_pred_b(brems_c), return_state=True)
        idx = np.arange(sl.start, sl.stop)
        res['r'][sl], res['sigma_r'][sl], res['ok'][sl] = o['r'], o['sigma_r'], o['ok']
        st = o.get('state')
        if st is None:
            continue
        g = idx[st['sel']]
        res['chi2_conv'][g] = st['chi2']; res['par_conv'][g] = st['par']
        if not alp:
            continue
        par, c, _ = mt.gauss_newton(st['par'].copy(), [st['h1'], st['h2']], [st['W1'], st['W2']],
                                    free=[0, 1, 2, 3, 4, 5], preds=st['preds'])
        res['chi2_alp'][g] = c
        res['dchi2'][g] = st['chi2'] - c
        res['r_alp'][g] = np.hypot(par[:, 0], par[:, 1])
        res['om1_alp'][g], res['om2_alp'][g] = par[:, 3], par[:, 5]; res['par_alp'][g] = par     # v1.6: curvature for P6
        res['ok_alp'][g] = np.isfinite(c) & np.isfinite(st['chi2'])
    return res
