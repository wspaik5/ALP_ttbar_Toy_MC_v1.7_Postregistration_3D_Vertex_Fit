"""v1.3 validation (prereg. Sec. 3): only single tracks and well-separated pion-like pairs, no bremsstrahlung.
No ALP or photon-conversion sample is used here."""
import numpy as np
import minitrack as mt

MPI = 0.1396


def mad(x):
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def v1_single(n=20000, seed=101):
    """V1/V3: tracks from the origin, beam-spot constrained (x = y = 0 fixed), fit (phi, omega)."""
    rng = np.random.default_rng(seed)
    out = {}
    for ptv in (1.0, 10.0, 30.0, 100.0):
        eta = rng.uniform(-1.2, 1.2, n); phi = rng.uniform(0, 2 * np.pi, n)
        q = rng.choice([-1.0, 1.0], n)
        p3 = np.stack([ptv * np.cos(phi), ptv * np.sin(phi), ptv * np.sinh(eta)], 1)
        vtx = np.zeros((n, 3))
        h, _ = mt.simulate_track(vtx, p3, q, rng, brems=False)
        p = np.linalg.norm(p3, axis=1); ch = np.cosh(eta)
        W = mt.weight(mt.hit_cov(vtx, p3, q), ~np.isnan(h))
        om_true = mt.omega_of(q, ptv * np.ones(n))
        par0 = np.stack([np.zeros(n), np.zeros(n), phi + 1e-4, om_true * 1.05, ], 1)
        par, chi2, cov = mt.gauss_newton(par0, [h], [W], free=[2, 3])
        pull_phi = mt.wrap(par[:, 2] - phi) / np.sqrt(cov[:, 0, 0])
        pull_om = (par[:, 3] - om_true) / np.sqrt(cov[:, 1, 1])
        dpt = (np.abs(om_true / par[:, 3]) - 1)
        ndf = (~np.isnan(h)).sum(1) - 2
        out[ptv] = dict(pull_phi=mad(pull_phi), pull_om=mad(pull_om), pt_res=mad(dpt), chi2ndf=np.median(chi2 / ndf))
        print(f'V1/V3 pT={ptv:5.0f} GeV: pull(phi) {mad(pull_phi):.3f}  pull(omega) {mad(pull_om):.3f}  '
              f'sigma(pT)/pT {100 * mad(dpt):.2f}%  median chi2/ndf {np.median(chi2 / ndf):.2f}')
    return out


def v2_pairs(n=20000, seed=202):
    """V2: two pion-like tracks from a displaced vertex, opening angle 10-100 mrad, no brems."""
    rng = np.random.default_rng(seed)
    r = rng.uniform(5, 500, n); a = rng.uniform(0, 2 * np.pi, n)
    eta = rng.uniform(-1.0, 1.0, n)
    vtx = np.stack([r * np.cos(a), r * np.sin(a), r * np.sinh(eta)], 1)
    out = {}
    for ptv in (2.0, 10.0):
        op = rng.uniform(0.01, 0.1, n) * rng.choice([-1, 1], n)
        ph1, ph2 = a + op / 2, a - op / 2
        pt1 = ptv * rng.uniform(0.5, 1.5, n); pt2 = ptv * rng.uniform(0.5, 1.5, n)
        p1 = np.stack([pt1 * np.cos(ph1), pt1 * np.sin(ph1), pt1 * np.sinh(eta)], 1)
        p2 = np.stack([pt2 * np.cos(ph2), pt2 * np.sin(ph2), pt2 * np.sinh(eta)], 1)
        h1, _ = mt.simulate_track(vtx, p1, -1, rng, brems=False)
        h2, _ = mt.simulate_track(vtx, p2, +1, rng, brems=False)
        o = mt.vertex_fit(h1, h2, vtx, p1, p2)
        ok = o['ok']
        pull = (o['r'] - r) / o['sigma_r']
        out[ptv] = dict(pull=mad(pull[ok]), res=mad((o['r'] - r)[ok]), okfrac=ok.mean(),
                        f3=np.mean(np.abs(pull[ok]) > 3))
        print(f'V2 pT~{ptv:4.0f} GeV: fit ok {100 * ok.mean():.1f}%  radial pull width {mad(pull[ok]):.3f}  '
              f'sigma_core {mad((o["r"] - r)[ok]):.3f} mm  |pull|>3: {100 * np.mean(np.abs(pull[ok]) > 3):.2f}%  '
              f'median chi2/ndf {np.median((o["chi2"] / o["ndf"])[ok]):.2f}')
    return out


if __name__ == '__main__':
    v1 = v1_single()
    v2 = v2_pairs()
    ok1 = all(0.9 <= d['pull_phi'] <= 1.15 and 0.9 <= d['pull_om'] <= 1.15 for d in v1.values())
    ok2 = all(0.9 <= d['pull'] <= 1.15 for d in v2.values())
    ok3 = all(0.01 <= v1[k]['pt_res'] <= 0.04 for k in (10.0, 30.0, 100.0))
    print('V1', 'PASS' if ok1 else 'FAIL', '| V2', 'PASS' if ok2 else 'FAIL', '| V3', 'PASS' if ok3 else 'FAIL')


def v4_collinear(n=8000, seed=404, brems=False, parallel=False):
    """V4 (added post hoc, see report): exactly collinear pion-like pairs (opening angle 0), as in a photon
    conversion but generated independently of the outcome sample, no brems. Energy sharing x ~ U(0.1, 0.9)."""
    rng = np.random.default_rng(seed)
    r = rng.uniform(30, 700, n); a = rng.uniform(0, 2 * np.pi, n); eta = rng.uniform(-1.0, 1.0, n)
    vtx = np.stack([r * np.cos(a), r * np.sin(a), r * np.sinh(eta)], 1)
    ptg = 20 * rng.uniform(0, 1, n)**-0.5
    x = rng.uniform(0.1, 0.9, n)
    d = np.stack([np.cos(a), np.sin(a), np.sinh(eta)], 1)
    p1, p2 = d * (x * ptg)[:, None], d * ((1 - x) * ptg)[:, None]
    h1, _ = mt.simulate_track(vtx, p1, -1, rng, brems=brems)
    h2, _ = mt.simulate_track(vtx, p2, +1, rng, brems=brems)
    o = mt.vertex_fit(h1, h2, vtx, p1, p2, parallel=parallel)
    ok = o['ok']; dr = (o['r'] - r)[ok]; pull = dr / o['sigma_r'][ok]
    pix = (r < 160)[ok]
    print(f'V4 collinear (parallel={parallel}): fit ok {100*ok.mean():.1f}%  pull width {mad(pull):.3f}  '
          f'|pull|>3 {100*np.mean(np.abs(pull) > 3):.2f}%  sigma_core pixel {mad(dr[pix]):.2f} mm / strip '
          f'{mad(dr[~pix]):.2f} mm  median sigma_rep pixel {np.median(o["sigma_r"][ok][pix]):.2f} / strip '
          f'{np.median(o["sigma_r"][ok][~pix]):.2f} mm  median bias {np.median(dr):.2f} mm')
    return dict(pull=mad(pull), f3=np.mean(np.abs(pull) > 3), okfrac=ok.mean())


if __name__ == '__main__':
    v4_collinear(parallel=False)
    v4_collinear(parallel=True)
