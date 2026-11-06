"""v1.4 validation W1-W3 (prereg. Sec. 5). Only samples independent of the outcome."""
import numpy as np
import minitrack as mt
import brfit as bf


def mad(x):
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def w1_single_electrons(n=6000, seed=505):
    """Single electrons from the origin with brems; beam-spot constrained fit of (phi, omega).
    Compare the brems-blind v1.3 fit with fit (b)."""
    rng = np.random.default_rng(seed)
    pt = rng.uniform(5, 100, n); eta = rng.uniform(-1.2, 1.2, n); phi = rng.uniform(0, 2 * np.pi, n)
    q = rng.choice([-1.0, 1.0], n)
    p3 = np.stack([pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)], 1)
    vtx = np.zeros((n, 3))
    h, _ = mt.simulate_track(vtx, p3, q, rng, brems=True)
    m = ~np.isnan(h)
    om = mt.omega_of(q, pt)
    out = {}
    for tag, cf, pf in (('blind', mt.hit_cov, None), ('b', bf.make_cov_b(True), bf.make_pred_b(True))):
        W = mt.weight(cf(vtx, p3, q), m)
        preds = None if pf is None else [pf(vtx, p3, q)]
        par0 = np.stack([np.zeros(n), np.zeros(n), phi, om], 1)
        par, chi2, cov = mt.gauss_newton(par0, [h], [W], free=[2, 3], preds=preds)
        rel = par[:, 3] / om - 1
        pull = (par[:, 3] - om) / np.sqrt(np.abs(cov[:, 1, 1]))
        out[tag] = dict(bias=np.median(rel), width=mad(rel), pull=mad(pull))
        print(f'W1 {tag:5s}: median curvature bias {100*np.median(rel):+.2f}%  spread {100*mad(rel):.2f}%  '
              f'curvature pull width {mad(pull):.2f}')
    return out


def v4_sample(n=8000, seed=404):
    """The v1.3 V4 collinear pion-like sample (same generator and seed)."""
    rng = np.random.default_rng(seed)
    r = rng.uniform(30, 700, n); a = rng.uniform(0, 2 * np.pi, n); eta = rng.uniform(-1.0, 1.0, n)
    vtx = np.stack([r * np.cos(a), r * np.sin(a), r * np.sinh(eta)], 1)
    ptg = 20 * rng.uniform(0, 1, n)**-0.5
    x = rng.uniform(0.1, 0.9, n)
    d = np.stack([np.cos(a), np.sin(a), np.sinh(eta)], 1)
    p1, p2 = d * (x * ptg)[:, None], d * ((1 - x) * ptg)[:, None]
    h1, _ = mt.simulate_track(vtx, p1, -1, rng, brems=False)
    h2, _ = mt.simulate_track(vtx, p2, +1, rng, brems=False)
    return r, vtx, p1, p2, h1, h2


def w23():
    r, vtx, p1, p2, h1, h2 = v4_sample()
    out = {}
    for tag, meth in (('W2 (b, c=0)', 'b0'), ('W3 (a, N=6)', 'a')):
        o = bf.fit_pairs(h1, h2, vtx, p1, p2, meth)
        ok = o['ok']; pull = ((o['r'] - r) / o['sigma_r'])[ok]
        out[tag] = mad(pull)
        print(f'{tag}: fit ok {100*ok.mean():.1f}%  pull width {mad(pull):.3f}  core {mad((o["r"]-r)[ok]):.2f} mm')
    return out


if __name__ == '__main__':
    w1 = w1_single_electrons()
    w = w23()
    ok1 = abs(w1['b']['bias']) <= 0.03 and abs(w1['b']['bias']) < abs(w1['blind']['bias'])
    ok2 = 0.9 <= w['W2 (b, c=0)'] <= 1.15
    ok3 = 0.9 <= w['W3 (a, N=6)'] <= 1.15
    print('W1', 'PASS' if ok1 else 'FAIL', '| W2', 'PASS' if ok2 else 'FAIL', '| W3', 'PASS' if ok3 else 'FAIL')
