"""v1.5 validation X1, X2 (prereg. Sec. 6). Only samples independent of the outcome; brems off, fit (b) with c_k = 0."""
import numpy as np
import minitrack as mt
import fit2h
from validate_v14 import v4_sample


def x1():
    r, vtx, p1, p2, h1, h2 = v4_sample()
    o = fit2h.fit_pairs_2h(h1, h2, vtx, p1, p2, brems_c=False)
    d = o['dchi2'][o['ok_alp']]
    med, f384 = np.median(d), np.mean(d > 3.84)
    ok = (0.30 <= med <= 0.65) and (0.03 <= f384 <= 0.08)
    print(f'X1: n={len(d)}  median dchi2 {med:.3f} [0.30-0.65]  frac>3.84 {100*f384:.2f}% [3-8%]  '
          f'frac<-0.1 {100*np.mean(d < -0.1):.2f}%  -> {"PASS" if ok else "FAIL"}', flush=True)
    return ok, dict(median=med, f384=f384)


def x2(n=2000, seed=606):
    rng = np.random.default_rng(seed)
    meds = []
    for dphi in (0.0, 2e-3, 5e-3, 10e-3):
        r = rng.uniform(30, 300, n); a = rng.uniform(0, 2 * np.pi, n); eta = rng.uniform(-1.0, 1.0, n)
        vtx = np.stack([r * np.cos(a), r * np.sin(a), r * np.sinh(eta)], 1)
        sgn = rng.choice([-1.0, 1.0], n)                 # opening on either side of the bending direction
        f1, f2 = a + sgn * dphi / 2, a - sgn * dphi / 2
        pt = 10.0
        p1 = np.stack([pt * np.cos(f1), pt * np.sin(f1), pt * np.sinh(eta)], 1)
        p2 = np.stack([pt * np.cos(f2), pt * np.sin(f2), pt * np.sinh(eta)], 1)
        h1, _ = mt.simulate_track(vtx, p1, -1, rng, brems=False)
        h2, _ = mt.simulate_track(vtx, p2, +1, rng, brems=False)
        o = fit2h.fit_pairs_2h(h1, h2, vtx, p1, p2, brems_c=False)
        d = o['dchi2'][o['ok_alp']]
        meds.append(np.median(d))
        print(f'X2: dphi = {dphi*1e3:4.1f} mrad  n={len(d)}  median dchi2 {np.median(d):10.2f}  '
              f'frac>9 {100*np.mean(d > 9):6.2f}%', flush=True)
    ok = all(b > a for a, b in zip(meds, meds[1:]))
    print(f'X2: monotonic increase -> {"PASS" if ok else "FAIL"}', flush=True)
    return ok, meds


if __name__ == '__main__':
    a, _ = x1()
    b, _ = x2()
    print('X1', 'PASS' if a else 'FAIL', '| X2', 'PASS' if b else 'FAIL')
