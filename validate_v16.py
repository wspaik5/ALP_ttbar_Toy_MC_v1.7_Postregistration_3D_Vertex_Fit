"""v1.6 validation Y1-Y3 (prereg. Sec. 5). The field is set by the environment variable V16_B before import."""
import sys
import numpy as np
import minitrack as mt
import fit2h
from validate_v14 import v4_sample


def y1():
    r, vtx, p1, p2, h1, h2 = v4_sample()
    o = fit2h.fit_pairs_2h(h1, h2, vtx, p1, p2, brems_c=False)
    d = o['dchi2'][o['ok_alp']]
    z = np.mean(d < 1e-3); med = np.median(d[d >= 1e-3])
    ok = (0.40 <= z <= 0.60) and (0.30 <= med <= 0.65)
    print(f'Y1 B={mt.B}: n={len(d)} frac(dchi2<1e-3) {z:.3f} [0.40-0.60]  median non-zero {med:.3f} [0.30-0.65]  '
          f'-> {"PASS" if ok else "FAIL"}', flush=True)


def y2(n=2000, seed=606, dphi=2e-3):
    """Identical generator to v1.5 X2 at 2 mrad (same rng sequence up to the first Delta-phi value used there)."""
    rng = np.random.default_rng(seed)
    for dp in (0.0, dphi):                       # reproduce the v1.5 X2 rng sequence: 0 mrad first, then 2 mrad
        r = rng.uniform(30, 300, n); a = rng.uniform(0, 2 * np.pi, n); eta = rng.uniform(-1.0, 1.0, n)
        vtx = np.stack([r * np.cos(a), r * np.sin(a), r * np.sinh(eta)], 1)
        sgn = rng.choice([-1.0, 1.0], n)
        f1, f2 = a + sgn * dp / 2, a - sgn * dp / 2
        p1 = np.stack([10 * np.cos(f1), 10 * np.sin(f1), 10 * np.sinh(eta)], 1)
        p2 = np.stack([10 * np.cos(f2), 10 * np.sin(f2), 10 * np.sinh(eta)], 1)
        h1, _ = mt.simulate_track(vtx, p1, -1, rng, brems=False)
        h2, _ = mt.simulate_track(vtx, p2, +1, rng, brems=False)
        if dp == 0.0:
            continue
        o = fit2h.fit_pairs_2h(h1, h2, vtx, p1, p2, brems_c=False)
        d = o['dchi2'][o['ok_alp']]
        print(f'Y2 B={mt.B}: dphi=2 mrad median dchi2 {np.median(d):.4f}  frac>9 {100*np.mean(d > 9):.2f}%', flush=True)


def y3(n=500, seed=707):
    rng = np.random.default_rng(seed)
    phi0 = rng.uniform(0, 2 * np.pi, n); q = rng.choice([-1.0, 1.0], n)
    p3 = np.stack([10 * np.cos(phi0), 10 * np.sin(phi0), np.zeros(n)], 1)
    h, _ = mt.simulate_track(np.zeros((n, 3)), p3, q, rng, brems=False, ms=False, smear=False)
    om = mt.omega_of(q, 10.0)
    ana = phi0[:, None] + np.arcsin(mt.SENS[None, :] * om[:, None] / 2)       # circle through the origin
    dev = np.nanmax(np.abs(mt.SENS[None, :] * mt.wrap(h - ana))) * 1e3
    print(f'Y3 B={mt.B}: max |hit - analytic| = {dev:.2e} um [< 0.1 um] -> {"PASS" if dev < 0.1 else "FAIL"}', flush=True)


if __name__ == '__main__':
    for t in sys.argv[1:]:
        {'y1': y1, 'y2': y2, 'y3': y3}[t]()
