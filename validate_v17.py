"""v1.7 validation Z1-Z3 (prereg. Sec. 6), default z geometry, no bremsstrahlung, fit (b) with c_k = 0.
Uses the same r-z machinery as the main run (lz.zcov, lz.dchi2_L with the H_conv vertex for both r-z hypotheses)."""
import sys
import numpy as np
import minitrack as mt
import fit2h, lz


def z_hits(inf, H, rngL, geom='default'):
    """Same formulas as lz.sim_pairs for one track sample."""
    n, NL = H.shape
    kick = rngL.normal(0, 1, (n, inf['s_mat'].shape[1])); zeta = rngL.normal(0, 1, (n, NL))
    th0 = lz.highland(inf['p_mat'], inf['t_mat'][None, :] * inf['cosh'][:, None])
    dcot = np.nan_to_num(kick * th0 * inf['cosh'][:, None]**2)
    arm = np.clip(inf['s_sens'][:, :, None] - np.nan_to_num(inf['s_mat'], nan=1e9)[:, None, :], 0, None)
    ztrue = inf['z0'][:, None] + inf['cot'][:, None] * inf['s_sens'] + (arm * dcot[:, None, :]).sum(2)
    sz = lz.GEOM[geom]
    zz = ztrue + zeta * np.where(np.isfinite(sz), sz, 0)[None, :]
    return np.where(np.isfinite(sz)[None, :] & ~np.isnan(H), zz, np.nan)


def run_pairs(vtx, p1, p2, h1, i1, h2, i2, rngL):
    Z1 = z_hits(i1, h1, rngL); Z2 = z_hits(i2, h2, rngL)
    o = fit2h.fit_pairs_2h(h1, h2, vtx, p1, p2, brems_c=False)
    T1 = {k: i1[k] for k in ('s_sens', 's_mat', 'cosh', 't_mat')}; T2 = {k: i2[k] for k in ('s_sens', 's_mat', 'cosh', 't_mat')}
    dL, info = lz.dchi2_L(Z1, Z2, T1, T2, p1, p2, o['par_conv'], o['par_conv'], 'default')
    return o, dL, info


def z1():
    rng = np.random.default_rng(404); n = 8000                       # v1.3 V4 generator, same draws
    r = rng.uniform(30, 700, n); a = rng.uniform(0, 2 * np.pi, n); eta = rng.uniform(-1.0, 1.0, n)
    vtx = np.stack([r * np.cos(a), r * np.sin(a), r * np.sinh(eta)], 1)
    ptg = 20 * rng.uniform(0, 1, n)**-0.5; x = rng.uniform(0.1, 0.9, n)
    d = np.stack([np.cos(a), np.sin(a), np.sinh(eta)], 1)
    p1, p2 = d * (x * ptg)[:, None], d * ((1 - x) * ptg)[:, None]
    h1, i1 = mt.simulate_track(vtx, p1, -1, rng, brems=False, return_path=True)
    h2, i2 = mt.simulate_track(vtx, p2, +1, rng, brems=False, return_path=True)
    o, dL, info = run_pairs(vtx, p1, p2, h1, i1, h2, i2, np.random.default_rng(1404))
    m = info & o['ok_alp']; L = dL[m]; D3 = (o['dchi2'] + dL)[m]
    okL = 0.30 <= np.median(L) <= 0.65 and 0.03 <= np.mean(L > 3.84) <= 0.08
    ok3 = 0.60 <= np.median(D3) <= 1.15 and 0.06 <= np.mean(D3 > 3.84) <= 0.14
    print(f'Z1: pairs with r-z information {m.sum()} of {n}')
    print(f'Z1: dchi2_L  median {np.median(L):.3f} [0.30-0.65]  frac>3.84 {100*np.mean(L > 3.84):.2f}% [3-8%]  -> {"PASS" if okL else "FAIL"}')
    print(f'Z1: dchi2_3D median {np.median(D3):.3f} [0.60-1.15]  frac>3.84 {100*np.mean(D3 > 3.84):.2f}% [6-14%]  -> {"PASS" if ok3 else "FAIL"}', flush=True)
    return okL and ok3


def z2(n=2000):
    rng = np.random.default_rng(808); rngL = np.random.default_rng(1808); meds = []
    for dth in (0.0, 0.5e-3, 1e-3, 2e-3):
        r = rng.uniform(30, 100, n); a = rng.uniform(0, 2 * np.pi, n); eta = rng.uniform(-1.0, 1.0, n)
        vtx = np.stack([r * np.cos(a), r * np.sin(a), r * np.sinh(eta)], 1)
        th = 2 * np.arctan(np.exp(-eta)); sg = rng.choice([-1.0, 1.0], n)
        t1, t2 = th + sg * dth / 2, th - sg * dth / 2
        p1 = np.stack([10 * np.cos(a), 10 * np.sin(a), 10 / np.tan(t1)], 1)
        p2 = np.stack([10 * np.cos(a), 10 * np.sin(a), 10 / np.tan(t2)], 1)
        h1, i1 = mt.simulate_track(vtx, p1, -1, rng, brems=False, return_path=True)
        h2, i2 = mt.simulate_track(vtx, p2, +1, rng, brems=False, return_path=True)
        o, dL, info = run_pairs(vtx, p1, p2, h1, i1, h2, i2, rngL)
        L = dL[info & o['ok_alp']]; meds.append(np.median(L))
        print(f'Z2: dtheta = {dth*1e3:.1f} mrad  n with r-z info {len(L)}  median dchi2_L {np.median(L):9.2f}  frac>9 {100*np.mean(L > 9):6.2f}%', flush=True)
    ok = all(b > a for a, b in zip(meds, meds[1:]))
    print(f'Z2: monotonic -> {"PASS" if ok else "FAIL"}')
    return ok


def z3(n=500):
    rng = np.random.default_rng(909)
    r = rng.uniform(30, 300, n); a = rng.uniform(0, 2 * np.pi, n); eta = rng.uniform(-1.0, 1.0, n)
    vtx = np.stack([r * np.cos(a), r * np.sin(a), r * np.sinh(eta)], 1)
    ph = a + rng.uniform(-0.05, 0.05, n)
    p = np.stack([10 * np.cos(ph), 10 * np.sin(ph), 10 * np.sinh(eta)], 1)
    h, inf = mt.simulate_track(vtx, p, -1, rng, brems=False, ms=False, smear=False, return_path=True)
    om = mt.omega_of(-1.0, np.full(n, 10.0))
    s_ana = mt.to_radius(vtx[:, :1], vtx[:, 1:2], ph[:, None], om[:, None], mt.SENS[None, :], 'forward')
    z_sim = inf['z0'][:, None] + inf['cot'][:, None] * inf['s_sens']
    z_ana = vtx[:, 2:3] + np.sinh(eta)[:, None] * s_ana
    m = ~np.isnan(z_sim)
    dev = np.nanmax(np.abs(z_sim - z_ana)[m]) * 1e3
    print(f'Z3: max |z_sim - (z_v + cot(theta) s)| = {dev:.2e} um [< 0.1 um] -> {"PASS" if dev < 0.1 else "FAIL"}')
    return dev < 0.1


if __name__ == '__main__':
    out = [{'z1': z1, 'z2': z2, 'z3': z3}[t]() for t in sys.argv[1:]]
