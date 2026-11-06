"""v1.3 preregistered analysis: vertex resolution and tails from the mini tracking simulation (minitrack.py),
fed into the v1.1 selection (bkg.scan_v11). Run only after the preregistration is public."""
import numpy as np
import time
import bkg, mag, vtx, hitpattern as hp
import minitrack as mt
import sys

PARALLEL = len(sys.argv) > 1 and sys.argv[1] == 'parallel'
TAG = 'B_parallel' if PARALLEL else 'A_prereg'

MASSES = [0.017, 0.05, 0.1, 0.2]
CTAU = 1.0
MODEL = 'kinematic'          # mass smearing exactly as in v1.1 (bkg.measure); r and sigma_r replaced below


def mad(x):
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


def conversions_truth(n=1_500_000, seed=5):
    """Exact copy of bkg.conversions() (same seed, same random sequence) that also keeps the truth
    vertex and momenta. Returns the bkg-style dict plus vtx, e1 (e-), e2 (e+)."""
    rng = np.random.default_rng(seed)
    A, X0, MAT, HW = bkg.A, bkg.X0, bkg.MAT, bkg.HW
    pt = 20 * rng.uniform(0, 1, n)**-0.5
    eta = rng.normal(0, 1.6, 3*n); eta = eta[np.abs(eta) < 1.2][:n]; n = len(eta); pt = pt[:n]
    phi = rng.uniform(0, 2*np.pi, n)
    pconv = 1 - np.exp(-7/9 * X0[None, :] * np.cosh(eta)[:, None])
    surv = np.cumprod(np.hstack([np.ones((n, 1)), 1-pconv[:, :-1]]), 1)
    prob = surv * pconv
    cum = np.cumsum(prob, 1); u = rng.uniform(0, 1, n)
    layer = (u[:, None] > cum).sum(1)
    conv = layer < len(MAT)
    pt, eta, phi, layer = pt[conv], eta[conv], phi[conv], layer[conv]
    r = MAT[layer] + rng.uniform(-1, 1, len(pt))*HW[layer]
    x = rng.uniform(0, 1, 4*len(pt)); acc = rng.uniform(0, 1, 4*len(pt)) < 1 - 4/3*x*(1-x)
    x = x[acc][:len(pt)]
    dirv = np.stack([np.cos(phi), np.sin(phi), np.sinh(eta)], 1) / np.cosh(eta)[:, None]
    p = pt*np.cosh(eta)
    e1, e2 = dirv*(x*p)[:, None], dirv*((1-x)*p)[:, None]
    mm, rm, sr = bkg.measure(r, e1, e2, rng, MODEL)
    w = np.full(len(r), A['sigma_ttgamma_pb']*bkg.F_BARREL*bkg.LUMI/n)
    v = np.stack([r*np.cos(phi), r*np.sin(phi), r*np.sinh(eta)], 1)
    return dict(r=r, rm=rm, sr=sr, m=mm, base=bkg.basic_cuts(r, e1, e2), w=w, vtx=v, e1=e1, e2=e2)


def signal_truth(m, ctau, n=300_000, seed=11):
    """Exact copy of bkg.signal() that also keeps the truth (mag.make_events returns vtx, e-, e+)."""
    rng = np.random.default_rng(seed)
    v, em, ep = mag.make_events(m, ctau, n, seed=seed)
    r = np.hypot(v[:, 0], v[:, 1])
    mm, rm, sr = bkg.measure(r, em, ep, rng, MODEL)
    w = np.full(len(r), bkg.F_BARREL*bkg.LUMI/len(r))
    return dict(r=r, rm=rm, sr=sr, m=mm, base=bkg.basic_cuts(r, em, ep), w=w, vtx=v, e1=em, e2=ep)


def apply_minisim(d, seed, brems=True):
    """Replace measured vertex radius and per-pair resolution by the mini-simulation fit (base pairs only).
    Pairs whose fit fails are removed from 'base'. Keeps the v1.1 values as rm_v11, sr_v11."""
    out = dict(d)
    out['rm_v11'], out['sr_v11'] = d['rm'], d['sr']
    idx = np.where(d['base'])[0]
    t = time.time()
    f = mt.reconstruct_pairs(d['vtx'][idx], d['e1'][idx], d['e2'][idx], seed=seed, brems=brems, parallel=PARALLEL)
    rm = np.full(len(d['r']), np.nan); sr = np.full(len(d['r']), np.nan); ok = np.zeros(len(d['r']), bool)
    rm[idx], sr[idx], ok[idx] = f['r'], f['sigma_r'], f['ok']
    out['rm'], out['sr'] = np.nan_to_num(rm, nan=-1.0), np.nan_to_num(sr, nan=1.0)
    out['fit_ok_frac'] = ok[idx].mean() if len(idx) else np.nan
    out['base'] = d['base'] & ok
    out['nbehind'] = np.zeros(len(d['r'])); out['nbehind'][idx] = f['nbehind']
    print(f'    mini-sim on {len(idx)} pairs ({"brems" if brems else "no brems"}): {time.time()-t:.0f} s, '
          f'fit ok {100*out["fit_ok_frac"]:.2f}%', flush=True)
    return out


def metrics(d, sel):
    dr = (d['rm'] - d['r'])[sel]; pull = dr / d['sr'][sel]
    return dict(n=int(sel.sum()), sigma_core=mad(dr), f3=np.mean(np.abs(pull) > 3) if sel.any() else np.nan,
                pull_width=mad(pull), median_sr=np.median(d['sr'][sel]) if sel.any() else np.nan)


def with_hp(d, t):
    d2 = dict(d); d2['base'] = d['base'] & hp.consistent(d['r'], d['rm'], d['sr'], t); return d2


if __name__ == '__main__':
    res = {}
    print(f'mode {TAG}', flush=True)
    print('conversions ...', flush=True)
    C0 = conversions_truth()
    Cb = apply_minisim(C0, seed=31, brems=True)
    Cn = apply_minisim(C0, seed=31, brems=False)
    res['conv_fit_ok'] = (Cb['fit_ok_frac'], Cn['fit_ok_frac'])
    for tag, D in (('brems', Cb), ('nobrems', Cn)):
        for reg, cut in (('pixel', D['r'] < 160), ('strip', D['r'] >= 160), ('all', D['r'] < 800)):
            sel = D['base'] & cut
            res[('conv', tag, reg)] = metrics(D, sel)
    # P1: median sigma_rep (no brems, pixel) vs median v1.1 vtx.py estimate for the same pairs
    sel = Cn['base'] & (Cn['r'] < 160)
    res['P1'] = dict(minisim=np.median(Cn['sr'][sel]), v11=np.median(Cn['sr_v11'][sel]))
    res['P1']['ratio'] = res['P1']['minisim'] / res['P1']['v11']

    print('signal ...', flush=True)
    for m in MASSES:
        S0 = signal_truth(m, CTAU)
        Sb = apply_minisim(S0, seed=41, brems=True)
        res[('sig_fit_ok', m)] = Sb['fit_ok_frac']
        res[('sig', m)] = metrics(Sb, Sb['base'])
        lim0 = bkg.upper_limit(Sb['w'][Sb['base']].sum(), 0)
        r0 = bkg.scan_v11(m, Sb, Cb)
        r1 = bkg.scan_v11(m, with_hp(Sb, 1.0), with_hp(Cb, 1.0))
        # v1.1 reference recomputed on the identical (fit-ok) pairs with the v1.1 per-pair Gaussian model
        Sv, Cv = dict(Sb, rm=Sb['rm_v11'], sr=Sb['sr_v11']), dict(Cb, rm=Cb['rm_v11'], sr=Cb['sr_v11'])
        rv = bkg.scan_v11(m, Sv, Cv)
        res[('deg', m)] = dict(primary=r0['limit']/lim0, hp=r1['limit']/lim0, v11_same_pairs=rv['limit']/lim0,
                               nsig=r0['nsig'], eff=r0['eff'], B=r0['B'], nsig_hp=r1['nsig'], B_hp=r1['B'])
        print(f'  {m*1000:4.0f} MeV: degradation primary {r0["limit"]/lim0:7.1f}x  (n_sigma={r0["nsig"]}, '
              f'eff {r0["eff"]:.2f}%, B={r0["B"]:.1f}) | +hit pattern {r1["limit"]/lim0:7.1f}x | '
              f'v1.1 per-pair Gaussian on same pairs {rv["limit"]/lim0:6.1f}x', flush=True)

    print('\n=== resolution metrics ===')
    for k, v in res.items():
        if isinstance(k, tuple) and k[0] in ('conv', 'sig'):
            print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})
    print('P1', res['P1'])
    print('fit ok (conv brems, nobrems)', res['conv_fit_ok'])
    np.save(f'v13_results_{TAG}.npy', res, allow_pickle=True)
    # keep per-pair arrays for the figures
    keep = lambda D: {k: D[k] for k in ('r', 'rm', 'sr', 'base', 'w', 'm', 'rm_v11', 'sr_v11', 'nbehind')}
    np.savez_compressed(f'v13_conv_arrays_{TAG}.npz', **{f'b_{k}': v for k, v in keep(Cb).items()},
                        **{f'n_{k}': v for k, v in keep(Cn).items()})
