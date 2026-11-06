"""v1.5 preregistered analysis (doi:10.5281/zenodo.23905511). Stages:
  python3 run_v15.py fit SAMPLE METHOD   SAMPLE in {conv, sig0.017, sig0.05, sig0.1, sig0.2}; METHOD in {2h, ab}
                                         -> v15_fit_METHOD_SAMPLE.npz   (same hits as v1.3/v1.4: v14_hits.npz)
  python3 run_v15.py analyze             -> sanity gate, scan_v15, V-nomass, P1-P6, v15_results.npy"""
import sys, time
import numpy as np
import bkg
import fit2h
from run_v13 import metrics
from run_v14 import samples, MASSES, V13_B, V13_NOBREMS, attach

SAMPLES = ['conv'] + [f'sig{m}' for m in MASSES]
KGRID = (None, 1, 2, 4, 9, 16, 25, 49)
V14_B = {0.017: 126, 0.05: 68, 0.1: 32, 0.2: 14}


def stage_fit(k, method):
    Z = np.load('v14_hits.npz'); S = samples(); d = S[k]
    idx = Z[f'{k}_idx']; t = time.time()
    kw = dict(alp=True) if method == '2h' else dict(alp=False, n_keep=6)
    o = fit2h.fit_pairs_2h(Z[f'{k}_H1'], Z[f'{k}_H2'], d['vtx'][idx], d['e1'][idx], d['e2'][idx], **kw)
    np.savez_compressed(f'v15_fit_{method}_{k}.npz', **o)
    print(f'fit {method} {k}: {len(idx)} pairs, {time.time()-t:.0f} s, ok {100*o["ok"].mean():.2f}%', flush=True)


def scan_v15(m, sig, bkgd, dchi_s, dchi_b, kgrid=KGRID, mass_window=True):
    """scan_v11 + exemption threshold k: pairs with dchi2 >= k skip the material veto."""
    S0 = sig['w'][sig['base']].sum(); lim0 = bkg.upper_limit(S0, 0)
    best = None
    cands = [(np.ones(len(sig['r']), bool), np.ones(len(bkgd['r']), bool))]
    if mass_window:
        for lo in np.linspace(0, 1.5 * m, 31):
            for hi in (3 * m + 0.01, np.inf):
                cands.append(((sig['m'] > lo) & (sig['m'] < hi), (bkgd['m'] > lo) & (bkgd['m'] < hi)))
    for ns in bkg.NSIG:
        vs0 = np.ones(len(sig['r']), bool) if ns is None else bkg.material_veto(sig['rm'], sig['sr'], ns)
        vb0 = np.ones(len(bkgd['r']), bool) if ns is None else bkg.material_veto(bkgd['rm'], bkgd['sr'], ns)
        for kk in kgrid:
            if kk is None:
                vs, vb = vs0, vb0
            else:
                vs, vb = vs0 | (dchi_s >= kk), vb0 | (dchi_b >= kk)
            for ms, mb in cands:
                cs, cb = sig['base'] & vs & ms, bkgd['base'] & vb & mb
                S, B = sig['w'][cs].sum(), bkgd['w'][cb].sum()
                lim = bkg.upper_limit(S, B)
                if best is None or lim < best['limit']:
                    best = dict(limit=lim, eff=cs.mean() * 100, B=B, nsig=ns, k=kk, deg=lim / lim0)
    return best


def full_dchi(d, F):
    """dchi2 on the full sample index (non-base / failed fits -> -inf, i.e. never exempt)."""
    out = np.full(len(d['r']), -np.inf); idx = np.where(d['base'])[0]
    x = np.where(F['ok_alp'], F['dchi2'], -np.inf); out[idx] = x
    return out


def stage_analyze():
    S = samples(); res = {'gate': {}}
    F2 = {k: np.load(f'v15_fit_2h_{k}.npz') for k in SAMPLES}
    Fab = {k: np.load(f'v15_fit_ab_{k}.npz') for k in SAMPLES}
    Fb = np.load('v14_fit_b.npz')
    # ---- sanity gate (prereg. Sec. 5)
    gate_ok = True
    for k in SAMPLES:
        F = F2[k]; old = Fb[f'{k}_r']
        same = np.isclose(F['r'], old, rtol=0, atol=1e-6) | (np.isnan(F['r']) & np.isnan(old))
        fa = F['ok_alp'].mean(); d = F['dchi2'][F['ok_alp']]; fneg = np.mean(d < -0.1)
        g = (same.mean() >= 0.999) and (fa >= 0.95) and (fneg <= 0.01)
        res['gate'][k] = dict(identity=same.mean(), ok_alp=fa, frac_neg=fneg, passed=g); gate_ok &= g
        print(f'gate {k:9s}: identity {100*same.mean():7.3f}%  H_ALP ok {100*fa:6.2f}%  dchi2<-0.1 {100*fneg:5.2f}%  '
              f'{"PASS" if g else "FAIL"}', flush=True)
    if not gate_ok:
        print('SANITY GATE FAILED: no degradation computed'); np.save('v15_results.npy', res, allow_pickle=True); return
    # ---- primary: two-hypothesis selection on v1.4 (b) vertices
    Cb = attach(S['conv'], Fb, 'conv'); dC = full_dchi(S['conv'], F2['conv'])
    Cab = attach(S['conv'], {'conv_r': Fab['conv']['r'], 'conv_sigma_r': Fab['conv']['sigma_r'],
                             'conv_ok': Fab['conv']['ok']}, 'conv')
    sel = Cb['base'] & (Cb['r'] < 800)
    dsel = dC[sel]
    res['P5_conv'] = float(np.mean(dsel > 9))
    res['dchi_conv'] = dC[sel]
    print(f'\nconversions r<800: frac dchi2>9 = {100*res["P5_conv"]:.1f}%  median {np.median(dsel):.3g}')
    for m in MASSES:
        k = f'sig{m}'
        Sg = attach(S[k], Fb, k); dS = full_dchi(S[k], F2[k])
        Sab = attach(S[k], {f'{k}_r': Fab[k]['r'], f'{k}_sigma_r': Fab[k]['sigma_r'], f'{k}_ok': Fab[k]['ok']}, k)
        ds = dS[Sg['base']]
        res[('P5_sig', m)] = float(np.mean(ds > 9)); res[('dchi_sig', m)] = ds
        r14 = scan_v15(m, Sg, Cb, dS, dC, kgrid=(None,))
        r15 = scan_v15(m, Sg, Cb, dS, dC)
        rab = scan_v15(m, Sab, Cab, dS, dC, kgrid=(None,))
        n14 = scan_v15(m, Sg, Cb, dS, dC, kgrid=(None,), mass_window=False)
        n15 = scan_v15(m, Sg, Cb, dS, dC, mass_window=False)
        res[('v14b', m)], res[('v15', m)], res[('ab', m)] = r14, r15, rab
        res[('v14b_nomass', m)], res[('v15_nomass', m)] = n14, n15
        print(f'{m*1000:4.0f} MeV: v1.4(b) re-scan {r14["deg"]:6.1f}x | v1.5 {r15["deg"]:6.1f}x (k={r15["k"]}, nsig={r15["nsig"]}, '
              f'eff {r15["eff"]:.2f}%, B={r15["B"]:.1f}) | (a)+(b) {rab["deg"]:6.1f}x | V-nomass v1.4(b) {n14["deg"]:6.1f}x '
              f'v1.5 {n15["deg"]:6.1f}x | ALP frac dchi2>9 {100*res[("P5_sig", m)]:.1f}%', flush=True)
    np.save('v15_results.npy', res, allow_pickle=True)


if __name__ == '__main__':
    if sys.argv[1] == 'fit':
        stage_fit(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == 'analyze':
        stage_analyze()
