"""v1.7 preregistered analysis (doi:10.5281/zenodo.23995668), config R of v1.6 (3.8 T, nominal material).
  python3 run_v17.py fit SAMPLE      SAMPLE in {conv, sig0.017, sig0.05, sig0.1, sig0.2} -> v17_SAMPLE.npz
  python3 run_v17.py analyze         -> sanity gate, scans, P1-P7, v17_results.npy
Deviation (sanity gate 3): both r-z hypotheses use s from the H_conv transverse fit (see lz.dchi2_L and the report);
the preregistered version (each hypothesis its own transverse vertex) is also stored as dL_prereg for the record."""
import sys, time
import numpy as np
import lz, fit2h, bkg
from run_v13 import conversions_truth, signal_truth
from run_v16 import subsample, N_CONV, N_SIG, MASSES
from run_v15 import scan_v15

SAMPLES = ['conv'] + [f'sig{m}' for m in MASSES]
SEEDS = {'conv': (31, 1031)}; SEEDS.update({f'sig{m}': (41, 1041) for m in MASSES})


def load(k):
    if k == 'conv':
        return conversions_truth(), N_CONV
    return signal_truth(float(k[3:]), 1.0), N_SIG


def stage_fit(k):
    d, nmax = load(k); _, use = subsample(d, nmax); t = time.time()
    v, e1, e2 = d['vtx'][use], d['e1'][use], d['e2'][use]
    H1, H2, Z, T1, T2 = lz.sim_pairs(v, e1, e2, *SEEDS[k])
    o = fit2h.fit_pairs_2h(H1, H2, v, e1, e2)
    out = dict(idx=use, r=o['r'], sigma_r=o['sigma_r'], ok=o['ok'], ok_alp=o['ok_alp'], dchi2T=o['dchi2'], r_alp=o['r_alp'])
    for g in lz.GEOM:
        dL, info = lz.dchi2_L(Z[g][0], Z[g][1], T1, T2, e1, e2, o['par_conv'], o['par_conv'], g)
        out[f'dL_{g}'], out[f'info_{g}'] = dL, info
        dP, infoP = lz.dchi2_L(Z[g][0], Z[g][1], T1, T2, e1, e2, o['par_conv'], o['par_alp'], g)
        out[f'dLprereg_{g}'], out[f'infoprereg_{g}'] = dP, infoP
    np.savez_compressed(f'v17_{k}.npz', **out)
    print(f'{k}: {len(use)} pairs, {time.time()-t:.0f} s, ok {100*o["ok"].mean():.2f}%, '
          f'info default/stereo/allpixel {100*out["info_default"].mean():.1f}/{100*out["info_stereo"].mean():.1f}/'
          f'{100*out["info_allpixel"].mean():.1f}%', flush=True)


def full(d, F, nmax, arr):
    S, use = subsample(d, nmax); assert np.array_equal(use, F['idx'])
    a = np.full(len(d['r']), -np.inf); a[use] = np.where(F['ok_alp'], arr, -np.inf)
    return a


def attach(d, F, nmax):
    S, use = subsample(d, nmax)
    rm = np.full(len(d['r']), -1.0); sr = np.ones(len(d['r'])); ok = np.zeros(len(d['r']), bool)
    rm[use] = np.nan_to_num(F['r'], nan=-1.0); sr[use] = np.nan_to_num(F['sigma_r'], nan=1.0); ok[use] = F['ok']
    S['rm'], S['sr'], S['base'] = rm, sr, S['base'] & ok
    return S


def stage_analyze():
    res = {'gate': {}}
    F = {k: np.load(f'v17_{k}.npz') for k in SAMPLES}
    gate_ok = True
    for k in SAMPLES:
        R = np.load(f'../v16/v16_R_{k}.npz'); f = F[k]
        same = (np.isclose(f['r'], R['r'], atol=1e-6, rtol=0) | (np.isnan(f['r']) & np.isnan(R['r']))) & \
               (np.isclose(f['dchi2T'], R['dchi2'], atol=1e-6, rtol=0) | (np.isnan(f['dchi2T']) & np.isnan(R['dchi2'])))
        g = dict(identity=float(same.mean()), ok=float(f['ok'].mean()), ok_alp=float(f['ok_alp'].mean()))
        for geo in lz.GEOM:
            m = f['ok_alp'] & f[f'info_{geo}']
            g[f'neg_{geo}'] = float(np.mean(f[f'dL_{geo}'][m] < -1e-6))
            mp = f['ok_alp'] & f[f'infoprereg_{geo}']
            g[f'negprereg_{geo}'] = float(np.mean(f[f'dLprereg_{geo}'][mp] < -1e-6))
        p = g['identity'] >= 0.999 and g['ok'] >= 0.95 and g['ok_alp'] >= 0.95 and all(g[f'neg_{x}'] <= 0.001 for x in lz.GEOM)
        g['passed'] = p; gate_ok &= p; res['gate'][k] = g
        print(f'gate {k:9s}: identity {100*g["identity"]:.3f}%  ok {100*g["ok"]:.2f}%  ' +
              '  '.join(f'neg[{x}] {100*g["neg_"+x]:.3f}% (prereg def. {100*g["negprereg_"+x]:.2f}%)' for x in lz.GEOM) +
              f'  {"PASS" if p else "FAIL"}', flush=True)
    res['gate_ok'] = gate_ok
    if not gate_ok:
        print('SANITY GATE FAILED'); np.save('v17_results.npy', res, allow_pickle=True); return
    C0, nc = load('conv'); Fc = F['conv']; C = attach(C0, Fc, nc)
    sel_conv = C['base'] & (C['r'] < 800)
    dT_c = full(C0, Fc, nc, Fc['dchi2T'])
    tests = {'T': lambda f, g: f['dchi2T'], 'L': lambda f, g: f['dL_default'],
             '3D': lambda f, g: f['dchi2T'] + f['dL_default'], '3D_stereo': lambda f, g: f['dchi2T'] + f['dL_stereo'],
             '3D_allpixel': lambda f, g: f['dchi2T'] + f['dL_allpixel']}
    # mechanism: conversions
    for geo in lz.GEOM:
        info_c = full(C0, Fc, nc, Fc[f'info_{geo}'].astype(float)) > 0
        dl = full(C0, Fc, nc, Fc[f'dL_{geo}'])
        res[('conv_info', geo)] = float(np.mean(info_c[sel_conv]))
        m = sel_conv & info_c
        res[('conv_L9', geo)] = float(np.mean(dl[m] > 9)) if m.any() else np.nan
        res[('dl_conv', geo)] = dl[m]
    for m in MASSES:
        k = f'sig{m}'; S0, ns = load(k); f = F[k]; S = attach(S0, f, ns)
        for geo in lz.GEOM:
            info = full(S0, f, ns, f[f'info_{geo}'].astype(float)) > 0
            dl = full(S0, f, ns, f[f'dL_{geo}'])
            sel = S['base'] & info
            res[('sig_info', geo, m)] = float(np.mean(info[S['base']]))
            res[('sig_L9', geo, m)] = float(np.mean(dl[sel] > 9)) if sel.any() else np.nan
            res[('dl_sig', geo, m)] = dl[sel]
        for name, fn in tests.items():
            dS = full(S0, f, ns, fn(f, None)); dC = full(C0, Fc, nc, fn(Fc, None))
            r1 = scan_v15(m, S, C, dS, dC); r0 = scan_v15(m, S, C, dS, dC, mass_window=False)
            res[('deg', name, m)] = dict(deg=r1['deg'], k=r1['k'], nsig=r1['nsig'], eff=r1['eff'], B=r1['B'])
            res[('deg_nomass', name, m)] = r0['deg']
    print('\ntest           17MeV   50MeV  100MeV  200MeV | no mass window 17 / 200')
    for name in tests:
        print(f'{name:12s} ' + ' '.join(f'{res[("deg", name, m)]["deg"]:7.1f}' for m in MASSES) +
              f' | {res[("deg_nomass", name, 0.017)]:6.1f} {res[("deg_nomass", name, 0.2)]:6.1f}   (k: ' +
              ', '.join(str(res[("deg", name, m)]["k"]) for m in MASSES) + ')')
    print('\nr-z information (fraction of base pairs) and fraction with dchi2_L > 9 among pairs with information:')
    for geo in lz.GEOM:
        print(f'{geo:9s} conv info {100*res[("conv_info", geo)]:5.1f}%  L>9 {100*res[("conv_L9", geo)]:5.2f}% | ' +
              ' | '.join(f'{m*1000:.0f} MeV info {100*res[("sig_info", geo, m)]:5.1f}% L>9 {100*res[("sig_L9", geo, m)]:5.1f}%' for m in MASSES))
    np.save('v17_results.npy', res, allow_pickle=True)


if __name__ == '__main__':
    if sys.argv[1] == 'fit':
        stage_fit(sys.argv[2])
    elif sys.argv[1] == 'analyze':
        stage_analyze()
