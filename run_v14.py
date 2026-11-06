"""v1.4 preregistered analysis. Stages (run in order):
  python3 run_v14.py sim            -> v14_hits.npz   (identical hits to v1.3: same samples, seeds, call sequence)
  python3 run_v14.py fit METHOD     -> v14_fit_METHOD.npz   METHOD in {a6, a4, a8, b}
  python3 run_v14.py analyze        -> sanity gate, metrics, degradations, v14_results.npy
Mode B of v1.3 is not re-fitted: its numbers are those of v1.3 (identical hits); check_hits() verifies identity."""
import sys, time
import numpy as np
import bkg
import brfit as bf
from run_v13 import conversions_truth, signal_truth, metrics, mad

MASSES = [0.017, 0.05, 0.1, 0.2]
CTAU = 1.0
SEED_CONV, SEED_SIG = 31, 41
V13_B = {0.017: 177.86, 0.05: 93.31, 0.1: 42.27, 0.2: 16.12}       # v1.3 mode B (published)
V13_NOBREMS = {0.017: 1.9, 0.05: 2.0, 0.1: 1.9, 0.2: 1.8}           # v1.3 exploratory bound


def samples():
    out = {'conv': conversions_truth()}
    for m in MASSES:
        out[f'sig{m}'] = signal_truth(m, CTAU)
    return out


def stage_sim():
    S = samples(); save = {}
    for k, d in S.items():
        idx = np.where(d['base'])[0]
        t = time.time()
        H1, H2 = bf.simulate_pairs(d['vtx'][idx], d['e1'][idx], d['e2'][idx],
                                   seed=SEED_CONV if k == 'conv' else SEED_SIG, brems=True)
        save[f'{k}_idx'], save[f'{k}_H1'], save[f'{k}_H2'] = idx, H1, H2
        print(f'sim {k}: {len(idx)} pairs, {time.time()-t:.0f} s', flush=True)
    np.savez_compressed('v14_hits.npz', **save)


def check_hits(n=4000):
    """Re-fit the first n conversion pairs with v1.3 mode B and compare with the saved v1.3 output."""
    import os
    c = [q for q in ('v13_conv_arrays_B_parallel.npz', '../v13/v13_conv_arrays_B_parallel.npz') if os.path.exists(q)]
    p = c[0] if c else 'missing.npz'
    if not os.path.exists(p):
        print('check_hits: v1.3 arrays not available'); return None
    v13 = np.load(p); Z = np.load('v14_hits.npz'); d = conversions_truth()
    idx = Z['conv_idx'][:n]
    o = bf.fit_pairs(Z['conv_H1'][:n], Z['conv_H2'][:n], d['vtx'][idx], d['e1'][idx], d['e2'][idx], 'B')
    same = np.allclose(o['r'][o['ok']], v13['b_rm'][idx][o['ok']], rtol=0, atol=1e-6)
    print(f'check_hits: v1.3 mode B reproduced on {n} conversion pairs: {same}', flush=True)
    return same


def stage_fit(method):
    Z = np.load('v14_hits.npz'); S = samples(); save = {}
    meth, nk = ('a', int(method[1])) if method.startswith('a') else (method, 6)
    for k, d in S.items():
        idx = Z[f'{k}_idx']; t = time.time()
        o = bf.fit_pairs(Z[f'{k}_H1'], Z[f'{k}_H2'], d['vtx'][idx], d['e1'][idx], d['e2'][idx], meth, n_keep=nk)
        for f in ('r', 'sigma_r', 'ok'):
            save[f'{k}_{f}'] = o[f]
        print(f'fit {method} {k}: {len(idx)} pairs, {time.time()-t:.0f} s, ok {100*o["ok"].mean():.2f}%', flush=True)
    np.savez_compressed(f'v14_fit_{method}.npz', **save)


def attach(d, F, k):
    """Return a copy of sample d with the measured radius / resolution from fit F (base pairs only)."""
    out = dict(d); idx = np.where(d['base'])[0]
    rm = np.full(len(d['r']), -1.0); sr = np.ones(len(d['r'])); ok = np.zeros(len(d['r']), bool)
    rm[idx] = np.nan_to_num(F[f'{k}_r'], nan=-1.0); sr[idx] = np.nan_to_num(F[f'{k}_sigma_r'], nan=1.0)
    ok[idx] = F[f'{k}_ok']
    out['rm'], out['sr'], out['base'] = rm, sr, d['base'] & ok
    return out


def stage_analyze(methods=('a6', 'b', 'a4', 'a8')):
    S = samples(); res = {'gate': {}}
    fits = {m: np.load(f'v14_fit_{m}.npz') for m in methods}
    # ---- sanity gate (prereg. Sec. 4) before any degradation
    gate_ok = True
    for m, F in fits.items():
        for k in S:
            okf = F[f'{k}_ok'].mean(); msr = np.nanmedian(F[f'{k}_sigma_r'][F[f'{k}_ok']])
            g = (okf >= 0.95) and (0.1 <= msr <= 100)
            res['gate'][(m, k)] = (okf, msr, g); gate_ok &= g
            print(f'gate {m:3s} {k:9s}: fit ok {100*okf:6.2f}%  median sigma_rep {msr:8.3f} mm  {"PASS" if g else "FAIL"}')
    if not gate_ok:
        print('SANITY GATE FAILED: no degradation computed'); np.save('v14_results.npy', res, allow_pickle=True); return
    for m, F in fits.items():
        C = attach(S['conv'], F, 'conv')
        sel = C['base'] & (C['r'] < 800)
        res[('conv', m)] = metrics(C, sel)
        res[('conv_bias', m)] = float(np.median((C['rm'] - C['r'])[sel]))
        for reg, cut in (('pixel', C['r'] < 160), ('strip', (C['r'] >= 160) & (C['r'] < 800))):
            res[('conv', m, reg)] = metrics(C, C['base'] & cut)
        for mass in MASSES:
            Sg = attach(S[f'sig{mass}'], F, f'sig{mass}')
            lim0 = bkg.upper_limit(Sg['w'][Sg['base']].sum(), 0)
            r0 = bkg.scan_v11(mass, Sg, C)
            res[('deg', m, mass)] = dict(deg=r0['limit'] / lim0, nsig=r0['nsig'], eff=r0['eff'], B=r0['B'])
            res[('sig', m, mass)] = metrics(Sg, Sg['base'])
        print(f'\n=== {m} ===  conversions r<800: bias {res[("conv_bias", m)]:+.2f} mm, f3 {100*res[("conv", m)]["f3"]:.1f}%, '
              f'pull width {res[("conv", m)]["pull_width"]:.2f}, core {res[("conv", m)]["sigma_core"]:.2f} mm')
        for mass in MASSES:
            r = res[('deg', m, mass)]
            print(f'  {mass*1000:4.0f} MeV: degradation {r["deg"]:7.1f}x (n_sigma={r["nsig"]}, eff {r["eff"]:.2f}%, B={r["B"]:.1f})'
                  f'   [v1.3 B {V13_B[mass]}x, no-brems bound {V13_NOBREMS[mass]}x]')
    np.save('v14_results.npy', res, allow_pickle=True)


if __name__ == '__main__':
    st = sys.argv[1]
    if st == 'sim':
        stage_sim(); check_hits()
    elif st == 'fit':
        stage_fit(sys.argv[2])
    elif st == 'analyze':
        stage_analyze()
