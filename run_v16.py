"""v1.6 preregistered analysis (doi:10.5281/zenodo.23942338). The configuration is set by the environment variables
V16_B (field, T) and V16_F (material scale factor) before import; defaults reproduce v1.5.
  V16_B=.. V16_F=.. python3 run_v16.py fit TAG      -> v16_TAG_{sample}.npz   (simulation + H_conv/H_ALP fits)
  python3 run_v16.py analyze                        -> sanity gate, scans, P1-P7, v16_results.npy (runs each config in a subprocess)
Subsample (prereg. Sec. 3): first 100,000 base conversion pairs, first 60,000 base signal pairs per mass (whole chunks)."""
import os, sys, time, json, subprocess
import numpy as np

CONFIGS = {'R': (3.8, 1.0), 'B2': (2.0, 1.0), 'B1': (1.0, 1.0), 'B05': (0.5, 1.0), 'B02': (0.2, 1.0),
           'M05': (3.8, 0.5), 'M025': (3.8, 0.25), 'C': (2.0, 0.5)}
MASSES = [0.017, 0.05, 0.1, 0.2]
N_CONV, N_SIG = 100_000, 60_000
SEED_CONV, SEED_SIG = 31, 41


def subsample(d, nmax):
    """First nmax base pairs; base flag cleared elsewhere, weights scaled by n_base / n_used."""
    idx = np.where(d['base'])[0]; use = idx[:nmax]
    out = dict(d); base = np.zeros(len(d['base']), bool); base[use] = True
    out['base'] = base; out['w'] = d['w'] * (len(idx) / len(use))
    return out, use


def stage_fit(tag):
    import brfit as bf, fit2h, mag
    from run_v13 import conversions_truth, signal_truth
    assert (mag.B, mag.F_MAT) == CONFIGS[tag], (mag.B, mag.F_MAT, tag)
    samples = {'conv': (conversions_truth(), N_CONV, SEED_CONV)}
    for m in MASSES:
        samples[f'sig{m}'] = (signal_truth(m, 1.0), N_SIG, SEED_SIG)
    for k, (d, nmax, seed) in samples.items():
        t = time.time()
        _, use = subsample(d, nmax)
        H1, H2 = bf.simulate_pairs(d['vtx'][use], d['e1'][use], d['e2'][use], seed=seed, brems=True)
        o = fit2h.fit_pairs_2h(H1, H2, d['vtx'][use], d['e1'][use], d['e2'][use])
        np.savez_compressed(f'v16_{tag}_{k}.npz', idx=use, **o)
        print(f'{tag} {k}: {len(use)} pairs, {time.time()-t:.0f} s, ok {100*o["ok"].mean():.2f}%, '
              f'ok_alp {100*o["ok_alp"].mean():.2f}%', flush=True)


def analyze_one(tag):
    """Runs inside a subprocess with V16_B/V16_F set: degradations and mechanism metrics for one configuration."""
    import mag, minitrack as mt
    from run_v13 import conversions_truth, signal_truth
    from run_v15 import scan_v15
    import bkg
    assert (mag.B, mag.F_MAT) == CONFIGS[tag]
    res = {'gate': {}}

    def attach(d, F, nmax):
        S, use = subsample(d, nmax)
        assert np.array_equal(use, F['idx'])
        rm = np.full(len(d['r']), -1.0); sr = np.ones(len(d['r'])); ok = np.zeros(len(d['r']), bool)
        dch = np.full(len(d['r']), -np.inf)
        rm[use] = np.nan_to_num(F['r'], nan=-1.0); sr[use] = np.nan_to_num(F['sigma_r'], nan=1.0); ok[use] = F['ok']
        dch[use] = np.where(F['ok_alp'], F['dchi2'], -np.inf)
        S['rm'], S['sr'], S['base'] = rm, sr, S['base'] & ok
        return S, dch

    C0 = conversions_truth(); Fc = np.load(f'v16_{tag}_conv.npz')
    C, dC = attach(C0, Fc, N_CONV)
    for k, F in [('conv', Fc)] + [(f'sig{m}', np.load(f'v16_{tag}_sig{m}.npz')) for m in MASSES]:
        d = F['dchi2'][F['ok_alp']]
        res['gate'][k] = dict(ok=float(F['ok'].mean()), ok_alp=float(F['ok_alp'].mean()), neg=float(np.mean(d < -0.1)),
                              median_sr=float(np.nanmedian(F['sigma_r'][F['ok']])))
    sel = C['base'] & (C['r'] < 800)
    res['conv_f9'] = float(np.mean(dC[sel] > 9)); res['n_conv_base'] = int(C['base'].sum())
    res['conv_sigma_core'] = float(1.4826 * np.median(np.abs((C['rm'] - C['r'])[sel] - np.median((C['rm'] - C['r'])[sel]))))
    for m in MASSES:
        F = np.load(f'v16_{tag}_sig{m}.npz'); S0 = signal_truth(m, 1.0)
        S, dS = attach(S0, F, N_SIG)
        r15 = scan_v15(m, S, C, dS, dC)
        n15 = scan_v15(m, S, C, dS, dC, mass_window=False)
        res[('deg', m)] = dict(deg=r15['deg'], k=r15['k'], nsig=r15['nsig'], eff=r15['eff'], B=r15['B'])
        res[('deg_nomass', m)] = n15['deg']
        res[('sig_f9', m)] = float(np.mean(dS[S['base']] > 9))
        # P6: core relative curvature resolution of signal tracks from the H_ALP fit
        use = F['idx']; okk = F['ok_alp']
        pt1 = np.hypot(S0['e1'][use, 0], S0['e1'][use, 1]); pt2 = np.hypot(S0['e2'][use, 0], S0['e2'][use, 1])
        rel = np.concatenate([F['om1_alp'][okk] / mt.omega_of(-1.0, pt1[okk]) - 1, F['om2_alp'][okk] / mt.omega_of(1.0, pt2[okk]) - 1])
        rel = rel[np.isfinite(rel)]
        res[('curv_res', m)] = float(1.4826 * np.median(np.abs(rel - np.median(rel))))
    return res


def stage_analyze():
    import pickle
    out = {}
    for tag, (B, f) in CONFIGS.items():
        env = dict(os.environ, V16_B=str(B), V16_F=str(f), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        code = (f"import pickle,run_v16; r=run_v16.analyze_one('{tag}'); pickle.dump(r, open('v16_res_{tag}.pkl','wb'))")
        subprocess.run([sys.executable, '-W', 'ignore', '-c', code], env=env, check=True)
        out[tag] = pickle.load(open(f'v16_res_{tag}.pkl', 'rb'))
    # ---- sanity gate (prereg. Sec. 4)
    gate_ok = True
    F15 = {k: np.load(f'../v15/v15_fit_2h_{k}.npz') for k in ['conv'] + [f'sig{m}' for m in MASSES]}
    for k, F15k in F15.items():
        FR = np.load(f'v16_R_{k}.npz'); n = len(FR['r'])
        same_r = np.isclose(FR['r'], F15k['r'][:n], rtol=0, atol=1e-6) | (np.isnan(FR['r']) & np.isnan(F15k['r'][:n]))
        a, b = FR['dchi2'], F15k['dchi2'][:n]
        same_d = np.isclose(a, b, rtol=0, atol=1e-6) | (np.isnan(a) & np.isnan(b))
        frac = float(np.mean(same_r & same_d)); g = frac >= 0.999; gate_ok &= g
        print(f'gate identity R {k:9s}: {100*frac:.3f}%  {"PASS" if g else "FAIL"}')
    for tag, r in out.items():
        for k, g in r['gate'].items():
            p = g['ok'] >= 0.95 and g['ok_alp'] >= 0.95 and g['neg'] <= 0.01; gate_ok &= p
            print(f'gate {tag:4s} {k:9s}: ok {100*g["ok"]:.2f}%  ok_alp {100*g["ok_alp"]:.2f}%  dchi2<-0.1 {100*g["neg"]:.2f}%  '
                  f'median sigma_rep {g["median_sr"]:.2f} mm  {"PASS" if p else "FAIL"}')
    out['gate_ok'] = gate_ok
    print('\nSANITY GATE', 'PASSED' if gate_ok else 'FAILED')
    print('\nconfig  B    f     17MeV   50MeV  100MeV  200MeV | nomass 17/200 | conv f9  ALP17 f9  ALP200 f9 | curv res 17MeV | conv core')
    for tag, (B, f) in CONFIGS.items():
        r = out[tag]
        print(f'{tag:5s} {B:4.1f} {f:5.2f} ' + ' '.join(f'{r[("deg", m)]["deg"]:7.1f}' for m in MASSES) +
              f' | {r[("deg_nomass", .017)]:6.1f} {r[("deg_nomass", .2)]:6.1f} | {100*r["conv_f9"]:6.1f}%  {100*r[("sig_f9", .017)]:6.1f}%  '
              f'{100*r[("sig_f9", .2)]:6.1f}% | {100*r[("curv_res", .017)]:7.2f}% | {r["conv_sigma_core"]:7.2f} mm')
    np.save('v16_results.npy', out, allow_pickle=True)


if __name__ == '__main__':
    if sys.argv[1] == 'fit':
        stage_fit(sys.argv[2])
    elif sys.argv[1] == 'analyze':
        stage_analyze()
