"""v1.2 preregistered analysis: limit degradation with/without hit-pattern requirement."""
import numpy as np, sys
import bkg, hitpattern as hp

MASSES = [0.017, 0.05, 0.1, 0.2]
SCEN = [('fixed', (0.0, 1.0), 'fixed, Gaussian'), ('kinematic', (0.0, 1.0), 'per-pair, Gaussian'),
        ('fixed', (0.1, 3.0), 'fixed, tails'), ('kinematic', (0.1, 3.0), 'per-pair, tails')]


def with_hp(d, t):
    d2 = dict(d); d2['base'] = d['base'] & hp.consistent(d['r'], d['rm'], d['sr'], t); return d2


def run(ctau, ts):
    rows = {}
    for model, tail, lab in SCEN:
        bkg.A['vtx_tail'] = tail
        B = bkg.conversions(model=model)
        for m in MASSES:
            S = bkg.signal(m, ctau, model=model)
            lim0 = bkg.upper_limit(S['w'][S['base']].sum(), 0)        # v1.1 no-conversion reference
            r0 = bkg.scan_v11(m, S, B)
            rows[(lab, m, None)] = (r0['limit'] / lim0, r0['eff'], r0['B'])
            for t in ts:
                r = bkg.scan_v11(m, with_hp(S, t), with_hp(B, t))
                rows[(lab, m, t)] = (r['limit'] / lim0, r['eff'], r['B'])
    bkg.A['vtx_tail'] = (0.0, 1.0)
    return rows


if __name__ == "__main__":
    out = {}
    for ctau, ts in [(1.0, [0.0, 1.0, 2.0]), (0.1, [1.0]), (10.0, [1.0])]:
        rows = run(ctau, ts); out[ctau] = rows
        print(f'\n=== ctau = {ctau} mm ===   degradation (sig.eff%, B)')
        for _, _, lab in SCEN:
            for m in MASSES:
                s = f'{lab:20s} {m*1000:5.0f} MeV  noHP {rows[(lab,m,None)][0]:7.1f}'
                for t in ts:
                    d, e, b = rows[(lab, m, t)]; s += f' | t={t:g}: {d:7.1f} ({e:4.1f}%, B={b:6.1f})'
                print(s)
    np.save('v12_results.npy', out, allow_pickle=True)
