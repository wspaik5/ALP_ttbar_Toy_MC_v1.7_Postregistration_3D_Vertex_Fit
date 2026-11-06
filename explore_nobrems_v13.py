"""v1.3 EXPLORATORY (not preregistered): ideal-electron-fit bound. Same as run_v13.py mode B (parallel constraint)
but with bremsstrahlung switched off in BOTH signal and conversions, i.e. the limit of a perfect brems treatment."""
import numpy as np, sys
sys.argv = ['run_v13.py', 'parallel']
import run_v13 as R, bkg
C = R.apply_minisim(R.conversions_truth(), seed=31, brems=False)
out = {}
for m in R.MASSES:
    S = R.apply_minisim(R.signal_truth(m, R.CTAU), seed=41, brems=False)
    lim0 = bkg.upper_limit(S['w'][S['base']].sum(), 0)
    r0 = bkg.scan_v11(m, S, C); r1 = bkg.scan_v11(m, R.with_hp(S, 1.0), R.with_hp(C, 1.0))
    out[m] = dict(deg=r0['limit']/lim0, hp=r1['limit']/lim0, nsig=r0['nsig'], sig=R.metrics(S, S['base']))
    print(f'  {m*1000:4.0f} MeV  no-brems bound: {r0["limit"]/lim0:6.1f}x (n_sigma={r0["nsig"]}) | +HP {r1["limit"]/lim0:6.1f}x | signal {out[m]["sig"]}', flush=True)
np.save('v13_explore_nobrems_B.npy', out, allow_pickle=True)
