"""Figure 5 (v1.2, preregistered): effect of the hit-pattern consistency requirement (toy MC)."""
import numpy as np, warnings, matplotlib; matplotlib.use('Agg'); warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import bkg, hitpattern as hp
from run_v12 import run, MASSES, SCEN

res = {1.0: run(1.0, [0.0, 1.0, 2.0]), 0.1: run(0.1, [1.0]), 10.0: run(10.0, [1.0])}
print('limit degradation vs no-conversion case; columns: no hit pattern | t=1 (primary)')
for ct in (1.0, 0.1, 10.0):
    print(f'-- ctau = {ct} mm')
    for _, _, lab in SCEN:
        print(f'{lab:20s}', ' '.join(f'{m*1000:4.0f}MeV {res[ct][(lab,m,None)][0]:5.1f}|{res[ct][(lab,m,1.0)][0]:5.1f}' for m in MASSES))
print('-- ctau = 1 mm, tolerance variants t=0/1/2 (per-pair, tails):',
      [round(res[1.0][('per-pair, tails', 0.017, t)][0], 1) for t in (0.0, 1.0, 2.0)])

# exploratory (not preregistered) diagnostic
bkg.A['vtx_tail'] = (0.1, 3.0)
B = bkg.conversions(model='kinematic'); b = B['base']; v = bkg.material_veto(B['rm'], B['sr'], 5.0)
h = hp.consistent(B['r'], B['rm'], B['sr'], 1.0); w = B['w']
frac = w[b & v & ~h].sum() / w[b & v].sum(); dr = np.abs(B['rm'] - B['r'])[b & v]
print(f'exploratory: conversions surviving a 5-sigma veto (per-pair, tails): {frac*100:.0f}% rejected by hit pattern; '
      f'median |r_meas - r_true| = {np.median(dr):.0f} mm')
bkg.A['vtx_tail'] = (0.0, 1.0)

fig, ax = plt.subplots(1, 3, figsize=(19, 6.2))
cols = ['#9ab7e6', '#2a6fdb', '#f2b58a', '#c0392b']; x = np.arange(len(MASSES)); wd = .2
for i, ((_, _, lab), c) in enumerate(zip(SCEN, cols)):
    a = [res[1.0][(lab, m, None)][0] for m in MASSES]; bb = [res[1.0][(lab, m, 1.0)][0] for m in MASSES]
    xx = x + (i - 1.5) * wd
    ax[0].bar(xx, a, wd, color=c, alpha=.35, edgecolor=c)
    ax[0].bar(xx, bb, wd * .55, color=c, label=lab)
    for xi, vv in zip(xx, bb): ax[0].text(xi, vv * 1.08, f'{vv:.0f}' if vv >= 10 else f'{vv:.1f}', ha='center', fontsize=7.5)
ax[0].plot([], [], 's', color='#bbb', alpha=.5, label='wide pale bar: without hit pattern')
ax[0].plot([], [], 's', color='#555', label='narrow bar: with hit pattern (t = 1)')
ax[0].set_yscale('log'); ax[0].axhline(1, color='k', lw=1); ax[0].set_ylim(.7, 300)
ax[0].set_xticks(x, [f'{m*1000:g} MeV' for m in MASSES]); ax[0].set_ylabel('limit degradation vs. no-conversion case')
ax[0].set_title('(a) cτ = 1 mm: hit pattern changes little'); ax[0].legend(fontsize=8, loc='upper right')

lab = 'per-pair, tails'
vals = [res[1.0][(lab, 0.017, None)][0]] + [res[1.0][(lab, 0.017, t)][0] for t in (0.0, 1.0, 2.0)]
ax[1].axhspan(13, 43, color='#16a085', alpha=.18, label='preregistered prediction P1 (13–43×)')
ax[1].bar(range(4), vals, color=['#c0392b', '#e8a598', '#c0392b', '#e8a598'])
for i, vv in enumerate(vals): ax[1].text(i, vv + 1.5, f'{vv:.1f}×', ha='center', fontsize=10)
ax[1].set_xticks(range(4), ['no hit\npattern', 't = 0', 't = 1\n(primary)', 't = 2']); ax[1].set_ylim(0, 80)
ax[1].set_ylabel('limit degradation (17 MeV, per-pair, tails)'); ax[1].set_title('(b) prediction P1 vs. outcome: refuted')
ax[1].legend(fontsize=9, loc='upper right')

for ct, mk in ((0.1, 's'), (1.0, 'o'), (10.0, '^')):
    for lab, c in (('per-pair, Gaussian', '#2a6fdb'), ('per-pair, tails', '#c0392b')):
        ax[2].plot(MASSES, [res[ct][(lab, m, 1.0)][0] for m in MASSES], '-' + mk, color=c, alpha=.85,
                   label=f'{lab}, cτ = {ct:g} mm')
ax[2].set_xscale('log'); ax[2].set_yscale('log'); ax[2].set_xlabel('$m_a$ [GeV]'); ax[2].set_ylim(1, 120)
ax[2].set_ylabel('limit degradation (with hit pattern, t = 1)'); ax[2].set_title('(c) lifetime scan: 17 MeV stays hardest')
ax[2].legend(fontsize=8)
fig.suptitle('v1.2 (preregistered): hit-pattern consistency vs. vertex-resolution tails (toy MC)', fontsize=15)
plt.tight_layout(rect=[0, 0, 1, .93]); fig.savefig('figures/fig5_hit_pattern_preregistered.png', dpi=130)
