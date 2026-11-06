"""Figure 10 (v1.6): field and material scan of the two-hypothesis selection. Reads v16_results.npy."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_v16 import CONFIGS, MASSES

R = np.load('v16_results.npy', allow_pickle=True).item()
FIELD = ['B02', 'B05', 'B1', 'B2', 'R']; MAT = ['M025', 'M05', 'R']
Bs = [CONFIGS[t][0] for t in FIELD]; fs = [CONFIGS[t][1] for t in MAT]
cols = dict(zip(MASSES, ('C0', 'C1', 'C2', 'C3')))
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
for m in MASSES:
    ax[0].plot(Bs, [R[t][('deg', m)]['deg'] for t in FIELD], 'o-', c=cols[m], label=f'{m*1000:.0f} MeV')
    ax[0].plot(Bs, [R[t][('deg_nomass', m)] for t in FIELD], 'o:', c=cols[m], alpha=0.5, ms=3)
    ax[1].plot(fs, [R[t][('deg', m)]['deg'] for t in MAT], 's-', c=cols[m], label=f'{m*1000:.0f} MeV')
    ax[1].plot([0.5], [R['C'][('deg', m)]['deg']], '*', c=cols[m], ms=11)
ax[0].plot([], [], 'k:', alpha=0.5, label='dotted: no mass window')
ax[1].plot([], [], 'k*', ms=9, label='star: C (2.0 T, f = 0.5)')
for x, lo, hi, t in ((0.5, 40, 90, 'P1'), (0.25, 20, 63, 'P2')):
    ax[1].fill_between([x - 0.02, x + 0.02], lo, hi, color='C0', alpha=0.2, lw=0); ax[1].text(x + 0.025, hi, t, fontsize=7, color='C0')
ax[0].set_xscale('log'); ax[0].set_xticks(Bs); ax[0].set_xticklabels([str(b) for b in Bs]); ax[0].minorticks_off()
ax[0].set_xlabel('B [T]  (f = 1)'); ax[1].set_xlabel('material scale factor f  (3.8 T)')
for a, t in ((ax[0], '(a) degradation vs field'), (ax[1], '(b) degradation vs material')):
    a.set_yscale('log'); a.set_ylim(1, 400); a.set_ylabel(r'limit degradation from conversions (c$\tau$ = 1 mm)')
    a.legend(fontsize=7); a.grid(alpha=0.25); a.set_title(t, fontsize=10)
ax[2].plot(Bs, [100 * R[t]['conv_f9'] for t in FIELD], 'ko-', label=r'conversions: $\Delta\chi^2>9$')
for m in (0.017, 0.2):
    ax[2].plot(Bs, [100 * R[t][('sig_f9', m)] for t in FIELD], 'o-', c=cols[m], label=fr'ALP {m*1000:.0f} MeV: $\Delta\chi^2>9$')
ax[2].plot(Bs, [100 * R[t][('curv_res', 0.017)] for t in FIELD], 'D--', c='gray', label='curvature resolution, 17 MeV signal (MAD)')
ax[2].set_xscale('log'); ax[2].set_xticks(Bs); ax[2].set_xticklabels([str(b) for b in Bs]); ax[2].minorticks_off()
ax[2].set_xlabel('B [T]'); ax[2].set_ylabel('%'); ax[2].legend(fontsize=7); ax[2].grid(alpha=0.25)
ax[2].set_title('(c) mechanism vs field', fontsize=10)
fig.suptitle('Fig. 10  v1.6 (preregistered, doi:10.5281/zenodo.23942338): field and material scan (toy MC, subsample)', fontsize=10)
plt.tight_layout(); plt.savefig('fig10_field_material_scan.png', dpi=150)
print('ok')
