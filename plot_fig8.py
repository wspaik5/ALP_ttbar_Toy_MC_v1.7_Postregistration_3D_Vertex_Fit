"""Figure 8 (v1.5): two-hypothesis vertex fit. Reads v15_results.npy (run_v15.py analyze)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import chi2 as chi2d
from run_v14 import MASSES, V13_B, V13_NOBREMS

R = np.load('v15_results.npy', allow_pickle=True).item()
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

# (a) survival fraction P(dchi2 > k)
k = np.logspace(-3, 4, 300)
def surv(d):
    d = np.sort(d[np.isfinite(d)]); return 1 - np.searchsorted(d, k, side='right') / len(d)
ax[0].plot(k, surv(R['dchi_conv']), c='k', lw=2, label='conversions (r < 800 mm)')
for m, c in zip(MASSES, ('C0', 'C1', 'C2', 'C3')):
    ax[0].plot(k, surv(R[('dchi_sig', m)]), c=c, label=f'ALP {m*1000:.0f} MeV')
ax[0].plot(k, chi2d.sf(k, 1), 'k:', lw=1, label=r'$\chi^2_1$ (not applicable, see text)')
ax[0].axvline(9, c='gray', lw=0.8, ls='--')
ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_ylim(1e-4, 1.5)
ax[0].set_xlabel(r'threshold $k$ on $\Delta\chi^2=\chi^2_{\rm conv}-\chi^2_{\rm ALP}$')
ax[0].set_ylabel(r'fraction with $\Delta\chi^2>k$'); ax[0].legend(fontsize=7, loc='lower left')
ax[0].set_title(r'(a) $\Delta\chi^2$ survival fractions', fontsize=10)

# (b) degradation vs mass
mm = np.array(MASSES) * 1000
ax[1].plot(mm, [V13_B[m] for m in MASSES], 'o-', c='C3', label='v1.3 mode B (brems-blind)')
ax[1].plot(mm, [R[('v14b', m)]['deg'] for m in MASSES], 'D-', c='C0', label='v1.4 (b) brems-noise fit')
ax[1].plot(mm, [R[('ab', m)]['deg'] for m in MASSES], 's:', c='C1', label='v1.5 (a)+(b), secondary')
ax[1].plot(mm, [R[('v15', m)]['deg'] for m in MASSES], 'P-', c='C2', ms=8, lw=2, label='v1.5 two-hypothesis')
ax[1].plot(mm, [V13_NOBREMS[m] for m in MASSES], 'x-.', c='gray', label='no-brems bound (v1.3)')
for x, lo, hi, t in ((17, 100, 126, 'P1'), (50, 34, 68, 'P4'), (100, 5, 20, 'P2'), (200, 2, 7, 'P3')):
    ax[1].fill_between([x * 0.9, x * 1.1], lo, hi, color='C2', alpha=0.2, lw=0)
    ax[1].text(x * 1.12, np.sqrt(lo * hi), t, fontsize=7, color='C2', va='center')
# (c) V-nomass
ax[2].plot(mm, [R[('v14b_nomass', m)]['deg'] for m in MASSES], 'D--', c='C0', label='v1.4 (b), no mass window')
ax[2].plot(mm, [R[('v15_nomass', m)]['deg'] for m in MASSES], 'P--', c='C2', ms=8, label='v1.5, no mass window')
ax[2].plot(mm, [R[('v14b', m)]['deg'] for m in MASSES], 'D-', c='C0', alpha=0.35, label='v1.4 (b), with mass window')
ax[2].plot(mm, [R[('v15', m)]['deg'] for m in MASSES], 'P-', c='C2', alpha=0.35, label='v1.5, with mass window')
for a, t in ((ax[1], '(b) degradation (preregistered ranges shaded)'), (ax[2], '(c) V-nomass check')):
    a.set_xscale('log'); a.set_yscale('log'); a.set_xticks(mm); a.set_xticklabels(['17', '50', '100', '200'])
    a.minorticks_off(); a.set_ylim(1, 1000); a.set_xlim(13, 280); a.set_xlabel(r'$m_a$ [MeV]')
    a.set_ylabel(r'limit degradation from conversions (c$\tau$ = 1 mm)'); a.legend(fontsize=7, loc='upper right')
    a.set_title(t, fontsize=10); a.grid(alpha=0.25)
fig.suptitle('Fig. 8  v1.5 (preregistered, doi:10.5281/zenodo.23905511): two-hypothesis vertex fit (toy MC)', fontsize=10)
plt.tight_layout(); import os; os.makedirs('figures', exist_ok=True); plt.savefig('figures/fig8_two_hypothesis_fit.png', dpi=150)
print('ok')
