"""Figure 11 (v1.7): three-dimensional two-hypothesis fit. Reads v17_results.npy (run_v17.py analyze)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import chi2
from run_v16 import MASSES

R = np.load('v17_results.npy', allow_pickle=True).item()
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
k = np.logspace(-3, 3, 300)
def surv(d):
    d = np.sort(d[np.isfinite(d)]); return 1 - np.searchsorted(d, k, side='right') / len(d)
ax[0].plot(k, surv(R[('dl_conv', 'default')]), 'k', lw=2, label='conversions (r < 800 mm)')
for m, c in zip(MASSES, ('C0', 'C1', 'C2', 'C3')):
    ax[0].plot(k, surv(R[('dl_sig', 'default', m)]), c=c, label=f'ALP {m*1000:.0f} MeV')
ax[0].plot(k, chi2.sf(k, 1), 'k:', label=r'$\chi^2_1$ (preregistered null reference)')
ax[0].axvline(9, c='gray', ls='--', lw=0.8)
ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_ylim(1e-4, 1.5)
ax[0].set_xlabel(r'threshold $k$ on $\Delta\chi^2_L$ (r–z)'); ax[0].set_ylabel(r'fraction with $\Delta\chi^2_L>k$')
ax[0].legend(fontsize=7, loc='lower left'); ax[0].set_title('(a) r–z test, pairs with r–z information (default)', fontsize=10)
mm = np.array(MASSES) * 1000
sty = {'T': ('o-', 'C3', r'transverse only ($\Delta\chi^2_T$ = v1.6 R)'), 'L': ('v:', 'C4', r'r–z only ($\Delta\chi^2_L$)'),
       '3D': ('D-', 'C0', '3D, pixel z (default)'), '3D_stereo': ('s-', 'C1', '3D + stereo strips (V-stereo)'),
       '3D_allpixel': ('P-', 'C2', '3D, z at every layer (V-allpixel)')}
for name, (mk, c, lab) in sty.items():
    ax[1].plot(mm, [R[('deg', name, m)]['deg'] for m in MASSES], mk, c=c, label=lab, ms=6)
    ax[2].plot(mm, [R[('deg_nomass', name, m)] for m in MASSES], mk, c=c, label=lab, ms=6)
ax[1].plot(mm, [1.9, 2.0, 1.9, 1.8], 'x-.', c='gray', label='no-brems bound (v1.3)')
for x, lo, hi, t in ((17, 100, 127, 'P1'), (100, 12, 25, 'P3'), (200, 3, 7, 'P2')):
    ax[1].fill_between([x * 0.9, x * 1.1], lo, hi, color='C0', alpha=0.15, lw=0); ax[1].text(x * 1.12, np.sqrt(lo * hi), t, fontsize=7, color='C0')
for a, t in ((ax[1], '(b) degradation (preregistered ranges shaded)'), (ax[2], '(c) without mass window')):
    a.set_xscale('log'); a.set_yscale('log'); a.set_xticks(mm); a.set_xticklabels(['17', '50', '100', '200']); a.minorticks_off()
    a.set_ylim(1, 400); a.set_xlim(13, 280); a.set_xlabel(r'$m_a$ [MeV]'); a.grid(alpha=0.25)
    a.set_ylabel(r'limit degradation from conversions (c$\tau$ = 1 mm)'); a.legend(fontsize=6.5, loc='lower left'); a.set_title(t, fontsize=10)
fig.suptitle('Fig. 11  v1.7 (preregistered, doi:10.5281/zenodo.23995668): three-dimensional two-hypothesis vertex fit (toy MC, 3.8 T, subsample)', fontsize=10)
plt.tight_layout(); plt.savefig('fig11_3d_vertex_fit.png', dpi=150)
print('ok')
