"""Figure 7 (v1.4): brems-aware vertex fits compared with the v1.3 brems-blind fit (mode B) and the no-brems bound."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_v13 import conversions_truth
from run_v14 import MASSES, V13_B, V13_NOBREMS

R = np.load('v14_results.npy', allow_pickle=True).item()
d = conversions_truth()
F = {m: np.load(f'v14_fit_{m}.npz') for m in ('a6', 'b')}
Z = np.load('v14_hits.npz'); idx = Z['conv_idx']; rt = d['r'][idx]
_p = [q for q in ('v13_conv_arrays_B_parallel.npz', '../v13/v13_conv_arrays_B_parallel.npz') if os.path.exists(q)]
V13 = np.load(_p[0]) if _p else None          # v1.3 mode-B arrays (written by run_v13.py parallel)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
bins = np.linspace(-60, 60, 121); pb = np.linspace(-10, 10, 101)
curves = []
if V13 is not None:
    sel = V13['b_base'] & (V13['b_r'] < 800)
    curves.append(('v1.3 mode B (brems-blind)', V13['b_rm'][sel] - V13['b_r'][sel],
                   (V13['b_rm'][sel] - V13['b_r'][sel]) / V13['b_sr'][sel], 'C3', '-'))
for m, lab, col in (('a6', 'v1.4 (a) inner 6 hits', 'C1'), ('b', 'v1.4 (b) brems-noise fit', 'C0')):
    ok = F[m]['conv_ok'] & (rt < 800)
    dr = F[m]['conv_r'][ok] - rt[ok]
    curves.append((lab, dr, dr / F[m]['conv_sigma_r'][ok], col, '-'))
for lab, dr, pull, col, ls in curves:
    ax[0].hist(np.clip(dr, -60, 60), bins, histtype='step', color=col, ls=ls, density=True, label=lab)
    ax[1].hist(np.clip(pull, -10, 10), pb, histtype='step', color=col, ls=ls, density=True, label=lab)
x = np.linspace(-10, 10, 400)
ax[1].plot(x, np.exp(-x**2 / 2) / np.sqrt(2 * np.pi), 'k:', lw=1, label='unit Gaussian')
for a, t in ((ax[0], r'$r_{\rm fit}-r_{\rm true}$ [mm]  (conversions, r < 800 mm)'),
             (ax[1], r'pull $(r_{\rm fit}-r_{\rm true})/\sigma_{\rm rep}$')):
    a.set_yscale('log'); a.set_xlabel(t); a.legend(fontsize=7)
ax[0].set_ylabel('density (outer bins = overflow)'); ax[0].set_title('(a) radial vertex residual')
ax[1].set_ylim(1e-5, 2); ax[1].set_title('(b) pulls')

mm = np.array(MASSES) * 1000
ax[2].plot(mm, [V13_B[m] for m in MASSES], 'o-', c='C3', label='v1.3 mode B (brems-blind)')
ax[2].plot(mm, [R[('deg', 'a6', m)]['deg'] for m in MASSES], 's-', c='C1', label='v1.4 (a) inner 6 hits')
for nk, mk in (('a4', 'v'), ('a8', '^')):
    if ('deg', nk, MASSES[0]) in R:
        ax[2].plot(mm, [R[('deg', nk, m)]['deg'] for m in MASSES], mk + ':', c='C1', alpha=.5, ms=4,
                   label=f'(a) inner {nk[1]} hits (sensitivity)')
ax[2].plot(mm, [R[('deg', 'b', m)]['deg'] for m in MASSES], 'D-', c='C0', label='v1.4 (b) brems-noise fit')
ax[2].plot(mm, [V13_NOBREMS[m] for m in MASSES], 'x-.', c='gray', label='no-brems bound (v1.3, exploratory)')
ax[2].fill_between([14.5, 16.5], 30, 150, color='C1', alpha=.25, label='P1 range (a), 17 MeV')
ax[2].fill_between([17.5, 19.5], 10, 80, color='C0', alpha=.25, label='P2 range (b), 17 MeV')
ax[2].set_xscale('log'); ax[2].set_yscale('log'); ax[2].set_xticks(mm); ax[2].set_xticklabels(['17', '50', '100', '200'])
ax[2].minorticks_off(); ax[2].set_ylim(1, 3000); ax[2].set_xlabel(r'$m_a$ [MeV]')
ax[2].set_ylabel('limit degradation from conversions'); ax[2].legend(fontsize=6, loc='upper right')
ax[2].set_title(r'(c) degradation, c$\tau$ = 1 mm')
fig.suptitle('v1.4 (preregistered): bremsstrahlung-aware vertex fits (toy MC)', fontsize=10)
plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/fig7_brems_aware_fits.png', dpi=150)
for k, v in R.items():
    print(k, v)
