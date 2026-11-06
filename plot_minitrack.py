"""Figure 6 (v1.3): vertex resolution and tails from the mini tracking simulation, and the resulting
limit degradation. Reads the outputs of run_v13.py (mode A = as preregistered, B = parallel constraint)."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MASSES = [0.017, 0.05, 0.1, 0.2]
V11_TAILS = [64, 21, 6.8, 3.2]          # v1.1 published: per-pair resolution + 10% tails at 3x


def load(tag):
    res = np.load(f'v13_results_{tag}.npy', allow_pickle=True).item()
    arr = np.load(f'v13_conv_arrays_{tag}.npz')
    return res, arr


A, aA = load('A_prereg')
B, aB = load('B_parallel')

fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
bins = np.linspace(-60, 60, 121)
for arr, lab, ls in ((aA, 'A (as prereg.)', '-'), (aB, 'B (parallel constr.)', '--')):
    for pre, col, bl in (('b', 'C3', 'brems'), ('n', 'C0', 'no brems')):
        sel = arr[f'{pre}_base'] & (arr[f'{pre}_r'] < 160)
        d = (arr[f'{pre}_rm'] - arr[f'{pre}_r'])[sel]
        ax[0].hist(np.clip(d, -60, 60), bins, histtype='step', color=col, ls=ls, density=True,
                   label=f'{lab}, {bl}')
ax[0].set_yscale('log'); ax[0].set_xlabel(r'$r_{\rm fit}-r_{\rm true}$ [mm]  (conversions, pixel region)')
ax[0].set_ylabel('density (outer bins = overflow)'); ax[0].legend(fontsize=7); ax[0].set_title('(a) radial vertex residual')

pb = np.linspace(-10, 10, 101)
for arr, lab, ls in ((aA, 'A', '-'), (aB, 'B', '--')):
    for pre, col, bl in (('b', 'C3', 'brems'), ('n', 'C0', 'no brems')):
        sel = arr[f'{pre}_base'] & (arr[f'{pre}_r'] < 800)
        pull = ((arr[f'{pre}_rm'] - arr[f'{pre}_r']) / arr[f'{pre}_sr'])[sel]
        ax[1].hist(np.clip(pull, -10, 10), pb, histtype='step', color=col, ls=ls, density=True, label=f'{lab}, {bl}')
x = np.linspace(-10, 10, 400)
ax[1].plot(x, np.exp(-x**2 / 2) / np.sqrt(2 * np.pi), 'k:', lw=1, label='unit Gaussian')
ax[1].set_yscale('log'); ax[1].set_ylim(1e-5, 2); ax[1].set_xlabel(r'pull $(r_{\rm fit}-r_{\rm true})/\sigma_{\rm rep}$')
ax[1].legend(fontsize=7); ax[1].set_title('(b) pulls, conversions r < 800 mm')

mm = np.array(MASSES) * 1000
ax[2].plot(mm, [A[('deg', m)]['v11_same_pairs'] for m in MASSES], 'o-', c='gray', label='v1.1 per-pair Gaussian (same pairs)')
ax[2].plot(mm, V11_TAILS, 's:', c='gray', label='v1.1 per-pair + tails (published)')
ax[2].plot(mm, [A[('deg', m)]['primary'] for m in MASSES], 'o-', c='C3', label='v1.3 A: as preregistered')
ax[2].plot(mm, [B[('deg', m)]['primary'] for m in MASSES], 'o--', c='C2', label='v1.3 B: parallel constraint')
ax[2].plot(mm, [A[('deg', m)]['hp'] for m in MASSES], 'v-', c='C3', alpha=.5, ms=4, label='A + hit pattern (t=1)')
ax[2].plot(mm, [B[('deg', m)]['hp'] for m in MASSES], 'v--', c='C2', alpha=.5, ms=4, label='B + hit pattern (t=1)')
import os
if os.path.exists('v13_explore_nobrems_B.npy'):
    X = np.load('v13_explore_nobrems_B.npy', allow_pickle=True).item()
    ax[2].plot(mm, [X[m]['deg'] for m in MASSES], 'D-.', c='C0', ms=4,
               label='B without brems (exploratory bound)')
ax[2].fill_between([14, 20.5], 20, 150, color='C1', alpha=.2, label='P3 preregistered range (17 MeV)')
ax[2].axhline(10, c='C1', lw=.8, ls=':'); ax[2].text(95, 7.5, 'P4: <= 10x at 100-200 MeV', fontsize=7, color='C1')
ax[2].set_xscale('log'); ax[2].set_yscale('log'); ax[2].set_xlabel(r'$m_a$ [MeV]')
ax[2].set_xticks(mm); ax[2].set_xticklabels(['17', '50', '100', '200']); ax[2].minorticks_off()
ax[2].set_ylabel('limit degradation from conversions'); ax[2].legend(fontsize=6, loc='upper right', ncol=1); ax[2].set_ylim(1, 3000)
ax[2].set_title(r'(c) degradation, c$\tau$ = 1 mm')
fig.suptitle('v1.3 (preregistered): vertex resolution and tails from a mini tracking simulation (toy MC)', fontsize=10)
plt.tight_layout()
os.makedirs('figures', exist_ok=True); plt.savefig('figures/fig6_minitrack_vertex_tails.png', dpi=150)

for tag, R in (('A_prereg', A), ('B_parallel', B)):
    print(f'\n=== {tag} ===')
    for k, v in R.items():
        if isinstance(k, tuple):
            print(k, v)
    print('P1', R['P1'], 'conv fit ok', R['conv_fit_ok'])
