"""Figure 1: decay-position geometry, boost/opening angle, and pre-selection efficiency vs. the paper's Table 3."""
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from alp_toy import simulate, MASSES, CTAUS, PAPER

KW = dict(pt_scale=80, eta_sigma=1.6)
toy = np.array([[simulate(m, c, n=100_000, **KW)['eff'] for c in CTAUS] for m in MASSES])
r = np.log(toy[PAPER > 0.1]) - np.log(PAPER[PAPER > 0.1]); k = np.exp(r.mean())
np.set_printoptions(precision=2, suppress=True)
print(f'global normalisation factor toy/paper = {k:.2f}'); print('toy / k:\n', toy/k)

fig, ax = plt.subplots(2, 2, figsize=(13, 10.5))
bins = np.logspace(-3, 5, 90)
for c, col in zip([0.1, 1, 10, 100], ['#2a6fdb', '#16a085', '#e67e22', '#c0392b']):
    d = simulate(0.3, c, n=100_000, **KW)
    ax[0, 0].hist(d['lxy'], bins=bins, histtype='step', lw=2, color=col,
                  label=f'cτ = {c:g} mm: {d["in_tracker"]:.0f}% in window')
ax[0, 0].axvspan(0.5, 800, color='gold', alpha=.25, label='0.5 mm < $l_{xy}$ < 0.8 m')
ax[0, 0].set_xscale('log'); ax[0, 0].set_xlabel('transverse decay distance $l_{xy}$ [mm]'); ax[0, 0].set_ylabel('events')
ax[0, 0].set_title('(a) $m_a$ = 0.3 GeV: where does the ALP decay?'); ax[0, 0].legend(fontsize=9, loc='upper left')

ms = np.logspace(np.log10(0.017), 1, 25); bg, ang = [], []
for m in ms:
    d = simulate(m, 1, n=40_000, **KW); bg.append(np.median(d['bg'])); ang.append(np.median(d['angle']))
a2 = ax[0, 1]; a2.loglog(ms, np.degrees(ang), 'o-', color='#8e44ad', label='e⁺e⁻ opening angle (median)')
a2.set_xlabel('$m_a$ [GeV]'); a2.set_ylabel('opening angle [deg]', color='#8e44ad')
a3 = a2.twinx(); a3.loglog(ms, bg, 's--', color='#555', label='βγ (median)'); a3.set_ylabel('βγ', color='#555')
a2.axvline(0.211, color='red', ls=':'); a2.text(0.22, 0.05, 'dimuon\nthreshold', color='red', fontsize=9)
a2.set_title('(b) lighter ALPs: larger boost, more collimated pairs')
h1, l1 = a2.get_legend_handles_labels(); h2, l2 = a3.get_legend_handles_labels(); a2.legend(h1+h2, l1+l2, fontsize=9, loc='lower left')

def hm(axx, Z, title):
    im = axx.imshow(Z, origin='lower', cmap='viridis', aspect='auto', vmin=0, vmax=26)
    axx.set_xticks(range(5), [f'{c:g}' for c in CTAUS]); axx.set_yticks(range(10), [f'{m:g}' for m in MASSES])
    axx.set_xlabel('cτ [mm]'); axx.set_ylabel('$m_a$ [GeV]'); axx.set_title(title)
    for i in range(10):
        for j in range(5):
            axx.text(j, i, f'{Z[i, j]:.1f}', ha='center', va='center', fontsize=8, color='w' if Z[i, j] < 13 else 'k')
    return im
hm(ax[1, 0], toy/k, f'(c) toy pre-selection efficiency [%] ÷ {k:.1f}')
im = hm(ax[1, 1], PAPER, '(d) paper, Table 3 [%]')
fig.colorbar(im, cax=fig.add_axes([0.89, 0.11, 0.015, 0.32]), label='pre-selection efficiency [%]')
fig.suptitle('Toy MC of displaced ALP → $e^+e^-$ (after arXiv:2610.38932)', fontsize=15)
plt.subplots_adjust(hspace=.32, wspace=.32, right=.86)
fig.savefig('figures/fig1_preselection_efficiency.png', dpi=130)
