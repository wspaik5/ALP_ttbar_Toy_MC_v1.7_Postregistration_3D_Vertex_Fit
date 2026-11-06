"""Figure 2: does the 3.8 T solenoid separate collimated e+e- pairs? (barrel, |eta|<1.2)"""
import numpy as np, warnings, matplotlib; matplotlib.use('Agg'); warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import mag

masses = [0.017, 0.025, 0.04, 0.05, 0.06, 0.1, 0.2, 0.3, 0.5, 1.0]
ctaus = [0.1, 1, 10]; cols = {0.1: '#2a6fdb', 1: '#16a085', 10: '#e67e22'}
tab = {}
print('m_a[GeV] ctau[mm]  resolved%(straight)  resolved%(3.8T)')
for c in ctaus:
    for m in masses:
        r, _, _ = mag.analyse(m, c, n=60000)
        tab[(m, c)] = (r['straight']['resolved'], r['field']['resolved'])
        print(f'{m:<8} {c:<8} {tab[(m,c)][0]:6.1f} {tab[(m,c)][1]:6.1f}')

fig, ax = plt.subplots(1, 3, figsize=(18, 6))
for m, col in [(0.017, '#c0392b'), (0.1, '#8e44ad')]:
    r, _, _ = mag.analyse(m, 1.0, n=60000)
    for tag, ls, lab in [('straight', '--', 'straight lines (paper)'), ('field', '-', '3.8 T field')]:
        ax[0].plot(mag.LAYERS/10, np.nanmedian(r[tag]['sep'], axis=0), ls, marker='o', color=col, label=f'{m*1000:g} MeV, {lab}')
ax[0].step(mag.LAYERS/10, mag.RES, where='mid', color='k', lw=2, label='assumed two-hit resolution')
ax[0].set_yscale('log'); ax[0].set_xlabel('barrel layer radius [cm]'); ax[0].set_ylabel('e⁺e⁻ hit separation [mm] (median)')
ax[0].axvspan(0, 20, color='#ddd', alpha=.5); ax[0].set_title('(a) hit separation vs. radius (cτ = 1 mm)'); ax[0].legend(fontsize=9)
for c in ctaus:
    ax[1].plot(masses, [tab[(m, c)][0] for m in masses], '--o', color=cols[c], alpha=.7)
    ax[1].plot(masses, [tab[(m, c)][1] for m in masses], '-o', color=cols[c], label=f'cτ = {c:g} mm')
ax[1].plot([], [], 'k--', label='dashed: straight lines'); ax[1].plot([], [], 'k-', label='solid: 3.8 T field')
ax[1].set_xscale('log'); ax[1].set_ylim(0, 105); ax[1].axvline(0.211, color='red', ls=':')
ax[1].set_xlabel('$m_a$ [GeV]'); ax[1].set_ylabel('pairs resolved in ≥3 layers [%]')
ax[1].set_title('(b) fraction of resolvable pairs'); ax[1].legend(fontsize=9, loc='center right', bbox_to_anchor=(1, .45))
vtx, em, ep = mag.make_events(0.017, 1.0, 20000, seed=3)
lxy = np.hypot(vtx[:, 0], vtx[:, 1]); i = np.where((lxy > 40) & (lxy < 80))[0][0]
for tag, bf, ls in [('straight', False, '--'), ('3.8 T', True, '-')]:
    h1 = mag.hits(vtx[i:i+1], em[i:i+1], -1, bf)[0]; h2 = mag.hits(vtx[i:i+1], ep[i:i+1], +1, bf)[0]
    mid = (h1+h2)/2; phi = np.arctan2(mid[:, 1], mid[:, 0])
    perp = lambda h: -np.sin(phi)*h[:, 0] + np.cos(phi)*h[:, 1]
    ax[2].plot(mag.LAYERS/10, perp(h1)-perp(mid), ls, marker='o', color='#2a6fdb', label=f'e⁻, {tag}')
    ax[2].plot(mag.LAYERS/10, perp(h2)-perp(mid), ls, marker='s', color='#c0392b', label=f'e⁺, {tag}')
ax[2].axvline(lxy[i]/10, color='gray', ls=':'); ax[2].text(lxy[i]/10+1, -8, '← ALP decay', fontsize=10, color='gray'); ax[2].set_xlim(0, 112)
ax[2].set_xlabel('radius [cm]'); ax[2].set_ylabel('transverse offset from pair axis [mm]')
ax[2].set_title(f'(c) one 17 MeV event (e pT = {np.hypot(*em[i,:2]):.0f}, {np.hypot(*ep[i,:2]):.0f} GeV)'); ax[2].legend(fontsize=9)
fig.suptitle('Effect of the 3.8 T solenoid on collimated displaced $e^+e^-$ pairs (toy MC)', fontsize=15)
plt.tight_layout(rect=[0, 0, 1, .94]); fig.savefig('figures/fig2_bfield_pair_separation.png', dpi=130)
