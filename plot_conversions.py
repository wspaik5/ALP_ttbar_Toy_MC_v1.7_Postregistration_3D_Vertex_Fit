"""Figure 3: photon conversions (gamma -> e+e- in tracker material) as a background (toy MC, cτ = 1 mm)."""
import numpy as np, warnings, matplotlib; matplotlib.use('Agg'); warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import bkg

B = bkg.conversions(); CT = 1.0
masses = [0.017, 0.05, 0.1, 0.2]; sigs = {m: bkg.signal(m, CT) for m in masses}
fig, ax = plt.subplots(1, 3, figsize=(19, 6.2))
bins = np.linspace(0, 800, 401); cb = B['base']
ax[0].hist(B['rm'][cb], bins=bins, weights=B['w'][cb], histtype='stepfilled', color='#999', alpha=.6, label='photon conversions')
s = sigs[0.017]; cs = s['base']
ax[0].hist(s['rm'][cs], bins=bins, weights=s['w'][cs]*1e-3, histtype='step', lw=2, color='#c0392b', label='17 MeV ALP (σ×B = $10^{-3}$ pb)')
for r in bkg.MAT: ax[0].axvline(r, color='#2a6fdb', lw=.6, alpha=.4)
ax[0].plot([], [], color='#2a6fdb', alpha=.6, label='assumed material layers')
ax[0].set_yscale('log'); ax[0].set_ylim(1e-2, 3e5); ax[0].set_xlabel('measured vertex radius [mm]'); ax[0].set_ylabel('events / 2 mm (500 $fb^{-1}$)')
ax[0].set_title('(a) conversions cluster at material'); ax[0].legend(fontsize=9, loc='upper right')
mb = np.linspace(0, 300, 61)
ax[1].hist(B['m'][cb]*1000, bins=mb, weights=B['w'][cb], density=True, histtype='stepfilled', color='#999', alpha=.6, label='conversions (true mass ≈ 0)')
for m, col in zip(masses, ['#c0392b', '#e67e22', '#16a085', '#2a6fdb']):
    s = sigs[m]; ax[1].hist(s['m'][s['base']]*1000, bins=mb, density=True, histtype='step', lw=2, color=col, label=f'{m*1000:g} MeV ALP')
ax[1].set_xlabel('measured $m_{ee}$ [MeV]'); ax[1].set_ylabel('normalised'); ax[1].set_title('(b) measured mass: 17 MeV overlaps conversions'); ax[1].legend(fontsize=9)
stages = ['baseline (paper-like cuts)', '+ mass window', '+ material veto', '+ mass window + material veto']
cols = ['#999', '#e67e22', '#2a6fdb', '#16a085']; x = np.arange(len(masses)); wd = .2
print('m_a[GeV]  stage  sig.eff%  B(events)  limit[pb]  degradation')
for i, (st, col) in enumerate(zip(stages, cols)):
    fac = []
    for m in masses:
        res, _ = bkg.scan(m, CT, sig=sigs[m], bkg=B)
        f = res[st]['limit']/res['no conversion background']['limit']; fac.append(f)
        print(f"{m:<8} {st:32s} {res[st]['eff']:5.1f} {res[st]['B']:10.1f} {res[st]['limit']:.2e} {f:6.1f}")
    ax[2].bar(x+(i-1.5)*wd, fac, wd, color=col, label=st)
    for xx, f in zip(x+(i-1.5)*wd, fac): ax[2].text(xx, f*1.08, f'{f:.0f}', ha='center', fontsize=8)
ax[2].set_yscale('log'); ax[2].axhline(1, color='k', lw=1); ax[2].set_ylim(.7, 4000)
ax[2].set_xticks(x, [f'{m*1000:g} MeV' for m in masses]); ax[2].set_xlabel('$m_a$')
ax[2].set_ylabel('limit degradation vs. no-conversion case'); ax[2].legend(fontsize=9)
ax[2].set_title('(c) mass window + material veto recover most reach')
fig.suptitle('Photon-conversion background for light displaced ALP → $e^+e^-$ (toy MC, cτ = 1 mm)', fontsize=15)
plt.tight_layout(rect=[0, 0, 1, .93]); fig.savefig('figures/fig3_photon_conversion_background.png', dpi=130)
