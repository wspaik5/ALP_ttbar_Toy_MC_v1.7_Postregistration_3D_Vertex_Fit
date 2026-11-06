"""Figure 4 (v1.1): how much does the conversion veto depend on the vertex-resolution assumption? (toy, cτ = 1 mm)"""
import numpy as np, warnings, matplotlib; matplotlib.use('Agg'); warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import bkg

CT = 1.0
MASSES = [0.017, 0.05, 0.1, 0.2]
SCEN = [('fixed', (0.0, 1.0), 'v1.0 fixed resolution, Gaussian'),
        ('kinematic', (0.0, 1.0), 'v1.1 kinematic resolution, Gaussian'),
        ('fixed', (0.1, 3.0), 'v1.0 fixed resolution, 10% tails ×3'),
        ('kinematic', (0.1, 3.0), 'v1.1 kinematic resolution, 10% tails ×3')]
COLS = ['#9ab7e6', '#2a6fdb', '#f2b58a', '#c0392b']

# ---- (a) per-pair radial vertex resolution vs mass
bkg.A['vtx_tail'] = (0.0, 1.0)
ma = [0.017, 0.025, 0.04, 0.06, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
res = {'pix': [], 'str': []}
print('median radial vertex resolution [mm] (kinematic model, base cuts, cτ = 1 mm)')
print('m_a[GeV]  pixel-region  strip-region')
for m in ma:
    s = bkg.signal(m, CT, n=100_000, model='kinematic'); b = s['base']
    p, q = np.median(s['sr'][b & (s['r'] < 160)]), np.median(s['sr'][b & (s['r'] >= 160)])
    res['pix'].append(p); res['str'].append(q); print(f'{m:<8}  {p:8.2f}  {q:8.2f}')
Bk = bkg.conversions(model='kinematic'); cb = Bk['base']
cpix, cstr = np.median(Bk['sr'][cb & (Bk['r'] < 160)]), np.median(Bk['sr'][cb & (Bk['r'] >= 160)])
print(f'conversions: pixel {cpix:.2f}  strip {cstr:.2f}')

# ---- (c) limit degradation, best mass window x veto width, per scenario
tab = {}
print('\nlimit degradation vs. no-conversion case (best mass window + per-pair veto; nsig optimised)')
print('scenario | m_a[GeV] | degradation | sig.eff% | B | nsig')
for model, tail, lab in SCEN:
    bkg.A['vtx_tail'] = tail
    B = bkg.conversions(model=model)
    for m in MASSES:
        r = bkg.scan_v11(m, bkg.signal(m, CT, model=model), B)
        tab[(lab, m)] = r
        print(f"{lab:42s} {m:<6} {r['degradation']:7.1f} {r['eff']:6.1f} {r['B']:8.1f} {r['nsig']}")
bkg.A['vtx_tail'] = (0.0, 1.0)
v10 = {0.017: 20.3, 0.05: 10.8, 0.1: 5.4, 0.2: 2.7}   # v1.0 Table 2 (fixed 2-sigma veto), figures/fig3_numbers.txt

# ---- (b) veto trade-off at 17 MeV: signal kept vs conversions kept, scanning nsig
print('\n17 MeV veto trade-off (Gaussian): nsig, signal kept %, conversions kept %')
trade = {}
for model in ('fixed', 'kinematic'):
    B = bkg.conversions(model=model); S = bkg.signal(0.017, CT, model=model)
    xs, ys = [], []
    for ns in [0.25, 0.5, 1, 1.5, 2, 3, 4, 5]:
        vs, vb = bkg.material_veto(S['rm'], S['sr'], ns), bkg.material_veto(B['rm'], B['sr'], ns)
        es = (S['base'] & vs).sum() / S['base'].sum() * 100
        eb = B['w'][B['base'] & vb].sum() / B['w'][B['base']].sum() * 100
        xs.append(es); ys.append(eb); print(f'{model:9s} {ns:4} {es:6.1f} {eb:9.4f}')
    trade[model] = (xs, ys)

fig, ax = plt.subplots(1, 3, figsize=(19, 6.2))
ax[0].plot(ma, res['pix'], 'o-', color='#2a6fdb', label='ALP pairs, vertex in pixel region')
ax[0].plot(ma, res['str'], 's-', color='#c0392b', label='ALP pairs, vertex in strip region')
ax[0].axhline(cpix, color='#2a6fdb', ls=':', label=f'conversions, pixel region ({cpix:.1f} mm)')
ax[0].axhline(cstr, color='#c0392b', ls=':', label=f'conversions, strip region ({cstr:.1f} mm)')
ax[0].axhline(1.0, color='#2a6fdb', ls='--', alpha=.5, label='v1.0 fixed: 1 mm (pixel)')
ax[0].axhline(3.0, color='#c0392b', ls='--', alpha=.5, label='v1.0 fixed: 3 mm (strip)')
ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_xlabel('$m_a$ [GeV]')
ax[0].set_ylabel('median radial vertex resolution [mm]'); ax[0].set_ylim(0.3, 30)
ax[0].set_title('(a) collimated pairs vertex poorly (kinematic model)'); ax[0].legend(fontsize=8.5)

for model, col, mk in (('fixed', '#9ab7e6', 's--'), ('kinematic', '#2a6fdb', 'o-')):
    xs, ys = trade[model]
    ax[1].plot(xs, ys, mk, color=col, label=f"{'v1.0 fixed' if model == 'fixed' else 'v1.1 kinematic'} resolution")
ax[1].set_yscale('log'); ax[1].set_xlim(0, 100)
ax[1].set_xlabel('17 MeV signal kept by veto [%]'); ax[1].set_ylabel('photon conversions kept by veto [%]')
ax[1].set_title('(b) 17 MeV veto trade-off (points: 0.25–5 σ band widening)'); ax[1].legend(fontsize=9)

x = np.arange(len(MASSES)); wd = .17
for i, ((model, tail, lab), col) in enumerate(zip(SCEN, COLS)):
    f = [tab[(lab, m)]['degradation'] for m in MASSES]
    ax[2].bar(x + (i - 1.5) * wd, f, wd, color=col, label=lab)
    for xx, v in zip(x + (i - 1.5) * wd, f): ax[2].text(xx, v * 1.08, f'{v:.0f}' if v >= 10 else f'{v:.1f}', ha='center', fontsize=7.5)
ax[2].scatter(x + 2.5 * wd, [v10[m] for m in MASSES], marker='_', s=300, color='k', zorder=3, label='v1.0 Table 2 (fixed 2σ veto)')
ax[2].set_yscale('log'); ax[2].axhline(1, color='k', lw=1); ax[2].set_ylim(.7, 400)
ax[2].set_xticks(x, [f'{m*1000:g} MeV' for m in MASSES]); ax[2].set_xlabel('$m_a$')
ax[2].set_ylabel('limit degradation vs. no-conversion case'); ax[2].legend(fontsize=8, loc='upper right')
ax[2].set_title('(c) result depends strongly on resolution and tails')
fig.suptitle('v1.1: sensitivity of the photon-conversion veto to the vertex-resolution assumption (toy MC, cτ = 1 mm)', fontsize=15)
plt.tight_layout(rect=[0, 0, 1, .93]); fig.savefig('figures/fig4_vertex_resolution_veto.png', dpi=130)
