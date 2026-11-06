"""v1.5 post-hoc diagnosis of the X1 failure (reported as a deviation; nothing in the analysis was changed).
(1) X1 null distribution vs chi2_1 and the boundary mixture 1/2 chi2_0 + 1/2 chi2_1 (Chernoff 1954; Self & Liang 1987).
(2) Profile scan: chi2 increase when the opening (phi2 - phi1) is FIXED at an offset and all other parameters are
    re-fitted, on the same no-brems collinear sample -> flat bottom ~ dphi^4 instead of dphi^2.
Output: x1_boundary.npz, x1_boundary.txt, fig9_x1_boundary_test.png"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import chi2
import minitrack as mt
import brfit as bf
import fit2h
from validate_v14 import v4_sample

r, vtx, p1, p2, h1, h2 = v4_sample()
o = fit2h.fit_pairs_2h(h1, h2, vtx, p1, p2, brems_c=False)
d = o['dchi2'][o['ok_alp']]

# (2) profile scan on the first 300 pairs
n = 300
fo = mt.vertex_fit(h1[:n], h2[:n], vtx[:n], p1[:n], p2[:n], parallel=True, cov_fn=bf.make_cov_b(False),
                   pred_fn=bf.make_pred_b(False), return_state=True)
st = fo['state']; c0 = st['chi2']; P = st['par']
offs = np.array([1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3])
prof = []
for dl in offs:
    rise = []
    for sgn in (+1, -1):
        p = P.copy(); p[:, 4] += sgn * dl
        _, c, _ = mt.gauss_newton(p, [st['h1'], st['h2']], [st['W1'], st['W2']], free=[0, 1, 2, 3, 5],
                                  preds=st['preds'], n_iter=20)
        rise.append(c - c0)
    prof.append(np.median(np.maximum(rise[0], rise[1])))   # opening in the 'visible' direction
prof = np.array(prof)

zero = np.mean(d < 1e-3); pos = d[d >= 1e-3]
lines = [f'X1 sample: n = {len(d)} (v1.3 V4 collinear sample, seed 404, no brems, fit (b) with c_k = 0)',
         f'fraction with dchi2 < 1e-3          : {zero:.3f}   (boundary mixture: 0.500)',
         f'median of non-zero part            : {np.median(pos):.3f}   (chi2_1: 0.455)',
         f'fraction dchi2 > 2.71              : {np.mean(d > 2.71):.4f}  (mixture 0.050, chi2_1 0.100)',
         f'fraction dchi2 > 3.84              : {np.mean(d > 3.84):.4f}  (mixture 0.025, chi2_1 0.050)',
         f'fraction dchi2 < -0.1              : {np.mean(d < -0.1):.4f}',
         'profile scan (300 pairs): offset [mrad] -> median chi2 rise, larger of the two signs']
lines += [f'   {x*1e3:5.2f} -> {y:10.4f}' for x, y in zip(offs, prof)]
sl = np.polyfit(np.log(offs[3:]), np.log(prof[3:]), 1)[0]
lines.append(f'log-log slope of profile rise (offsets >= 1 mrad; below, the rise is within the minimiser tolerance): {sl:.2f}  (regular case: 2; second-order identification: 4)')
open('x1_boundary.txt', 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))
np.savez('x1_boundary.npz', dchi2=d, offs=offs, prof=prof)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
k = np.logspace(-3, 1.5, 300)
ds = np.sort(d)
ax[0].plot(k, 1 - np.searchsorted(ds, k, side='right') / len(ds), c='k', lw=2, label='X1 sample (8000 pairs)')
ax[0].plot(k, chi2.sf(k, 1), 'C3--', label=r'$\chi^2_1$ (preregistered expectation)')
ax[0].plot(k, 0.5 * chi2.sf(k, 1), 'C0:', lw=2, label=r'$\frac{1}{2}\chi^2_0+\frac{1}{2}\chi^2_1$ (boundary)')
ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_ylim(1e-3, 1.2)
ax[0].set_xlabel(r'$k$'); ax[0].set_ylabel(r'fraction with $\Delta\chi^2>k$'); ax[0].legend(fontsize=8)
ax[0].set_title(r'(a) null distribution of $\Delta\chi^2$ (no brems, zero opening)', fontsize=10)
g = prof > 1e-3
ax[1].plot(offs[g] * 1e3, prof[g], 'ko-', label='profile scan (median of 300 pairs)')
ref = prof[3] * (offs / offs[3])**4
ax[1].plot(offs * 1e3, ref, 'C0:', lw=2, label=r'$\propto\Delta\phi^4$')
ax[1].plot(offs * 1e3, prof[3] * (offs / offs[3])**2, 'C3--', label=r'$\propto\Delta\phi^2$ (regular)')
ax[1].set_xscale('log'); ax[1].set_yscale('log'); ax[1].set_xlabel(r'fixed opening offset $\Delta\phi$ [mrad]')
ax[1].set_ylabel(r'$\chi^2$ rise after re-fitting all other parameters'); ax[1].legend(fontsize=8)
ax[1].set_title('(b) profile of the opening angle', fontsize=10)
fig.suptitle('Fig. 9  Why validation X1 failed: the opening angle is identified only at second order (post-hoc diagnosis)',
             fontsize=9.5)
plt.tight_layout(); import os; os.makedirs('figures', exist_ok=True); plt.savefig('figures/fig9_x1_boundary_test.png', dpi=150)
