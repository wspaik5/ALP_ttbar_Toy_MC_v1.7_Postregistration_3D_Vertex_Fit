"""v1.3 (preregistered 2026-11-01): mini tracking simulation in the transverse plane.

Simulation : exact helices between cylindrical surfaces, Highland multiple scattering, Bethe-Heitler
             bremsstrahlung, Gaussian r-phi hit smearing (15 um pixel / 35 um strip), 3.8 T.
Fit        : joint two-track vertex fit (x, y, phi1, omega1, phi2, omega2) by Gauss-Newton least squares on
             all r-phi hits, with hit + multiple-scattering covariance; bremsstrahlung NOT modelled in the fit.
Units      : mm, GeV, rad. omega = signed curvature d(phi)/ds [1/mm] = -q * 0.3 B / (1000 pT).
"""
import numpy as np
import mag
import bkg

B = mag.B
SENS = mag.LAYERS.astype(float)                                   # 14 sensor radii
HIT_RES = np.where(SENS < 200, 0.015, 0.035)                      # mm, r-phi
# material: bkg.py list + 0.03 X0 at the three outer TOB layers (prereg. Sec. 2)
MAT_R = np.concatenate([bkg.MAT, [868., 965., 1080.]])
MAT_X0 = np.concatenate([bkg.X0, np.array([0.03, 0.03, 0.03]) * mag.F_MAT])
NL = len(SENS)


def omega_of(q, pt):
    return -q * 0.3 * B / (1000.0 * pt)


def wrap(a):
    return a - 2 * np.pi * np.round(a / (2 * np.pi))


# ----------------------------------------------------------------------------------------- geometry
def to_radius(x0, y0, phi0, om, R, mode='forward'):
    """Arc length s from (x0,y0,phi0,om) to the cylinder of radius R.
    mode='forward': smallest s > 0 (NaN if none); mode='nearest': signed s with smallest |s|."""
    rho = 1.0 / np.abs(om)
    sg = np.sign(om)
    cx, cy = x0 - np.sin(phi0) / om, y0 + np.cos(phi0) / om
    Cn = np.hypot(cx, cy); psi = np.arctan2(cy, cx)
    th0 = np.arctan2(y0 - cy, x0 - cx)
    k = (R**2 - Cn**2 - rho**2) / (2 * rho * Cn)
    ok = np.abs(k) <= 1
    a = np.arccos(np.clip(k, -1, 1))
    best = np.full(np.broadcast(x0, R).shape, np.inf)
    for th in (psi + a, psi - a):
        if mode == 'forward':
            d = np.mod(sg * (th - th0), 2 * np.pi)
            d = np.where(d < 1e-12, 2 * np.pi, d)
            best = np.minimum(best, d)
        else:   # signed: forward preferred, a backward solution only if >10x closer
            d = wrap(sg * (th - th0))
            score = np.where(d >= 0, d, -10 * d)
            bscore = np.where(best >= 0, best, -10 * best)
            best = np.where(score < bscore, d, best)
    s = best * rho
    return np.where(ok, s, np.nan)


def advance(x0, y0, phi0, om, s):
    phi = phi0 + om * s
    x = x0 + (np.sin(phi) - np.sin(phi0)) / om
    y = y0 - (np.cos(phi) - np.cos(phi0)) / om
    return x, y, phi


# ----------------------------------------------------------------------------------------- simulation
def simulate_track(vtx, p3, q, rng, brems=True, ms=True, smear=True, return_path=False):
    """vtx (n,3), p3 (n,3) true momentum at production, q charge (+1/-1 scalar or array).
    Returns phi_meas (n,NL) [rad, NaN if no hit], info dict."""
    n = len(vtx)
    x, y = vtx[:, 0].copy(), vtx[:, 1].copy()
    pt = np.hypot(p3[:, 0], p3[:, 1]); p = np.linalg.norm(p3, axis=1)
    ch = p / pt                                                     # cosh(eta)
    phi = np.arctan2(p3[:, 1], p3[:, 0])
    q = np.broadcast_to(np.asarray(q, float), (n,))
    r0 = np.hypot(x, y)
    alive = np.ones(n, bool)
    # ordered surfaces: (radius, is_sensor index or -1, X0 or 0); at equal radius: hit then material
    surf = sorted([(r, i, 0.0) for i, r in enumerate(SENS)] + [(r, -1, t) for r, t in zip(MAT_R, MAT_X0)],
                  key=lambda s: (s[0], s[1] < 0))
    hits = np.full((n, NL), np.nan)
    ploss = np.ones(n)
    if return_path:                     # v1.7: transverse path length and momentum along the simulated track
        scum = np.zeros(n); s_sens = np.full((n, NL), np.nan)
        s_mat = np.full((n, len(MAT_R)), np.nan); p_mat = np.full((n, len(MAT_R)), np.nan)
    imat = -1
    for R, isens, t0 in surf:
        if isens < 0:
            imat += 1                   # index of this material surface in surf order (counted even if inactive)
        act = alive & (R > r0)
        if not act.any():
            continue
        om = omega_of(q, pt)
        s = to_radius(x, y, phi, om, R, 'forward')
        s = np.where(np.abs(np.hypot(x, y) - R) < 1e-6, 0.0, s)      # already on this surface
        dead = act & np.isnan(s)
        alive &= ~dead
        act &= ~dead
        sx, sy, sphi = advance(x, y, phi, om, np.where(act, s, 0.0))
        x, y, phi = np.where(act, sx, x), np.where(act, sy, y), np.where(act, sphi, phi)
        if return_path:
            scum = scum + np.where(act, s, 0.0)
            if isens >= 0:
                s_sens[:, isens] = np.where(act, scum, np.nan)
            else:
                s_mat[:, imat] = np.where(act, scum, np.nan); p_mat[:, imat] = np.where(act, p, np.nan)
        if isens >= 0:
            ph = np.arctan2(y, x)
            if smear:
                ph = ph + rng.normal(0, 1, n) * HIT_RES[isens] / R
            hits[:, isens] = np.where(act, ph, np.nan)
        else:
            t = t0 * ch
            if ms:
                th0 = 0.0136 / p * np.sqrt(t) * (1 + 0.038 * np.log(t))
                phi = np.where(act, phi + rng.normal(0, 1, n) * th0 * ch, phi)
            if brems:
                z = np.exp(-rng.gamma(t / np.log(2), 1.0, n))
                z = np.where(act, z, 1.0)
                p, pt = p * z, pt * z
                ploss *= z
    info = dict(ploss=ploss)
    if return_path:
        info.update(s_sens=s_sens, s_mat=s_mat, p_mat=p_mat, t_mat=np.array([t for _, i, t in surf if i < 0]), cosh=ch, cot=p3[:, 2] / np.hypot(p3[:, 0], p3[:, 1]),
                    z0=vtx[:, 2].copy())
    return hits, info


# ----------------------------------------------------------------------------------------- covariance
def hit_cov(vtx, p3, q):
    """(n,NL,NL) r-phi covariance: hit resolution + multiple-scattering correlations from every material
    surface crossed after the vertex. Lever arms are computed numerically along the TRUE trajectory and the
    scattering angle uses the TRUE momentum (the fit itself never sees the truth otherwise)."""
    n = len(vtx)
    x0, y0 = vtx[:, 0], vtx[:, 1]
    pt = np.hypot(p3[:, 0], p3[:, 1]); p = np.linalg.norm(p3, axis=1); ch = p / pt
    phi0 = np.arctan2(p3[:, 1], p3[:, 0]); om = omega_of(np.broadcast_to(np.asarray(q, float), (n,)), pt)
    r0 = np.hypot(x0, y0)
    C = np.zeros((n, NL, NL))
    C[:, np.arange(NL), np.arange(NL)] = HIT_RES**2
    eps = 1e-6
    for R, t0 in zip(MAT_R, MAT_X0):
        use = R > r0
        s = to_radius(x0, y0, phi0, om, R, 'forward')
        use &= ~np.isnan(s)
        xk, yk, fk = advance(x0, y0, phi0, om, np.nan_to_num(s))
        later = SENS[None, :] > R + 1e-6
        def hit_phi(f):
            sj = to_radius(xk[:, None], yk[:, None], f[:, None], om[:, None], SENS[None, :], 'forward')
            xj, yj, _ = advance(xk[:, None], yk[:, None], f[:, None], om[:, None], np.nan_to_num(sj))
            return np.arctan2(yj, xj)
        lev = SENS[None, :] * wrap(hit_phi(fk + eps) - hit_phi(fk)) / eps
        lev = np.where(later & use[:, None] & np.isfinite(lev), lev, 0.0)
        t = t0 * ch
        th2 = (0.0136 / p * np.sqrt(t) * (1 + 0.038 * np.log(t)) * ch)**2
        C += th2[:, None, None] * lev[:, :, None] * lev[:, None, :]
    return C


def weight(C, mask):
    """Inverse covariance restricted to measured layers (mask (n,NL))."""
    n = len(C)
    Cm = np.where(mask[:, :, None] & mask[:, None, :], C, 0.0)
    Cm[:, np.arange(NL), np.arange(NL)] = np.where(mask, Cm[:, np.arange(NL), np.arange(NL)], 1.0)
    W = np.linalg.inv(Cm)
    return W * (mask[:, :, None] & mask[:, None, :])


# ----------------------------------------------------------------------------------------- fits
def predict(x0, y0, phi0, om):
    """Predicted hit azimuth at every sensor (n,NL) for tracks starting at (x0,y0) (nearest intersection)."""
    s = to_radius(x0[:, None], y0[:, None], phi0[:, None], om[:, None], SENS[None, :], 'nearest')
    x, y, _ = advance(x0[:, None], y0[:, None], phi0[:, None], om[:, None], np.nan_to_num(s))
    return np.where(np.isnan(s), np.nan, np.arctan2(y, x))


def residuals(par, hits_list, ntrk, preds=None):
    """par (n, 2 + 2*ntrk): x, y, then (phi, omega) per track. Returns (n, ntrk*NL) r-phi residuals.
    preds (v1.4): optional list of per-track predictor functions replacing the single-helix predict()."""
    out = []
    for i in range(ntrk):
        pf = predict if preds is None else preds[i]
        ph = pf(par[:, 0], par[:, 1], par[:, 2 + 2 * i], par[:, 3 + 2 * i])
        r = SENS[None, :] * wrap(hits_list[i] - ph)
        out.append(r)
    return np.concatenate(out, 1)


def gauss_newton(par0, hits_list, W_list, n_iter=12, free=None, tie=None, preds=None):
    """Batch Gauss-Newton with step halving. Returns par, chi2, covariance (n,npar,npar), converged flag."""
    ntrk = len(hits_list)
    W = np.zeros((len(par0), ntrk * NL, ntrk * NL))
    for i, Wi in enumerate(W_list):
        W[:, i * NL:(i + 1) * NL, i * NL:(i + 1) * NL] = Wi
    mask = np.concatenate([~np.isnan(h) for h in hits_list], 1)
    scale = np.array([1e-3, 1e-3] + [1e-7, 1e-9] * ntrk)            # finite-difference steps

    def chi(par):
        if tie is not None:                                         # e.g. (4, 2): par[4] := par[2]
            par = par.copy(); par[:, tie[0]] = par[:, tie[1]]
        r = np.where(mask, residuals(par, hits_list, ntrk, preds), 0.0)
        bad = np.isnan(r).any(1)
        r = np.nan_to_num(r)
        c = (r[:, None, :] @ W @ r[:, :, None])[:, 0, 0]
        return r, np.where(bad, np.inf, c)

    par = par0.copy()
    free = list(range(par.shape[1])) if free is None else list(free)
    r, c = chi(par)
    H = None
    for _ in range(n_iter):
        J = np.empty(r.shape + (len(free),))
        for j, k in enumerate(free):
            dp = par.copy(); dp[:, k] += scale[k]
            rk, _ = chi(dp)
            J[:, :, j] = (rk - r) / scale[k]
        JW = J.transpose(0, 2, 1) @ W
        H = JW @ J
        g = (JW @ r[:, :, None])[:, :, 0]
        # Jacobi scaling: parameters differ in scale by ~1e10 (mm vs 1/mm); solve in scaled variables with a
        # tiny relative (Levenberg-Marquardt-like) damping so no parameter is biased by another's scale
        d = np.sqrt(np.clip(np.diagonal(H, axis1=1, axis2=2), 1e-300, None))
        Hs = H / (d[:, :, None] * d[:, None, :]) + 1e-9 * np.eye(len(free))[None]
        step = np.zeros_like(par)
        step[:, free] = -(np.linalg.solve(Hs, (g / d)[:, :, None])[:, :, 0]) / d
        lam = np.ones(len(par))
        newpar, newc = par + step, None
        for _h in range(6):
            trial = par + lam[:, None] * step
            _, ct = chi(trial)
            better = ct <= c
            if _h == 0:
                newpar, newc = np.where(better[:, None], trial, par), np.where(better, ct, c)
            else:
                upd = better & (newc >= c)
                newpar = np.where(upd[:, None], trial, newpar); newc = np.where(upd, ct, newc)
            lam = np.where(better, lam, lam / 2)
        with np.errstate(invalid='ignore'):
            dchi = c - newc
        par, c = newpar, newc
        r, c = chi(par)
        if np.all(~np.isfinite(dchi) | (dchi < 1e-4)):
            break
    # covariance from the scaled Hessian (pinv of the unscaled H truncated the weakly constrained direction)
    d = np.sqrt(np.clip(np.diagonal(H, axis1=1, axis2=2), 1e-300, None))
    Hs = H / (d[:, :, None] * d[:, None, :])
    with np.errstate(all='ignore'):
        try:
            Hi = np.linalg.inv(Hs + 1e-12 * np.eye(len(free))[None])
        except np.linalg.LinAlgError:
            Hi = np.linalg.pinv(Hs, rcond=1e-14)
    cov = Hi / (d[:, :, None] * d[:, None, :])
    if tie is not None:
        par = par.copy(); par[:, tie[0]] = par[:, tie[1]]
    return par, c, cov


def n_behind(par, i, _unused, hits):
    """Number of measured hits reached only by going BACKWARD from the fitted vertex along track i."""
    s = to_radius(par[:, 0:1], par[:, 1:2], par[:, i:i + 1], par[:, i + 1:i + 2], SENS[None, :], 'nearest')
    return ((s < -1e-3) & ~np.isnan(hits)).sum(1)


def circle_fit(phis):
    """Kasa algebraic circle fit to hits (n,NL) of azimuths at SENS. Returns cx, cy, rho, omega sign."""
    x, y = SENS[None, :] * np.cos(phis), SENS[None, :] * np.sin(phis)
    m = ~np.isnan(phis)
    x, y = np.where(m, x, 0), np.where(m, y, 0)
    A = np.stack([x, y, np.ones_like(x)], 2) * m[:, :, None]
    b = -(x**2 + y**2) * m
    AtA = np.einsum('nik,nil->nkl', A, A); Atb = np.einsum('nik,ni->nk', A, b)
    D, E, F = np.linalg.solve(AtA, Atb[:, :, None])[:, :, 0].T
    cx, cy = -D / 2, -E / 2
    rho = np.sqrt(np.clip(cx**2 + cy**2 - F, 1e-9, None))
    # rotation sense from first and last hit
    idx = np.where(m, np.arange(NL)[None, :], NL)
    i1 = idx.min(1); i2 = np.where(m, np.arange(NL)[None, :], -1).max(1)
    xa, ya = x[np.arange(len(x)), i1] - cx, y[np.arange(len(x)), i1] - cy
    xb, yb = x[np.arange(len(x)), i2] - cx, y[np.arange(len(x)), i2] - cy
    sg = np.sign(xa * yb - ya * xb)
    return cx, cy, rho, sg


def start_params(vx, vy, circ):
    cx, cy, rho, sg = circ
    dx, dy = vx - cx, vy - cy
    d = np.hypot(dx, dy)
    ux, uy = dx / d, dy / d
    phi = np.arctan2(sg * ux, -sg * uy)                             # tangent for rotation sense sg
    return phi, sg / rho


def vertex_fit(hits1, hits2, vtx, e1, e2, parallel=False, cov_fn=None, pred_fn=None, return_state=False):
    """Joint 2-track vertex fit. hits*: (n,NL) measured azimuths (track 1 = e-, track 2 = e+); vtx, e1, e2:
    truth, used ONLY for the scattering covariance (hit_cov). parallel=True adds the conversion-fit constraint
    phi1 = phi2 at the vertex (5 free parameters), as used by CMS for conversion vertices.
    v1.4 hooks: cov_fn(vtx, p3, q) replaces hit_cov; pred_fn(vtx, p3, q) returns a per-track predictor
    (both evaluated on the fitted subset). Defaults reproduce v1.3 exactly.
    Returns dict(x, y, r, sigma_r, chi2, ok)."""
    n = len(hits1)
    m1, m2 = ~np.isnan(hits1), ~np.isnan(hits2)
    ok = (m1.sum(1) >= 3) & (m2.sum(1) >= 3)
    out = dict(x=np.full(n, np.nan), y=np.full(n, np.nan), r=np.full(n, np.nan), sigma_r=np.full(n, np.nan),
               chi2=np.full(n, np.inf), ndf=np.zeros(n), ok=ok.copy())
    if not ok.any():
        return out
    sel = np.where(ok)[0]
    h1, h2 = hits1[sel], hits2[sel]
    cf = hit_cov if cov_fn is None else cov_fn
    W1 = weight(cf(vtx[sel], e1[sel], -1), m1[sel])
    W2 = weight(cf(vtx[sel], e2[sel], +1), m2[sel])
    preds = None if pred_fn is None else [pred_fn(vtx[sel], e1[sel], -1), pred_fn(vtx[sel], e2[sel], +1)]
    c1, c2 = circle_fit(h1), circle_fit(h2)
    # candidate starting vertices: circle-circle intersections and closest-approach point
    dx, dy = c2[0] - c1[0], c2[1] - c1[1]; d = np.hypot(dx, dy)
    ex, ey = dx / d, dy / d
    a = (d**2 + c1[2]**2 - c2[2]**2) / (2 * d)
    hh = np.sqrt(np.clip(c1[2]**2 - a**2, 0, None))
    inter = np.abs(a) <= c1[2]
    # closest approach: points on each circle along the centre line, nearest to each other
    cands = []
    pA = np.stack([c1[0] + c1[2] * ex, c1[1] + c1[2] * ey], 1); pA2 = np.stack([c1[0] - c1[2] * ex, c1[1] - c1[2] * ey], 1)
    qB = np.stack([c2[0] - c2[2] * ex, c2[1] - c2[2] * ey], 1); qB2 = np.stack([c2[0] + c2[2] * ex, c2[1] + c2[2] * ey], 1)
    best_ca, best_dd = None, None
    for P in (pA, pA2):
        for Q in (qB, qB2):
            dd = np.hypot(*(P - Q).T); mid = (P + Q) / 2
            if best_ca is None:
                best_ca, best_dd = mid, dd
            else:
                upd = dd < best_dd
                best_ca = np.where(upd[:, None], mid, best_ca); best_dd = np.where(upd, dd, best_dd)
    cands.append(best_ca)
    for sgn in (+1, -1):
        px = c1[0] + a * ex - sgn * hh * ey; py = c1[1] + a * ey + sgn * hh * ex
        cands.append(np.where(inter[:, None], np.stack([px, py], 1), best_ca))
    best = None
    for v in cands:
        f1, o1 = start_params(v[:, 0], v[:, 1], c1)
        f2, o2 = start_params(v[:, 0], v[:, 1], c2)
        if parallel:
            fm = f1 + wrap(f2 - f1) / 2
            par0 = np.stack([v[:, 0], v[:, 1], fm, o1, fm, o2], 1)
            par, chi2, cov = gauss_newton(par0, [h1, h2], [W1, W2], free=[0, 1, 2, 3, 5], tie=(4, 2), preds=preds)
        else:
            par0 = np.stack([v[:, 0], v[:, 1], f1, o1, f2, o2], 1)
            par, chi2, cov = gauss_newton(par0, [h1, h2], [W1, W2], preds=preds)
        # the two circle intersections give the SAME chi2 (same circles); the physical one has the hits in
        # front of the vertex -> choose by chi2 + 100 x (number of hits lying behind the fitted vertex)
        # residual two-fold ambiguity (both crossings before the first hit, equal chi2): a 1e-3/mm tie-break
        # prefers the crossing closer to the innermost hits (as a vertex fit seeded at the first hits would)
        chi2 = chi2 + 100.0 * (n_behind(par, 2, 0, h1) + n_behind(par, 4, 0, h2)) - 1e-3 * np.hypot(par[:, 0], par[:, 1])
        if best is None:
            best = [par, chi2, cov]
        else:
            upd = chi2 < best[1]
            best = [np.where(upd[:, None], par, best[0]), np.where(upd, chi2, best[1]),
                    np.where(upd[:, None, None], cov, best[2])]
    par, chi2, cov = best
    nb = n_behind(par, 2, 0, h1) + n_behind(par, 4, 0, h2)
    chi2 = chi2 - 100.0 * nb + 1e-3 * np.hypot(par[:, 0], par[:, 1])
    r = np.hypot(par[:, 0], par[:, 1])
    ux, uy = par[:, 0] / r, par[:, 1] / r
    sr2 = ux**2 * cov[:, 0, 0] + 2 * ux * uy * cov[:, 0, 1] + uy**2 * cov[:, 1, 1]
    good = np.isfinite(chi2) & (sr2 > 0)
    out['x'][sel], out['y'][sel], out['r'][sel] = par[:, 0], par[:, 1], r
    out['sigma_r'][sel] = np.sqrt(np.abs(sr2)); out['chi2'][sel] = chi2
    out['ndf'][sel] = m1[sel].sum(1) + m2[sel].sum(1) - (5 if parallel else 6)
    out['nbehind'] = np.zeros(n); out['nbehind'][sel] = nb
    out['ok'][sel] = good
    if return_state:          # v1.5 hook: everything needed to re-fit the same pairs under another hypothesis
        out['state'] = dict(sel=sel, par=par, chi2=chi2, h1=h1, h2=h2, W1=W1, W2=W2, preds=preds)
    return out


def reconstruct_pairs(vtx, e1, e2, seed, brems=True, chunk=20000, parallel=False):
    """Simulate e- (e1) and e+ (e2) from vtx and run the vertex fit. Returns dict of arrays."""
    rng = np.random.default_rng(seed)
    n = len(vtx)
    res = {k: np.full(n, np.nan) for k in ('r', 'sigma_r', 'chi2', 'ndf', 'x', 'y')}
    res['ok'] = np.zeros(n, bool); res['nbehind'] = np.zeros(n)
    for s in range(0, n, chunk):
        sl = slice(s, min(n, s + chunk))
        h1, _ = simulate_track(vtx[sl], e1[sl], -1, rng, brems=brems)
        h2, _ = simulate_track(vtx[sl], e2[sl], +1, rng, brems=brems)
        o = vertex_fit(h1, h2, vtx[sl], e1[sl], e2[sl], parallel=parallel)
        for k in res:
            res[k][sl] = o[k]
    return res
