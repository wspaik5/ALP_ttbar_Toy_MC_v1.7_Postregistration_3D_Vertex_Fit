"""Effect of the CMS 3.8 T solenoid on displaced, collimated e+e- pairs from ALP decays.
Barrel-only toy: ALP kinematics as in alp_toy.py (pT = 20 + Exp(80) GeV, eta ~ N(0,1.6) restricted to |eta|<1.2)."""
import numpy as np

ME = 0.000511
import os
B = float(os.environ.get('V16_B', '3.8'))      # v1.6: field configurable (default 3.8 T = v1.0-v1.5)
F_MAT = float(os.environ.get('V16_F', '1'))    # v1.6: material scale factor (default 1)
# CMS Phase-1 barrel layer radii [mm] (pixel BPix L1-4, strip TIB L1-4, TOB L1-6)
LAYERS = np.array([29, 68, 109, 160, 255, 339, 418, 498, 608, 692, 780, 868, 965, 1080.])
PIXEL = LAYERS < 200
RES = np.where(PIXEL, 0.2, 0.4)   # two-hit separation needed [mm] (~2x pitch, toy assumption)
N_REQ = 3                          # need >= 3 layers where the two hits are separated


def make_events(m, ctau, n, seed=7, pt_scale=80, eta_sigma=1.6):
    rng = np.random.default_rng(seed)
    pt = 20 + rng.exponential(pt_scale, 3 * n)
    eta = rng.normal(0, eta_sigma, 3 * n)
    keep = np.abs(eta) < 1.2
    pt, eta = pt[keep][:n], eta[keep][:n]
    n = len(pt)
    phi = rng.uniform(0, 2 * np.pi, n)
    p = pt * np.cosh(eta); E = np.sqrt(p**2 + m**2)
    L = rng.exponential(p / m * ctau)
    pa = np.stack([pt*np.cos(phi), pt*np.sin(phi), pt*np.sinh(eta)], 1)
    vtx = pa / p[:, None] * L[:, None]
    pstar = np.sqrt(m**2/4 - ME**2)
    cth = rng.uniform(-1, 1, n); sth = np.sqrt(1-cth**2); ph = rng.uniform(0, 2*np.pi, n)
    d = np.stack([sth*np.cos(ph), sth*np.sin(ph), cth], 1)
    bvec = pa / E[:, None]; gam = E / m
    def boost(pv, e):
        bp = (bvec*pv).sum(1)
        return pv + (gam**2/(gam+1)*bp + gam*e)[:, None]*bvec
    Es = np.sqrt(pstar**2 + ME**2)
    return vtx, boost(pstar*d, Es), boost(-pstar*d, Es)   # e- , e+


def hits(vtx, pe, q, bfield=True):
    """Intersection of each track with each barrel layer. Returns (x,y,z) [n,layers,3]; NaN if not reached."""
    x0, y0, z0 = vtx[:, 0], vtx[:, 1], vtx[:, 2]
    pt = np.hypot(pe[:, 0], pe[:, 1]); phi = np.arctan2(pe[:, 1], pe[:, 0]); cot = pe[:, 2] / pt
    out = np.full((len(pt), len(LAYERS), 3), np.nan)
    r0 = np.hypot(x0, y0)
    for j, r in enumerate(LAYERS):
        if not bfield:
            ux, uy = np.cos(phi), np.sin(phi)
            b = x0*ux + y0*uy; c = x0**2 + y0**2 - r**2
            t = -b + np.sqrt(b**2 - c)                    # forward intersection, transverse path length
            x, y, s = x0 + t*ux, y0 + t*uy, t
        else:
            R = pt / (0.3 * B) * 1000.0                   # radius of curvature [mm]
            sgn = -q                                      # rotation sense (opposite for e+ / e-)
            th0 = phi - sgn*np.pi/2
            cx, cy = x0 - R*np.cos(th0), y0 - R*np.sin(th0)
            Cn = np.hypot(cx, cy); psi = np.arctan2(cy, cx)
            k = (r**2 - Cn**2 - R**2) / (2*R*Cn)
            ok = np.abs(k) <= 1
            a = np.arccos(np.clip(k, -1, 1))
            best = np.full(len(pt), np.inf)
            for th in (psi + a, psi - a):
                dlt = np.mod(sgn*(th - th0), 2*np.pi)
                best = np.minimum(best, dlt)
            best = np.where(ok, best, np.nan)
            th = th0 + sgn*best
            x, y, s = cx + R*np.cos(th), cy + R*np.sin(th), R*best
        z = z0 + s*cot
        reach = (r > r0)
        out[:, j, 0] = np.where(reach, x, np.nan)
        out[:, j, 1] = np.where(reach, y, np.nan)
        out[:, j, 2] = np.where(reach, z, np.nan)
    return out


def analyse(m, ctau=1.0, n=100_000, seed=7):
    vtx, em, ep = make_events(m, ctau, n, seed)
    lxy = np.hypot(vtx[:, 0], vtx[:, 1])
    sel = (lxy > 0.5) & (lxy < 800)
    vtx, em, ep = vtx[sel], em[sel], ep[sel]
    res = {}
    for tag, bf in (('straight', False), ('field', True)):
        h1, h2 = hits(vtx, em, -1, bf), hits(vtx, ep, +1, bf)
        sep = np.linalg.norm(h1 - h2, axis=2)            # [n, layers], NaN where not reached
        both = ~np.isnan(sep)
        nres = ((sep > RES) & both).sum(1)
        res[tag] = dict(sep=sep, resolved=(nres >= N_REQ).mean()*100,
                        reach3=(both.sum(1) >= N_REQ).mean()*100)
    ptmin = np.minimum(np.hypot(em[:, 0], em[:, 1]), np.hypot(ep[:, 0], ep[:, 1]))
    return res, ptmin, sel.mean()*100
