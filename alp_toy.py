"""Toy Monte Carlo of displaced ALP -> e+e- decays (pp -> ttbar a), inspired by arXiv:2610.38932.
Only the ALP flight + decay geometry is simulated; ALP kinematics are a simple parametrisation."""
import numpy as np

ME = 0.000511  # GeV
MASSES = [0.017, 0.1, 0.2, 0.3, 0.5, 1, 2, 5, 7, 10]
CTAUS = [0.1, 1, 10, 100, 1000]  # mm

# Paper Table 3 pre-selection efficiencies (%), rows=mass, cols=ctau
PAPER = np.array([
    [18.144, 3.997, 0.434, 0.046, 0.016],
    [24.827, 14.278, 2.540, 0.270, 0.034],
    [25.027, 19.456, 4.698, 0.541, 0.058],
    [25.002, 21.655, 6.524, 1.349, 0.086],
    [24.682, 23.614, 9.485, 1.312, 0.438],
    [23.817, 24.840, 14.459, 2.487, 0.718],
    [22.479, 25.120, 19.354, 11.897, 0.539],
    [18.768, 24.268, 23.628, 16.317, 1.253],
    [16.683, 24.194, 23.341, 17.886, 1.740],
    [14.419, 23.424, 23.456, 19.132, 2.206],
])


def sample_alp(n, rng, pt_scale, eta_sigma):
    pt = 20 + rng.exponential(pt_scale, n)          # generator cut pT > 20 GeV
    eta = rng.normal(0, eta_sigma, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    return pt, eta, phi


def simulate(m, ctau, n=200_000, pt_scale=80.0, eta_sigma=1.6, seed=1):
    rng = np.random.default_rng(seed)
    pt, eta, phi = sample_alp(n, rng, pt_scale, eta_sigma)
    p = pt * np.cosh(eta)
    E = np.sqrt(p**2 + m**2)
    bg = p / m                                       # beta*gamma
    L = rng.exponential(bg * ctau)                   # lab decay length [mm]
    lxy = L * pt / p                                 # transverse displacement

    # decay a -> e+ e- isotropic in rest frame, then boost
    pstar = np.sqrt(max(m**2 / 4 - ME**2, 0))
    cth = rng.uniform(-1, 1, n); sth = np.sqrt(1 - cth**2); ph = rng.uniform(0, 2*np.pi, n)
    d = np.stack([sth*np.cos(ph), sth*np.sin(ph), cth], 1)
    Estar = m / 2
    beta_vec = np.stack([pt*np.cos(phi), pt*np.sin(phi), pt*np.sinh(eta)], 1) / E[:, None]
    gam = E / m                                      # numerically safe for tiny masses
    def boost(pv, e):
        bp = (beta_vec*pv).sum(1)
        coef = gam**2/(gam+1)*bp + gam*e
        return pv + coef[:, None]*beta_vec
    p1 = boost(pstar*d, Estar); p2 = boost(-pstar*d, Estar)
    def eta_of(v):
        pt_ = np.hypot(v[:, 0], v[:, 1]); return np.arcsinh(v[:, 2]/pt_)
    e1, e2 = eta_of(p1), eta_of(p2)
    cosang = (p1*p2).sum(1)/np.linalg.norm(p1, axis=1)/np.linalg.norm(p2, axis=1)
    angle = np.arccos(np.clip(cosang, -1, 1))

    gen = (np.abs(e1) < 3.5) & (np.abs(e2) < 3.5)   # generator-level fiducial cut
    sel = gen & (np.abs(e1) <= 2.5) & (np.abs(e2) <= 2.5) & (lxy > 0.5) & (lxy < 800)
    return dict(eff=sel.sum()/gen.sum()*100, lxy=lxy[gen], angle=angle[gen], bg=bg[gen],
                in_tracker=((lxy > 0.5) & (lxy < 800))[gen].mean()*100)


def grid(**kw):
    return np.array([[simulate(m, c, n=60_000, **kw)['eff'] for c in CTAUS] for m in MASSES])


if __name__ == '__main__':
    best = None
    for s in [40, 80, 150, 250, 400]:
        for es in [1.0, 1.6, 2.5]:
            g = grid(pt_scale=s, eta_sigma=es)
            mask = PAPER > 0.1
            r = np.log(g[mask]+1e-3) - np.log(PAPER[mask])
            k = np.exp(r.mean()); score = np.mean((r - r.mean())**2)   # shape-only, free scale
            print(f"pt_scale={s:4} eta_sigma={es}: rms log ratio={np.sqrt(score):.3f}  scale={k:.2f}")
            if best is None or score < best[0]:
                best = (score, s, es)
    print('best', best)
