"""v1.1: kinematics-dependent radial vertex resolution for an e+e- pair in a 3.8 T solenoid.

v1.0 (bkg.py) used a fixed radial vertex resolution (1 mm in the pixel region, 3 mm otherwise).
For nearly collinear pairs this is optimistic: two almost-parallel tracks only cross at a well-defined
point if they open up, either because of their own opening angle or because the field bends them apart.

Toy model (all numbers are stated assumptions, see VTX):
  * free-vertex term: tracks extrapolated from their first measured layer back to the vertex cross with
    transverse opening angle alpha_T, so the along-pair position error is
        sigma_free = sqrt(2) * sqrt(sigma_hit^2 + (L_gap * sigma_phi)^2) / alpha_T
  * tangent-point term: opposite charges bend apart, the transverse direction difference grows as
    k * s with k = 1/R1 + 1/R2; locating where the two directions coincide gives
        sigma_tan = sqrt(2) * sigma_phi / k
  * the two are combined as independent constraints: sigma_r^-2 = sigma_free^-2 + sigma_tan^-2,
    with a small floor. For photon conversions alpha_T ~ 0, so only the tangent term contributes.
The pair direction is taken as radial (displaced vertices point back to the beam line)."""
import numpy as np
import mag

VTX = dict(
    hit_res_pixel=0.015,   # mm, single-hit r-phi resolution, pixel
    hit_res_strip=0.035,   # mm, single-hit r-phi resolution, strip
    floor=0.05,            # mm, best achievable radial resolution
)


def first_layer(vtx_r):
    """Radius [mm] of the first barrel layer outside each vertex (inf if none)."""
    L = mag.LAYERS
    idx = np.searchsorted(L, vtx_r, side='right')
    return np.where(idx < len(L), L[np.minimum(idx, len(L) - 1)], np.inf)


def sigma_r(vtx_r, e1, e2, sig_phi1, sig_phi2):
    """Per-pair radial vertex resolution [mm].
    vtx_r: true vertex radius [mm]; e1, e2: true momenta [GeV] (n,3); sig_phi*: per-track direction resolution [rad]."""
    r1 = first_layer(vtx_r)
    gap = np.clip(r1 - vtx_r, 0, None)
    pix = r1 < 200
    hit = np.where(pix, VTX['hit_res_pixel'], VTX['hit_res_strip'])
    pt1, pt2 = np.hypot(e1[:, 0], e1[:, 1]), np.hypot(e2[:, 0], e2[:, 1])
    k = 0.3 * mag.B / 1000.0 * (1 / pt1 + 1 / pt2)                    # 1/mm
    dphi = np.arctan2(e1[:, 1], e1[:, 0]) - np.arctan2(e2[:, 1], e2[:, 0])
    alpha = np.abs(np.angle(np.exp(1j * dphi)))                      # transverse opening angle [rad]
    sphi = np.hypot(sig_phi1, sig_phi2)                              # = sqrt(2)*sigma_phi for equal tracks
    pos = np.sqrt(2) * np.sqrt(hit**2 + (gap**2) * (sig_phi1**2 + sig_phi2**2) / 2)
    with np.errstate(divide='ignore'):
        s_free = np.where(alpha > 0, pos / alpha, np.inf)
    s_tan = sphi / k
    s = 1 / np.sqrt(1 / s_free**2 + 1 / s_tan**2)
    return np.hypot(s, VTX['floor']), dict(alpha=alpha, k=k, s_free=s_free, s_tan=s_tan, pixel=pix)
