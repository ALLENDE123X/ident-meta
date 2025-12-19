"""
Tiny-12 feature extraction for PDE selector.

Extracts 12 characteristic features from spatiotemporal windows u(t,x)
WITHOUT using any IDENT method outputs (no leakage).

Features:
    A. Sampling/geometry (3): dx, dt, aspect
    B. Derivative difficulty (3): R_x, R_xx, R_t
    C. Noise/outliers (2): SNR_dB, outlier_frac
    D. Spatial spectrum (2): k_centroid, slope
    E. Temporal spectrum (1): w_centroid
    F. Boundary/periodicity (1): rho_per

Reference: pde-selector-implementation-plan.md §2
"""

import numpy as np
from scipy.signal import savgol_filter
from numpy.fft import rfft, rfftfreq

EPS = 1e-12


def _sg(u, wx=7, wt=5, order=3):
    """Apply Savitzky-Golay smoothing in time then space."""
    u1 = savgol_filter(u, window_length=wt, polyorder=order, axis=0, mode="interp")
    u2 = savgol_filter(u1, window_length=wx, polyorder=order, axis=1, mode="interp")
    return u2


def _central_diff(a, h, axis):
    """Central difference approximation."""
    return (np.roll(a, -1, axis) - np.roll(a, 1, axis)) / (2 * h)


def extract_tiny12(u, dx, dt, wx=7, wt=5):
    """
    Extract Tiny-12 feature vector from a spatiotemporal window.

    Args:
        u: np.ndarray of shape (nt, nx), the spatiotemporal field
        dx: float, spatial step
        dt: float, temporal step
        wx: int, SG window size in x (default 7)
        wt: int, SG window size in t (default 5)

    Returns:
        np.ndarray of shape (12,) with features:
        [dx, dt, A, Rx, Rxx, Rt, snr_db, out_frac,
         k_centroid, slope, w_centroid, rho_per]
    """
    u = np.asarray(u, dtype=np.float64)
    nt, nx = u.shape

    # A. Sampling/geometry
    A = (nt * dt) / (nx * dx + EPS)

    # Smooth signal
    ut = _sg(u, wx=wx, wt=wt)
    r = u - ut

    # Derivatives on smoothed signal
    ux = _central_diff(ut, dx, axis=1)
    uxx = _central_diff(ux, dx, axis=1)
    ut_t = _central_diff(ut, dt, axis=0)

    # B. Derivative difficulty (use interior to avoid edge artifacts)
    def nrm(x):
        return np.linalg.norm(x[1:-1, 1:-1])

    Rx = nrm(ux) / (nrm(ut) + EPS)
    Rxx = nrm(uxx) / (nrm(ux) + EPS)
    Rt = nrm(ut_t) / (nrm(ut) + EPS)

    # C. Noise/outliers
    snr_db = 20 * np.log10((np.linalg.norm(ut) + EPS) / (np.linalg.norm(r) + EPS))
    mad = np.median(np.abs(r - np.median(r)))
    sig = 1.4826 * mad + EPS
    out_frac = np.mean(np.abs(r) > 3 * sig)

    # D. Spatial spectrum (FFT in x, averaged over t)
    u_zm = u - u.mean(axis=1, keepdims=True)  # zero-mean across x
    k = rfftfreq(nx, d=dx)
    Pk = np.mean(np.abs(rfft(u_zm, axis=1)) ** 2, axis=0) + EPS
    k_centroid = float(np.sum(k * Pk) / np.sum(Pk))

    # Mid-band slope (10-60% of Nyquist)
    lo = max(1, int(0.10 * len(k)))
    hi = max(lo + 2, int(0.60 * len(k)))
    x = np.log(k[lo:hi] + EPS)
    y = np.log(Pk[lo:hi])
    slope = float(np.polyfit(x, y, 1)[0])

    # E. Temporal spectrum (FFT in t, averaged over x)
    u_zm_t = u - u.mean(axis=0, keepdims=True)  # zero-mean across t
    w = rfftfreq(nt, d=dt)
    Pw = np.mean(np.abs(rfft(u_zm_t, axis=0)) ** 2, axis=1) + EPS
    w_centroid = float(np.sum(w * Pw) / np.sum(Pw))

    # F. Periodicity (correlation between left and right boundaries)
    left, right = u[:, 0], u[:, -1]
    if left.std() < 1e-12 or right.std() < 1e-12:
        rho_per = 0.0
    else:
        rho_per = float(np.corrcoef(left, right)[0, 1])

    return np.array(
        [dx, dt, A, Rx, Rxx, Rt, snr_db, out_frac, k_centroid, slope, w_centroid, rho_per],
        dtype=np.float64,
    )

