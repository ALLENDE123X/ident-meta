"""
Tests for Tiny-12 feature extraction.

Reference: pde-selector-implementation-plan.md §13
"""

import sys
import os
import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import extract_tiny12


def test_constant_signal():
    """Test that constant signal produces expected features."""
    # Constant field
    u = np.ones((100, 50)) * 2.0
    dx = 0.1
    dt = 0.05

    phi = extract_tiny12(u, dx, dt)

    # Check shape
    assert phi.shape == (12,), f"Expected shape (12,), got {phi.shape}"

    # Check dx, dt
    assert np.isclose(phi[0], dx), f"dx mismatch: {phi[0]} vs {dx}"
    assert np.isclose(phi[1], dt), f"dt mismatch: {phi[1]} vs {dt}"

    # Constant signal should have near-zero derivatives
    # R_x and R_t should be very small
    # Note: R_xx can be numerically unstable when both u_xx and u_x are near zero
    assert phi[3] < 0.1, f"R_x too large for constant: {phi[3]}"
    assert phi[5] < 0.1, f"R_t too large for constant: {phi[5]}"

    # All features should be finite
    assert np.all(np.isfinite(phi)), f"Non-finite values in features: {phi}"


def test_sine_wave():
    """Test that sine wave produces reasonable features."""
    # Sine wave in space
    nx, nt = 128, 100
    dx = 1.0 / nx
    dt = 0.01
    x = np.linspace(0, 1, nx, endpoint=False)
    t = np.linspace(0, nt * dt, nt)

    u = np.sin(2 * np.pi * x)[None, :] * np.ones((nt, 1))

    phi = extract_tiny12(u, dx, dt)

    # Check shape
    assert phi.shape == (12,), f"Expected shape (12,), got {phi.shape}"

    # Sine wave should have spatial structure
    # R_x should be non-zero
    assert phi[3] > 0.1, f"R_x should be positive for sine wave: {phi[3]}"

    # All features should be finite
    assert np.all(np.isfinite(phi)), f"Non-finite values in features: {phi}"


def test_traveling_wave():
    """Test that traveling wave produces reasonable features."""
    # Traveling wave
    nx, nt = 128, 100
    dx = 1.0 / nx
    dt = 0.01
    x = np.linspace(0, 1, nx, endpoint=False)
    t = np.linspace(0, nt * dt, nt)

    # u(t, x) = sin(2*pi*(x - c*t))
    c = 1.0
    X, T = np.meshgrid(x, t)
    u = np.sin(2 * np.pi * (X - c * T))

    phi = extract_tiny12(u, dx, dt)

    # Check shape
    assert phi.shape == (12,), f"Expected shape (12,), got {phi.shape}"

    # Traveling wave should have temporal activity
    # R_t should be non-zero
    assert phi[5] > 0.1, f"R_t should be positive for traveling wave: {phi[5]}"

    # All features should be finite
    assert np.all(np.isfinite(phi)), f"Non-finite values in features: {phi}"


def test_noisy_signal():
    """Test that noisy signal produces reasonable noise features."""
    # Clean + noise
    nx, nt = 128, 100
    dx = 0.01
    dt = 0.01

    u_clean = np.ones((nt, nx))
    noise = np.random.randn(nt, nx) * 0.1
    u_noisy = u_clean + noise

    phi = extract_tiny12(u_noisy, dx, dt)

    # Check shape
    assert phi.shape == (12,), f"Expected shape (12,), got {phi.shape}"

    # SNR should be finite and reasonable
    snr_db = phi[6]
    assert np.isfinite(snr_db), f"SNR is not finite: {snr_db}"

    # Outlier fraction should be between 0 and 1
    out_frac = phi[7]
    assert 0 <= out_frac <= 1, f"Outlier fraction out of range: {out_frac}"

    # All features should be finite
    assert np.all(np.isfinite(phi)), f"Non-finite values in features: {phi}"


def test_feature_names():
    """Test that we can identify which feature is which."""
    u = np.random.randn(50, 30)
    dx = 0.1
    dt = 0.05

    phi = extract_tiny12(u, dx, dt)

    # Feature order: [dx, dt, A, Rx, Rxx, Rt, snr_db, out_frac,
    #                 k_centroid, slope, w_centroid, rho_per]
    assert phi[0] == dx, "Feature 0 should be dx"
    assert phi[1] == dt, "Feature 1 should be dt"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

