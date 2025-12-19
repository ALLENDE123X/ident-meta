"""
Data generation for PDE selector: Burgers and KdV simulators.

Provides:
- simulate_burgers(): viscous Burgers equation solver
- simulate_kdv(): Korteweg-de Vries equation solver  
- make_windows(): extract overlapping windows from spatiotemporal fields
- add_noise(): add Gaussian noise to fields

Reference: pde-selector-implementation-plan.md §6
"""

import numpy as np
from scipy.fft import fft, ifft, fftfreq


def simulate_burgers(
    nu=0.01,
    L=1.0,
    T=1.0,
    nx=256,
    nt=200,
    ic_type="sine",
    ic_params=None,
    bc="periodic",
):
    """
    Simulate viscous Burgers equation: u_t + u*u_x = nu*u_xx

    Args:
        nu: viscosity coefficient
        L: spatial domain length [0, L]
        T: temporal domain length [0, T]
        nx: number of spatial points
        nt: number of time steps
        ic_type: initial condition type ("sine", "gaussian", "shock")
        ic_params: dict of parameters for IC
        bc: boundary condition ("periodic")

    Returns:
        u: np.ndarray of shape (nt, nx)
        dx: float, spatial step
        dt: float, temporal step
    """
    if ic_params is None:
        ic_params = {}

    # Grid
    dx = L / nx
    dt = T / nt
    x = np.linspace(0, L, nx, endpoint=False)
    
    # Wavenumbers for spectral method
    k = 2 * np.pi * fftfreq(nx, d=dx)

    # Initial condition
    if ic_type == "sine":
        amp = ic_params.get("amp", 1.0)
        freq = ic_params.get("freq", 2.0)
        u0 = amp * np.sin(2 * np.pi * freq * x / L)
    elif ic_type == "gaussian":
        amp = ic_params.get("amp", 1.0)
        x0 = ic_params.get("x0", L / 2)
        sigma = ic_params.get("sigma", L / 10)
        u0 = amp * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
    elif ic_type == "shock":
        u0 = np.where(x < L / 2, 1.0, 0.0)
    else:
        raise ValueError(f"Unknown ic_type: {ic_type}")

    # Storage
    u = np.zeros((nt, nx))
    u[0, :] = u0

    # Time stepping with splitting (convection explicit, diffusion implicit)
    u_hat = fft(u0)
    
    for n in range(nt - 1):
        # Nonlinear term in physical space
        u_n = ifft(u_hat).real
        u[n + 1, :] = u_n  # store
        
        # Convection: -u*u_x (pseudo-spectral)
        ux_hat = 1j * k * u_hat
        ux = ifft(ux_hat).real
        conv = -u_n * ux
        conv_hat = fft(conv)
        
        # Diffusion: nu*u_xx (exact in Fourier)
        # Combined update: u_hat_new = (u_hat + dt*conv_hat) / (1 + nu*dt*k^2)
        u_hat = (u_hat + dt * conv_hat) / (1 + nu * dt * k ** 2)
    
    # Store final step
    u[-1, :] = ifft(u_hat).real

    return u, dx, dt


def simulate_kdv(
    alpha=1.0,
    beta=0.01,
    L=2.0,
    T=1.0,
    nx=256,
    nt=200,
    ic_type="sech",
    ic_params=None,
    bc="periodic",
):
    """
    Simulate Korteweg-de Vries equation: u_t + alpha*u*u_x + beta*u_xxx = 0

    Args:
        alpha: nonlinear coefficient
        beta: dispersion coefficient
        L: spatial domain length [0, L]
        T: temporal domain length [0, T]
        nx: number of spatial points
        nt: number of time steps
        ic_type: initial condition type ("sech", "sine", "cnoidal")
        ic_params: dict of parameters for IC
        bc: boundary condition ("periodic")

    Returns:
        u: np.ndarray of shape (nt, nx)
        dx: float, spatial step
        dt: float, temporal step
    """
    if ic_params is None:
        ic_params = {}

    # Grid
    dx = L / nx
    dt = T / nt
    x = np.linspace(0, L, nx, endpoint=False)
    
    # Wavenumbers
    k = 2 * np.pi * fftfreq(nx, d=dx)

    # Initial condition
    if ic_type == "sech":
        amp = ic_params.get("amp", 1.0)
        x0 = ic_params.get("x0", L / 2)
        width = ic_params.get("width", 0.1)
        u0 = amp / np.cosh((x - x0) / width) ** 2
    elif ic_type == "sine":
        amp = ic_params.get("amp", 0.5)
        freq = ic_params.get("freq", 2.0)
        u0 = amp * np.sin(2 * np.pi * freq * x / L)
    elif ic_type == "cnoidal":
        # Simplified cnoidal wave approximation
        amp = ic_params.get("amp", 1.0)
        freq = ic_params.get("freq", 2.0)
        u0 = amp * (1 + np.cos(2 * np.pi * freq * x / L))
    else:
        raise ValueError(f"Unknown ic_type: {ic_type}")

    # Storage
    u = np.zeros((nt, nx))
    u[0, :] = u0

    # Time stepping: RK4 with exponential integrator for dispersion
    u_hat = fft(u0)
    
    # Precompute dispersion operator
    disp_op = np.exp(-1j * beta * k ** 3 * dt)
    
    for n in range(nt - 1):
        # Nonlinear term: -alpha*u*u_x
        u_n = ifft(u_hat).real
        u[n + 1, :] = u_n  # store
        
        ux_hat = 1j * k * u_hat
        ux = ifft(ux_hat).real
        nonlin = -alpha * u_n * ux
        nonlin_hat = fft(nonlin)
        
        # Simple Euler step with exponential integrator for dispersion
        # u_hat_new = exp(-i*beta*k^3*dt) * (u_hat + dt*nonlin_hat)
        u_hat = disp_op * (u_hat + dt * nonlin_hat)
    
    # Store final step
    u[-1, :] = ifft(u_hat).real

    return u, dx, dt


def make_windows(u, nt_win, nx_win, stride_t, stride_x):
    """
    Extract overlapping windows from a spatiotemporal field.

    Args:
        u: np.ndarray of shape (nt, nx), full field
        nt_win: int, window size in time
        nx_win: int, window size in space
        stride_t: int, stride in time
        stride_x: int, stride in space

    Returns:
        windows: list of np.ndarray, each of shape (nt_win, nx_win)
    """
    nt, nx = u.shape
    windows = []
    
    t_starts = range(0, nt - nt_win + 1, stride_t)
    x_starts = range(0, nx - nx_win + 1, stride_x)
    
    for t0 in t_starts:
        for x0 in x_starts:
            win = u[t0 : t0 + nt_win, x0 : x0 + nx_win]
            windows.append(win.copy())
    
    return windows


def add_noise(u, noise_level=0.0, noise_type="gaussian"):
    """
    Add noise to a spatiotemporal field.

    Args:
        u: np.ndarray, input field
        noise_level: float, noise level as fraction of signal std (e.g., 0.01 = 1%)
        noise_type: str, "gaussian" or "uniform"

    Returns:
        u_noisy: np.ndarray, noisy field
    """
    if noise_level == 0.0:
        return u.copy()
    
    signal_std = np.std(u)
    noise_std = noise_level * signal_std
    
    if noise_type == "gaussian":
        noise = np.random.randn(*u.shape) * noise_std
    elif noise_type == "uniform":
        noise = np.random.uniform(-noise_std, noise_std, size=u.shape)
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")
    
    return u + noise

