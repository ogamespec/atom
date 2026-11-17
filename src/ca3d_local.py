# -*- coding: utf-8 -*-
"""Локальный клеточный автомат: никакого FFT и глобального решателя.

Состояние: p, n — счётчики бит, φ — локальное поле. За такт:
  * φ обновляется из шести соседей (один шаг Якоби): φ' = (Σφ/6 + ρ/6);
  * p и n делят биты по направлениям с весами от локальных разностей φ;
  * только клетки с битами (и их соседи по φ) участвуют.
Поле распространяется на одну клетку за такт — причинность сохраняется.
"""
from __future__ import annotations
import numpy as np

DIRS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def seed_one(L, N=16_000_000, charge_frac=0.5):
    p = np.zeros((L, L, L), np.int64)
    n = np.zeros_like(p)
    c = L // 2
    P = int(round(N * charge_frac))
    p[c, c, c] = P
    n[c, c, c] = N - P
    return p, n


def relax_phi(p, n, phi, omega=0.8):
    """Один локальный шаг Якоби: φ(x) = (Σ соседей + ρ)/6."""
    N = max(int(p.sum() + n.sum()), 1)
    rho = (p.astype(np.float32) - n.astype(np.float32)) / N
    s = sum(np.roll(phi, d, axis=(0, 1, 2)) for d in DIRS)
    phi_new = (s + rho) / 6.0
    phi = phi + omega * (phi_new - phi)
    # открытые границы: поле на стенках нулевое
    phi[0, :, :] = phi[-1, :, :] = 0
    phi[:, 0, :] = phi[:, -1, :] = 0
    phi[:, :, 0] = phi[:, :, -1] = 0
    return phi


def step(p, n, phi, beta=0.5, bias=0.0, bias_mode='x', rep=0.0,
         noise=0.0, K=None, rng=None):
    L = p.shape[0]
    N = max(int(p.sum() + n.sum()), 1)
    phi = relax_phi(p, n, phi)
    dens = (p + n).astype(np.float32) / N
    pd = p.astype(np.float32) / N
    nd = n.astype(np.float32) / N
    if rng is None:
        rng = np.random.default_rng(0)
    if bias_mode == 'x':
        B = [1.0, -1.0, 0.0, 0.0, 0.0, 0.0]
    elif bias_mode == 'diag':
        B = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    elif bias_mode == 'radial':
        xg, yg, zg = np.indices((L, L, L))
        c0 = (L - 1) / 2
        sx = np.sign(xg - c0); sy = np.sign(yg - c0); sz = np.sign(zg - c0)
        B = [sx, -sx, sy, -sy, sz, -sz]
    else:
        B = [0.0] * 6
    wp, wn = [], []
    for d in range(6):
        dphi = phi - np.roll(phi, DIRS[d], axis=(0, 1, 2))
        bd = B[d]
        rp = rng.integers(-1, 2, phi.shape) if noise > 0 else 0
        rn = rng.integers(-1, 2, phi.shape) if noise > 0 else 0
        wp.append(np.clip(1.0 + beta * 6.0 * dphi + bias * 3.0 * bd
                          + rep * 6.0 * (nd - np.roll(nd, DIRS[d], axis=(0, 1, 2)))
                          + noise * rp, 0.0, None))
        wn.append(np.clip(1.0 - beta * 6.0 * dphi - bias * 3.0 * bd
                          + rep * 6.0 * (pd - np.roll(pd, DIRS[d], axis=(0, 1, 2)))
                          + noise * rn, 0.0, None))
    wp.append(np.ones_like(phi)); wn.append(np.ones_like(phi))
    wp = [np.maximum(1, np.rint(w * 100)).astype(np.int64) for w in wp]
    wn = [np.maximum(1, np.rint(w * 100)).astype(np.int64) for w in wn]
    sp, sn = sum(wp), sum(wn)
    shp = [(p * wp[d]) // sp for d in range(6)]
    shn = [(n * wn[d]) // sn for d in range(6)]
    if K is not None:
        shp = [np.minimum(s, K) for s in shp]
        shn = [np.minimum(s, K) for s in shn]
    p2 = p - sum(shp); n2 = n - sum(shn)
    for d in range(6):
        p2 = p2 + np.roll(shp[d], DIRS[d], axis=(0, 1, 2))
        n2 = n2 + np.roll(shn[d], DIRS[d], axis=(0, 1, 2))
    return p2, n2, phi


def run_local(L=97, T=200, N=16_000_000, charge_frac=0.5, beta=0.5,
              bias=0.0, bias_mode='x', rep=0.0, noise=0.0, K=None,
              seed_num=1, checkpoints=()):
    p, n = seed_one(L, N, charge_frac)
    phi = np.zeros((L, L, L), np.float32)
    rng = np.random.default_rng(seed_num)
    snaps = {}
    for t in range(1, T + 1):
        p, n, phi = step(p, n, phi, beta, bias, bias_mode, rep, noise, K, rng)
        if t in checkpoints:
            snaps[t] = (p.copy(), n.copy(), phi.copy())
    return p, n, phi, snaps
