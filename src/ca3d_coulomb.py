# -*- coding: utf-8 -*-
"""Часть 5: заряженный битовый газ с кулоновским полем.

Два сорта бит: p (+1) и n (−1), сумма сохраняется. Поле φ считается
из зарядовой плотности ρ = p − n решением решёточного уравнения Пуассона
(FFT, ∇²φ = −ρ). Биты дрейфуют по градиенту φ: p — вниз, n — вверх.
Обмен p/n между соседями добавляет короткодействие. Все переносы —
перестановки бит, поэтому p и n сохраняются по отдельности.
"""
from __future__ import annotations

import numpy as np

DIRS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def seed_neutral(L, N=16_000_000):
    p = np.zeros((L, L, L), np.int64)
    n = np.zeros_like(p)
    c = L // 2
    p[c, c, c] = N // 2
    n[c, c, c] = N - N // 2
    return p, n


def seed_positive(L, N=16_000_000):
    p = np.zeros((L, L, L), np.int64)
    p[L // 2, L // 2, L // 2] = N
    return p, np.zeros_like(p)


def seed_quadrupole(L, N=16_000_000, d=8):
    p = np.zeros((L, L, L), np.int64)
    n = np.zeros_like(p)
    c = L // 2
    q = N // 4
    p[c - d // 2, c - d // 2, c] = q
    n[c + d // 2, c - d // 2, c] = q
    n[c - d // 2, c + d // 2, c] = q
    p[c + d // 2, c + d // 2, c] = N - 3 * q
    return p, n


def seed_checker(L, N=16_000_000, size=5):
    p = np.zeros((L, L, L), np.int64)
    n = np.zeros_like(p)
    c = L // 2
    h = size // 2
    cells = [(x, y, z) for x in range(c - h, c + h + 1)
             for y in range(c - h, c + h + 1) for z in range(c - h, c + h + 1)]
    q = N // len(cells)
    for x, y, z in cells:
        if (x + y + z) % 2 == 0:
            p[x, y, z] = q
        else:
            n[x, y, z] = q
    return p, n


def seed_random(L, N=16_000_000, size=5, seed_num=1):
    rng = np.random.default_rng(seed_num)
    p = np.zeros((L, L, L), np.int64)
    n = np.zeros_like(p)
    c = L // 2
    h = size // 2
    cells = [(x, y, z) for x in range(c - h, c + h + 1)
             for y in range(c - h, c + h + 1) for z in range(c - h, c + h + 1)]
    q = N // len(cells)
    for i, (x, y, z) in enumerate(cells):
        if rng.random() < 0.5:
            p[x, y, z] = q
        else:
            n[x, y, z] = q
    return p, n


def seed_dipole(L, N=16_000_000, d=8):
    p = np.zeros((L, L, L), np.int64)
    n = np.zeros_like(p)
    c = L // 2
    p[c - d // 2, c, c] = N // 2
    n[c + d // 2, c, c] = N - N // 2
    return p, n


def poisson(rho: np.ndarray) -> np.ndarray:
    L = rho.shape[0]
    f = np.fft.rfftn(rho)
    kx = 2 * np.pi * np.fft.fftfreq(L)
    ky = 2 * np.pi * np.fft.fftfreq(L)
    kz = 2 * np.pi * np.fft.rfftfreq(L)
    K2 = kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2
    K2[0, 0, 0] = 1.0
    phi = np.fft.irfftn(f / K2, s=(L, L, L))
    phi -= phi.mean()
    return phi.astype(np.float32)


def _split(a, w, K=None):
    """Делит a по весам w[6] (плюс остаток); возвращает out и доли."""
    shares = []
    for d in range(6):
        s = (a * w[d]) // 7
        if K is not None:
            s = np.minimum(s, K)
        shares.append(s)
    out = a - sum(shares)
    for d in range(6):
        out += np.roll(shares[d], DIRS[d], axis=(0, 1, 2))
    return out


def step(p, n, beta=1.0, K=None, exchange=False, gamma=0.0):
    N = max(int(p.sum() + n.sum()), 1)
    rho = (p.astype(np.float32) - n.astype(np.float32)) / N
    phi = poisson(rho)
    # веса: p — вниз по φ, n — вверх
    dens = (p + n).astype(np.float32) / N
    wp = []
    wn = []
    for d in range(6):
        dphi = phi - np.roll(phi, DIRS[d], axis=(0, 1, 2))
        dd = dens - np.roll(dens, DIRS[d], axis=(0, 1, 2))
        wp.append(np.clip(1.0 + beta * 6.0 * dphi + gamma * 6.0 * dd, 0.0, None))
        wn.append(np.clip(1.0 - beta * 6.0 * dphi + gamma * 6.0 * dd, 0.0, None))
    # добавляем «стоячую» долю 1 и переводим веса в целые
    wp.append(np.ones_like(phi)); wn.append(np.ones_like(phi))
    wp = [np.maximum(1, np.rint(w * 100)).astype(np.int64) for w in wp]
    wn = [np.maximum(1, np.rint(w * 100)).astype(np.int64) for w in wn]
    sp = sum(wp); sn = sum(wn)
    p2 = np.zeros_like(p); n2 = np.zeros_like(n)
    # целочисленное деление с остатком у клетки
    shares_p = [(p * wp[d]) // sp for d in range(6)]
    shares_n = [(n * wn[d]) // sn for d in range(6)]
    if K is not None:
        shares_p = [np.minimum(s, K) for s in shares_p]
        shares_n = [np.minimum(s, K) for s in shares_n]
    p2 += p - sum(shares_p)
    n2 += n - sum(shares_n)
    for d in range(6):
        p2 += np.roll(shares_p[d], DIRS[d], axis=(0, 1, 2))
        n2 += np.roll(shares_n[d], DIRS[d], axis=(0, 1, 2))
    if exchange:
        # обмен долями p с p и n с n (заряды сохраняются по отдельности)
        left = slice(0, -1, 2); right = slice(1, None, 2)
        ctrl = ((p2[left] + n2[left] + p2[right] + n2[right]) & 1).astype(bool)
        sp_ = p2[left] // 10; sp2 = p2[right] // 10
        pn = np.where(ctrl, p2[left] - sp_ + sp2, p2[left])
        pn2 = np.where(ctrl, p2[right] - sp2 + sp_, p2[right])
        sn_ = n2[left] // 10; sn2 = n2[right] // 10
        nn = np.where(ctrl, n2[left] - sn_ + sn2, n2[left])
        nn2 = np.where(ctrl, n2[right] - sn2 + sn_, n2[right])
        p2[left] = pn; p2[right] = pn2
        n2[left] = nn; n2[right] = nn2
    return p2, n2, phi


def run(L=97, T=32, N=16_000_000, seed='neutral', beta=1.0, K=None,
        exchange=False, d=8, gamma=0.0):
    if seed == 'neutral':
        p, n = seed_neutral(L, N)
    elif seed == 'positive':
        p, n = seed_positive(L, N)
    elif seed == 'quadrupole':
        p, n = seed_quadrupole(L, N, d)
    elif seed == 'checker':
        p, n = seed_checker(L, N, 5)
    elif seed == 'random':
        p, n = seed_random(L, N, 5)
    else:
        p, n = seed_dipole(L, N, d)
    hist = [(0, int(p.sum()), int(n.sum()))]
    phi = None
    for _ in range(T):
        p, n, phi = step(p, n, beta, K, exchange, gamma)
        hist.append((len(hist), int(p.sum()), int(n.sum())))
    return p, n, phi, hist


def density(a):
    return np.log10(1.0 + a.astype(np.float64)).astype(np.float32)
