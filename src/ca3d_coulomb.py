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
    P = int(round(N * charge_frac))
    p[c, c, c] = P
    n[c, c, c] = N - P
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


def poisson(rho: np.ndarray, mu: float = 0.0) -> np.ndarray:
    L = rho.shape[0]
    f = np.fft.rfftn(rho)
    kx = 2 * np.pi * np.fft.fftfreq(L)
    ky = 2 * np.pi * np.fft.fftfreq(L)
    kz = 2 * np.pi * np.fft.rfftfreq(L)
    K2 = kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2
    K2[0, 0, 0] = 1.0
    if mu:
        K2 = K2 + mu * mu
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


def step(p, n, beta=1.0, K=None, exchange=False, gamma=0.0,
         bias=0.0, rep=0.0, noise=0.0, rng=None, bias_mode='x',
         screening=0.0, anti=False):
    L = p.shape[0]
    N = max(int(p.sum() + n.sum()), 1)
    rho = (p.astype(np.float32) - n.astype(np.float32)) / N
    phi = poisson(rho, screening)
    if anti:
        beta = -beta
    # веса: p — вниз по φ, n — вверх
    dens = (p + n).astype(np.float32) / N
    pdens = p.astype(np.float32) / N
    ndens = n.astype(np.float32) / N
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
    elif bias_mode == 'random':
        B = [rng.integers(-1, 2, (L, L, L)).astype(np.float32) for _ in range(6)]
    else:
        B = [0.0] * 6
    wp = []
    wn = []
    for d in range(6):
        dphi = phi - np.roll(phi, DIRS[d], axis=(0, 1, 2))
        dd = dens - np.roll(dens, DIRS[d], axis=(0, 1, 2))
        bd = B[d]
        rp = rng.integers(-1, 2, phi.shape) if noise > 0 else 0
        rn = rng.integers(-1, 2, phi.shape) if noise > 0 else 0
        wpp = np.clip(1.0 + beta * 6.0 * dphi + gamma * 6.0 * dd
                      + bias * 3.0 * bd
                      + rep * 6.0 * (ndens - np.roll(ndens, DIRS[d], axis=(0, 1, 2)))
                      + noise * rp, 0.0, None)
        wnn = np.clip(1.0 - beta * 6.0 * dphi + gamma * 6.0 * dd
                      - bias * 3.0 * bd
                      + rep * 6.0 * (pdens - np.roll(pdens, DIRS[d], axis=(0, 1, 2)))
                      + noise * rn, 0.0, None)
        wp.append(wpp)
        wn.append(wnn)
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


def run_one(L=97, T=32, N=16_000_000, beta=1.0, K=None, exchange=False,
            gamma=0.0, bias=0.0, rep=0.0, noise=0.0, seed_num=1,
            bias_mode='x', screening=0.0, anti=False, charge_frac=0.5):
    """Зародыш — ОДНА клетка: p=n=N/2. Всё остальное — эволюция правила."""
    p = np.zeros((L, L, L), np.int64)
    n = np.zeros_like(p)
    c = L // 2
    P = int(round(N * charge_frac))
    p[c, c, c] = P
    n[c, c, c] = N - P
    rng = np.random.default_rng(seed_num)
    hist = [(0, int(p.sum()), int(n.sum()))]
    phi = np.zeros((L, L, L), np.float32)
    for _ in range(T):
        p, n, phi = step(p, n, beta, K, exchange, gamma, bias, rep, noise, rng,
                         bias_mode, screening, anti)
        hist.append((len(hist), int(p.sum()), int(n.sum())))
    return p, n, phi, hist


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


# --------------------------------------------------------------------------
#  Гелевое правило: оба сорта заряжены одинаково, короткое притяжение +
#  дальнее кулоновское отталкивание. Кандидат на сеть из блобов.
# --------------------------------------------------------------------------


def step_gel(p, n, beta=0.5, self_att=1.0, gamma=0.5, noise=0.5, K=None, rng=None, cross_rep=0.0):
    L = p.shape[0]
    N = max(int(p.sum() + n.sum()), 1)
    rho = (p.astype(np.float32) + n.astype(np.float32)) / N
    phi = poisson(rho)
    pd = p.astype(np.float32) / N
    nd = n.astype(np.float32) / N
    dens = pd + nd
    if rng is None:
        rng = np.random.default_rng(0)
    wp, wn = [], []
    for d in range(6):
        dphi = phi - np.roll(phi, DIRS[d], axis=(0, 1, 2))
        dd = dens - np.roll(dens, DIRS[d], axis=(0, 1, 2))
        dp = pd - np.roll(pd, DIRS[d], axis=(0, 1, 2))
        dn = nd - np.roll(nd, DIRS[d], axis=(0, 1, 2))
        rp = rng.integers(-1, 2, phi.shape) if noise > 0 else 0
        rn = rng.integers(-1, 2, phi.shape) if noise > 0 else 0
        wp.append(np.clip(1.0 + beta * 6.0 * dphi + self_att * 6.0 * dp
                          - cross_rep * 6.0 * dn
                          + gamma * 6.0 * dd + noise * rp, 0.0, None))
        wn.append(np.clip(1.0 + beta * 6.0 * dphi + self_att * 6.0 * dn
                          - cross_rep * 6.0 * dp
                          + gamma * 6.0 * dd + noise * rn, 0.0, None))
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
    return p2, n2


def run_gel(L=65, T=20, N=16_000_000, beta=0.5, self_att=1.0, gamma=0.5,
            noise=0.5, K=None, charge_frac=0.5, seed_num=1, cross_rep=0.0):
    p = np.zeros((L, L, L), np.int64)
    n = np.zeros_like(p)
    c = L // 2
    P = int(round(N * charge_frac))
    p[c, c, c] = P
    n[c, c, c] = N - P
    rng = np.random.default_rng(seed_num)
    for _ in range(T):
        p, n = step_gel(p, n, beta, self_att, gamma, noise, K, rng, cross_rep)
    return p, n
