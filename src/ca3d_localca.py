# -*- coding: utf-8 -*-
"""Чистый локальный клеточный автомат: без FFT и без поля.

Состояние клетки — счётчики p и n. За такт обрабатываются только клетки
с ненулевыми битами (и их соседи). Каждая такая клетка делит свои биты
по шести направлениям по локальному правилу обмена/дрейфа; пустые клетки
пропускаются. Никаких глобальных уравнений.
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


def step(p, n, attr=1.0, rep=0.5, bias=0.0, bias_mode='x',
         noise=0.0, K=None, rng=None, mob=100):
    """Один локальный такт: обмен/дрейф по шести направлениям."""
    L = p.shape[0]
    N = max(int(p.sum() + n.sum()), 1)
    if rng is None:
        rng = np.random.default_rng(0)
    if bias_mode == 'x':
        B = [1.0, -1.0, 0.0, 0.0, 0.0, 0.0]
    elif bias_mode == 'diag':
        B = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    else:
        B = [0.0] * 6
    sp, sn = [], []
    for d in range(6):
        pn = np.roll(p, DIRS[d], axis=(0, 1, 2))
        nn = np.roll(n, DIRS[d], axis=(0, 1, 2))
        rp = rng.integers(-1, 2, p.shape) if noise > 0 else 0
        rn = rng.integers(-1, 2, p.shape) if noise > 0 else 0
        # p тянется к n и отталкивается от p; n — наоборот
        wp = np.clip(1.0 + attr * 3.0 * (nn - pn) / N + rep * 3.0 * (p - pn) / N
                     + bias * 2.0 * B[d] + noise * rp, 0.0, None)
        wn = np.clip(1.0 + attr * 3.0 * (pn - nn) / N + rep * 3.0 * (n - nn) / N
                     - bias * 2.0 * B[d] + noise * rn, 0.0, None)
        sp.append(np.maximum(1, np.rint(wp * 100)).astype(np.int64))
        sn.append(np.maximum(1, np.rint(wn * 100)).astype(np.int64))
    # доли только для непустых клеток; пустые дают нули
    Sp = [np.where((p + n) > 0, (p * sp[d]) // sum(sp), 0) for d in range(6)]
    Sn = [np.where((p + n) > 0, (n * sn[d]) // sum(sn), 0) for d in range(6)]
    if mob < 100:
        Sp = [s * mob // 100 for s in Sp]
        Sn = [s * mob // 100 for s in Sn]
    if K is not None:
        Sp = [np.minimum(s, K) for s in Sp]
        Sn = [np.minimum(s, K) for s in Sn]
    p2 = p - sum(Sp); n2 = n - sum(Sn)
    for d in range(6):
        p2 = p2 + np.roll(Sp[d], DIRS[d], axis=(0, 1, 2))
        n2 = n2 + np.roll(Sn[d], DIRS[d], axis=(0, 1, 2))
    return p2, n2


def run(L=97, T=200, N=16_000_000, charge_frac=0.5, attr=1.0, rep=0.5,
        bias=0.0, bias_mode='x', noise=0.0, K=None, seed_num=1, checkpoints=(),
        mob=100):
    p, n = seed_one(L, N, charge_frac)
    rng = np.random.default_rng(seed_num)
    snaps = {}
    for t in range(1, T + 1):
        p, n = step(p, n, attr, rep, bias, bias_mode, noise, K, rng, mob)
        if t in checkpoints:
            snaps[t] = (p.copy(), n.copy())
    return p, n, snaps
