# -*- coding: utf-8 -*-
"""Разреженный локальный CA: только активные клетки, огромное пространство.

Состояние — dict (x,y,z) -> (p,n). За такт обрабатываются только непустые
клетки; пустые пропускаются. Никаких массивов L^3, FFT или поля.
"""
from __future__ import annotations
import numpy as np

DIRS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def seed(N=1_000_000, charge_frac=0.8):
    P = int(round(N * charge_frac))
    return {(0, 0, 0): [P, N - P]}


def step(state, N=1_000_000, attr=1.0, rep=1.0, bias=0.2, noise=0.5,
         mob=20, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    new = {}
    B = (1.0, -1.0, 1.0, -1.0, 1.0, -1.0)  # diag
    for (x, y, z), (p, n) in state.items():
        if p + n <= 0:
            continue
        shares_p = []
        shares_n = []
        wp = []
        wn = []
        for d in range(6):
            dx, dy, dz = DIRS[d]
            q = state.get((x + dx, y + dy, z + dz))
            pn, nn = (q[0], q[1]) if q else (0, 0)
            rp = int(rng.integers(-1, 2)) if noise > 0 else 0
            rn = int(rng.integers(-1, 2)) if noise > 0 else 0
            wp.append(max(0.0, 1.0 + attr * 3.0 * (nn - pn) / N
                          + rep * 3.0 * (p - pn) / N + bias * 2.0 * B[d] + noise * rp))
            wn.append(max(0.0, 1.0 + attr * 3.0 * (pn - nn) / N
                          + rep * 3.0 * (n - nn) / N - bias * 2.0 * B[d] + noise * rn))
        wp.append(1.0); wn.append(1.0)
        sp = sum(wp); sn = sum(wn)
        for d in range(6):
            shares_p.append((p * int(round(wp[d] * 100))) // max(1, int(round(sp * 100))))
            shares_n.append((n * int(round(wn[d] * 100))) // max(1, int(round(sn * 100))))
        if mob < 100:
            shares_p = [s * mob // 100 for s in shares_p]
            shares_n = [s * mob // 100 for s in shares_n]
        kp = p - sum(shares_p)
        kn = n - sum(shares_n)
        if kp or kn:
            cell = new.get((x, y, z))
            if cell:
                cell[0] += kp; cell[1] += kn
            else:
                new[(x, y, z)] = [kp, kn]
        for d in range(6):
            if shares_p[d] or shares_n[d]:
                dx, dy, dz = DIRS[d]
                q = (x + dx, y + dy, z + dz)
                cell = new.get(q)
                if cell:
                    cell[0] += shares_p[d]; cell[1] += shares_n[d]
                else:
                    new[q] = [shares_p[d], shares_n[d]]
    return new


def run_sparse(T=20000, N=1_000_000, charge_frac=0.8, attr=1.0, rep=1.0,
               bias=0.2, noise=0.5, mob=20, seed_num=1, checkpoints=()):
    state = seed(N, charge_frac)
    rng = np.random.default_rng(seed_num)
    hist = []
    for t in range(1, T + 1):
        state = step(state, N, attr, rep, bias, noise, mob, rng)
        if t in checkpoints:
            hist.append((t, len(state), sum(v[0] + v[1] for v in state.values())))
    return state, hist
