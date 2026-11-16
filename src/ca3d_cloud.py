# -*- coding: utf-8 -*-
"""Часть 4: один зародыш — 16 млн бит; перенос долей и обмен.

Клетка хранит счётчик бит (int64). Зародыш — ОДНА клетка с N = 16 000 000.
За такт клетка отдаёт соседям доли своих бит (веса направлений), остаток
остаётся. Обмен — соседи меняются долями (10%) при нечётной сумме; это
сохраняет общую сумму. Пустые клетки не обрабатываются.
"""
from __future__ import annotations

import numpy as np

DIRS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
# 6 направлений + остаток; сумма = 7
WEIGHTS = {
    'uniform': (1, 1, 1, 1, 1, 1, 1),
    'x': (2, 2, 1, 1, 1, 0, 0),
    'z': (0, 0, 1, 1, 2, 2, 1),
    'diag': (2, 1, 2, 0, 1, 0, 1),
}


def seed(L: int, N: int = 16_000_000):
    a = np.zeros((L, L, L), np.int64)
    a[L // 2, L // 2, L // 2] = N
    return a


def total(a) -> int:
    return int(a.sum(dtype=np.int64))


def step(a: np.ndarray, wmode='uniform', K=None, exchange='none', tick=0):
    w = np.array(WEIGHTS[wmode], np.int64).copy()
    w[:6] = np.roll(w[:6], -(tick % 6))
    shares = []
    for d in range(6):
        s = (a * w[d]) // 7
        if K is not None:
            s = np.minimum(s, K)
        shares.append(s)
    out = a - sum(shares)
    for d in range(6):
        out += np.roll(shares[d], DIRS[d], axis=(0, 1, 2))
    # обмен: 10% долей, если сумма в паре нечётная
    axes = {'x': [0], 'y': [1], 'z': [2], 'all': [0, 1, 2]}.get(exchange, [])
    for ax in axes:
        if ax == 0:
            left, right = out[0:-1:2], out[1::2]
        elif ax == 1:
            left, right = out[:, 0:-1:2], out[:, 1::2]
        else:
            left, right = out[:, :, 0:-1:2], out[:, :, 1::2]
        ctrl = ((left + right) & 1).astype(bool)
        sl = left // 10
        sr = right // 10
        nl = np.where(ctrl, left - sl + sr, left)
        nr = np.where(ctrl, right - sr + sl, right)
        if ax == 0:
            out[0:-1:2] = nl; out[1::2] = nr
        elif ax == 1:
            out[:, 0:-1:2] = nl; out[:, 1::2] = nr
        else:
            out[:, :, 0:-1:2] = nl; out[:, :, 1::2] = nr
    return out


def radial_weights(L: int) -> np.ndarray:
    """Веса наружу: 2 — от центра, 1 — вбок, 0 — внутрь, 1 — остаток."""
    x, y, z = np.indices((L, L, L))
    c = (L - 1) / 2
    W = np.zeros((7, L, L, L), np.int8)
    for ax, coord in enumerate((x, y, z)):
        plus = 2 * (coord > c) + 1 * (coord == c)
        minus = 2 * (coord < c) + 1 * (coord == c)
        W[2 * ax] = plus
        W[2 * ax + 1] = minus
    W[6] = 1
    return W


def run(L=161, T=64, N=16_000_000, wmode='uniform', K=None, exchange='none'):
    a = seed(L, N)
    W = radial_weights(L) if wmode == 'radial' else None
    hist = [(0, total(a), int(np.count_nonzero(a)))]
    for t in range(T):
        if W is None:
            a = step(a, wmode, K, exchange, t)
        else:
            shares = []
            for d in range(6):
                sh = (a * W[d].astype(np.int64)) // 7
                if K is not None:
                    sh = np.minimum(sh, K)
                shares.append(sh)
            out = a - sum(shares)
            for d in range(6):
                out += np.roll(shares[d], DIRS[d], axis=(0, 1, 2))
            # обмен
            axes = {'x': [0], 'y': [1], 'z': [2], 'all': [0, 1, 2]}.get(exchange, [])
            for ax in axes:
                if ax == 0:
                    left, right = out[0:-1:2], out[1::2]
                elif ax == 1:
                    left, right = out[:, 0:-1:2], out[:, 1::2]
                else:
                    left, right = out[:, :, 0:-1:2], out[:, :, 1::2]
                ctrl = ((left + right) & 1).astype(bool)
                sl, sr = left // 10, right // 10
                nl = np.where(ctrl, left - sl + sr, left)
                nr = np.where(ctrl, right - sr + sl, right)
                if ax == 0:
                    out[0:-1:2] = nl; out[1::2] = nr
                elif ax == 1:
                    out[:, 0:-1:2] = nl; out[:, 1::2] = nr
                else:
                    out[:, :, 0:-1:2] = nl; out[:, :, 1::2] = nr
            a = out
        hist.append((t + 1, total(a), int(np.count_nonzero(a))))
    return a, hist


def density(a):
    return np.log10(1.0 + a.astype(np.float64)).astype(np.float32)


# --------------------------------------------------------------------------
#  Стохастические правила: «Бог играет в кости»
# --------------------------------------------------------------------------


def step_god(a: np.ndarray, mode: str, rng: np.random.Generator, K=None):
    """Случайные правила, сохраняющие сумму бит.

    mode:
      * 'split'  — клетка выбирает 3 случайных направления и отдаёт по 2/7;
      * 'rot'    — радиальные веса случайно поворачиваются в каждой клетке;
      * 'swap'   — радиальный перенос + случайный обмен;
      * 'weights'— случайные целые веса 0..2 на направления.
    """
    L = a.shape[0]
    shares = []
    if mode == 'packet':
        dirs = rng.integers(0, 6, (L, L, L), dtype=np.uint8)
        for d in range(6):
            shares.append(a * (dirs == d))
    elif mode == 'branch':
        dirs = rng.integers(0, 6, (3, L, L, L), dtype=np.uint8)
        k = rng.integers(1, 4, (L, L, L)).astype(np.int64)
        for d in range(6):
            cnt = np.zeros((L, L, L), np.int64)
            for j in range(3):
                cnt += ((dirs[j] == d) & (j < k)).astype(np.int64)
            sh = (a * cnt) // np.maximum(k, 1)
            shares.append(sh)
    elif mode == 'split':
        dirs = rng.integers(0, 6, (3, L, L, L), dtype=np.uint8)
        for d in range(6):
            cnt = (dirs == d).sum(0).astype(np.int64)
            sh = (a * 2 * cnt) // 7
            if K is not None:
                sh = np.minimum(sh, K)
            shares.append(sh)
    elif mode == 'rot':
        W = radial_weights(L)
        off = rng.integers(0, 6, (L, L, L), dtype=np.uint8)
        x, y, z = np.indices((L, L, L))
        for d in range(6):
            w = W[(d + off) % 6, x, y, z].astype(np.int64)
            sh = (a * w) // 7
            if K is not None:
                sh = np.minimum(sh, K)
            shares.append(sh)
    elif mode == 'weights':
        w = rng.integers(0, 3, (6, L, L, L)).astype(np.int64)
        tot = w.sum(0)
        for d in range(6):
            sh = (a * w[d]) // np.maximum(tot, 1)
            if K is not None:
                sh = np.minimum(sh, K)
            shares.append(sh)
    else:  # swap: радиальный перенос + случайный обмен
        W = radial_weights(L)
        for d in range(6):
            sh = (a * W[d].astype(np.int64)) // 7
            if K is not None:
                sh = np.minimum(sh, K)
            shares.append(sh)
    out = a - sum(shares)
    for d in range(6):
        out += np.roll(shares[d], DIRS[d], axis=(0, 1, 2))
    if mode == 'swap':
        p = 0.15
        for ax in range(3):
            if ax == 0:
                left, right = out[0:-1:2], out[1::2]
            elif ax == 1:
                left, right = out[:, 0:-1:2], out[:, 1::2]
            else:
                left, right = out[:, :, 0:-1:2], out[:, :, 1::2]
            ctrl = rng.random(left.shape) < p
            sl, sr = left // 10, right // 10
            nl = np.where(ctrl, left - sl + sr, left)
            nr = np.where(ctrl, right - sr + sl, right)
            if ax == 0:
                out[0:-1:2] = nl; out[1::2] = nr
            elif ax == 1:
                out[:, 0:-1:2] = nl; out[:, 1::2] = nr
            else:
                out[:, :, 0:-1:2] = nl; out[:, :, 1::2] = nr
    return out


def run_god(L=161, T=64, N=16_000_000, mode='split', seed_num=1, K=None):
    rng = np.random.default_rng(seed_num)
    a = seed(L, N)
    hist = [(0, total(a), int(np.count_nonzero(a)))]
    for t in range(T):
        a = step_god(a, mode, rng, K)
        hist.append((t + 1, total(a), int(np.count_nonzero(a))))
    return a, hist
