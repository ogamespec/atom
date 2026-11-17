# -*- coding: utf-8 -*-
"""Четвёртая часть: обмен битами (гейты Фредкина) на большой решётке.

Клетка — 64 бита: 6 каналов по 10 бит + 4 «стоячих». За такт:
  1) локальная перестановка бит (поворот/перестановка байтов);
  2) каналы d=0..5 переносятся к шести соседям;
  3) обмен: если контрольный бит 63 у клетки равен 1, она меняется
     младшими 10 битами с соседом по выбранной оси.
Все операции — перестановки битов, поэтому popcount сохраняется точно.
Обрабатываются только ненулевые клетки (в плотной реализации нули
остаются нулями, пока не получат биты).
"""
from __future__ import annotations

import numpy as np

DIRS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
M64 = np.uint64(0xFFFFFFFFFFFFFFFF)
STAY = np.uint64(0xF000000000000000)
LOW = np.uint64(0x3FF)
NOTLOW = np.uint64(0xFFFFFFFFFFFFFC00)


def rotl(a: np.ndarray, r: int) -> np.ndarray:
    r %= 64
    if r == 0:
        return a
    return (a << np.uint64(r)) | (a >> np.uint64(64 - r))


def byte_swap(a: np.ndarray) -> np.ndarray:
    m1 = np.uint64(0x00FF00FF00FF00FF)
    m2 = np.uint64(0xFF00FF00FF00FF00)
    return ((a & m1) << np.uint64(8)) | ((a & m2) >> np.uint64(8))


def seed(L: int, size: int = 250) -> np.ndarray:
    """Плотный зародыш size^3, в каждой клетке все 64 бита."""
    a = np.zeros((L, L, L), np.uint64)
    c = L // 2
    h = size // 2
    a[c - h:c + h + 1, c - h:c + h + 1, c - h:c + h + 1] = M64
    return a


def bits(a: np.ndarray) -> int:
    return int(np.bitwise_count(a).sum(dtype=np.int64))


def density(a: np.ndarray) -> np.ndarray:
    return np.bitwise_count(a).astype(np.uint8)


def _exchange_axis(a: np.ndarray, axis: int) -> int:
    """Условный своп младших 10 бит между парами вдоль оси. Возвращает число свопов."""
    if axis == 0:
        left = a[0:-1:2]
        right = a[1::2]
    elif axis == 1:
        left = a[:, 0:-1:2]
        right = a[:, 1::2]
    else:
        left = a[:, :, 0:-1:2]
        right = a[:, :, 1::2]
    ctrl = ((left >> np.uint64(63)) & np.uint64(1)).astype(bool)
    lb = left & LOW
    rb = right & LOW
    nl = (left & NOTLOW) | np.where(ctrl, rb, lb)
    nr = (right & NOTLOW) | np.where(ctrl, lb, rb)
    if axis == 0:
        a[0:-1:2] = nl
        a[1::2] = nr
    elif axis == 1:
        a[:, 0:-1:2] = nl
        a[:, 1::2] = nr
    else:
        a[:, :, 0:-1:2] = nl
        a[:, :, 1::2] = nr
    return int(ctrl.sum())


def step(a: np.ndarray, rot: int = 1, swap: bool = False, exchange='x',
         tick: int = 0, field: bool = False):
    L = a.shape[0]
    w = rotl(a, rot)
    if swap:
        w = byte_swap(w)
    out = np.zeros_like(a)
    for d in range(6):
        block = (w >> np.uint64(10 * d)) & LOW
        k = (d + (tick if field else 0)) % 6
        out |= np.roll(block, DIRS[k], axis=(0, 1, 2)) << np.uint64(10 * d)
    out |= w & STAY
    swaps = 0
    if exchange == 'all':
        for ax in range(3):
            swaps += _exchange_axis(out, ax)
    elif exchange in ('x', 'y', 'z'):
        swaps = _exchange_axis(out, {'x': 0, 'y': 1, 'z': 2}[exchange])
    return out, swaps


def run(L=512, T=24, size=250, rot=0, swap=False, exchange='x', field=False):
    a = seed(L, size)
    b0 = bits(a)
    hist = []
    for t in range(T):
        a, sw = step(a, rot, swap, exchange, t, field)
        hist.append((t + 1, int(np.count_nonzero(a)), bits(a), sw))
    return a, hist
