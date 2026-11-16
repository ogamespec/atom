# -*- coding: utf-8 -*-
"""Третья часть: консервативный битовый автомат, растущий из зародыша.

Клетка хранит B = 64 бита. Шесть каналов по 10 бит уходят к шести соседям,
4 бита остаются. За такт клетка сначала переставляет свои биты (поворот,
перестановка байтов), затем каналы d = 0..5 переносятся в соседние клетки.
Биты не рождаются и не исчезают: глобальный popcount постоянен.
Обрабатываются только непустые клетки; пустые пропускаются.
"""
from __future__ import annotations

import numpy as np

DIRS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
M64 = (1 << 64) - 1
STAY = 0xF000000000000000  # биты 60..63
CH = 10                    # ширина канала


def rotl(w: int, r: int) -> int:
    r %= 64
    return ((w << r) | (w >> (64 - r))) & M64 if r else w


def byte_perm(w: int, kind: str) -> int:
    if kind == 'id':
        return w
    b = [(w >> (8 * i)) & 0xFF for i in range(8)]
    if kind == 'rev':
        return sum(b[7 - i] << (8 * i) for i in range(8))
    if kind == 'swap':
        o = [1, 0, 3, 2, 5, 4, 7, 6]
        return sum(b[o[i]] << (8 * i) for i in range(8))
    if kind == 'cycle':
        return sum(b[(i + 1) % 8] << (8 * i) for i in range(8))
    if kind == 'bitrev':
        return int(f'{w:064b}'[::-1], 2)
    raise ValueError(kind)


def seed(L: int, size: int = 15, dense: bool = True, seed_num: int = 1) -> dict:
    """Зародыш size^3; в каждой клетке 64 единичных бита."""
    c = L // 2
    h = size // 2
    g = {}
    rng = np.random.default_rng(seed_num)
    for x in range(c - h, c + h + 1):
        for y in range(c - h, c + h + 1):
            for z in range(c - h, c + h + 1):
                g[(x, y, z)] = M64 if dense else int(rng.integers(0, M64))
    return g


def total_bits(g: dict) -> int:
    return sum(w.bit_count() for w in g.values())


def step(g: dict, L: int, rot: int = 1, pk: str = 'id',
         route=None, tick: int = 0, field: bool = False) -> dict:
    """Один такт: перестановка бит + перенос шести каналов."""
    if route is None:
        route = list(range(6))
    new = {}
    for (x, y, z), w in g.items():
        if not w:
            continue  # пустая клетка пропускается
        w = byte_perm(rotl(w, rot), pk)
        for d in range(6):
            b = (w >> (CH * d)) & ((1 << CH) - 1)
            if b:
                k = (d + (x + y + z if field else 0)) % 6
                dx, dy, dz = DIRS[route[k]]
                q = ((x + dx) % L, (y + dy) % L, (z + dz) % L)
                new[q] = new.get(q, 0) | (b << (CH * d))
        st = w & STAY
        if st:
            new[(x, y, z)] = new.get((x, y, z), 0) | st
    return new


def dense(g: dict, L: int) -> np.ndarray:
    lut = np.array([i.bit_count() for i in range(256)], np.uint8)
    a = np.zeros((L, L, L), np.uint8)
    for (x, y, z), w in g.items():
        if w:
            a[x, y, z] = lut[np.frombuffer(w.to_bytes(8, 'little'), np.uint8)].sum()
    return a


def front_radius(g: dict, L: int) -> float:
    if not g:
        return 0.0
    c = (L - 1) / 2
    return max(((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2) ** 0.5 for x, y, z in g)


def run(L=161, T=48, size=15, rot=1, pk='id', route=None, seed_num=1,
        field=False, track=False):
    g = seed(L, size, seed_num=seed_num)
    b0 = total_bits(g)
    hist = [(0, len(g), b0, front_radius(g, L))]
    for t in range(T):
        g = step(g, L, rot, pk, route, t, field)
        if track:
            hist.append((t + 1, len(g), total_bits(g), front_radius(g, L)))
    if not track:
        hist.append((T, len(g), total_bits(g), front_radius(g, L)))
    return g, hist
