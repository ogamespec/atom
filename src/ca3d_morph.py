# -*- coding: utf-8 -*-
"""Второе поколение правил: сети, пена, блобы.

Четыре семейства:
  * Potts3D   — Q-фазная модель Поттса (мажоритарное правило): домены и стенки;
  * Tumor3D   — рост опухоли с питательным полем: блобы с некротическим ядром;
  * Invasion3D— инвазионная перколяция: ветвящаяся сеть по порогам;
  * Bond3D    — рост графа связей: сеть из узлов и рёбер.

Все правила локальны (кроме Invasion/Bond — это ростовые автоматы),
детерминированы при фиксированном seed и работают на кубе L^3.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import convolve1d

DIRS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
OPP = (1, 0, 5, 4, 3, 2)


def von6(a, mode='wrap'):
    k = np.array([1, 0, 1], dtype=np.int32)
    out = np.zeros(a.shape, dtype=np.int32)
    aa = a.astype(np.int32, copy=False)
    for ax in range(3):
        out += convolve1d(aa, k, axis=ax, mode=mode)
    return out


def box26(a, mode='wrap'):
    k = np.ones(3, dtype=np.int32)
    out = a.astype(np.int32, copy=False)
    for ax in range(3):
        out = convolve1d(out, k, axis=ax, mode=mode)
    return out


def hash3(L: int, t: int, salt: int = 0) -> np.ndarray:
    x, y, z = np.indices((L, L, L))
    h = (x.astype(np.int64) * 73856093) ^ (y.astype(np.int64) * 19349663) \
        ^ (z.astype(np.int64) * 83492791) ^ ((t + 1) * 2654435761 + salt * 97531)
    return (h & 0x7fffffff)


class Potts3D:
    """Q-фазная модель Поттса: клетка принимает фазу большинства соседей.

    Из случайного блока вырастает пена: домены (блобы) разделены стенками
    (сетью поверхностей). Параметры: Q, T, init, noise, neighborhood.
    """
    family = 'potts'

    def __init__(self, Q=8, T=60, init='full', seed_size=31, noise=0.0,
                 neighborhood='von6', vacuum=False, name=None):
        self.Q, self.T, self.init = Q, T, init
        self.seed_size, self.noise = seed_size, noise
        self.neighborhood = neighborhood
        self.vacuum = vacuum
        self.name = name or f'potts-Q{Q}-{init}-T{T}-{neighborhood}'

    def run(self, L=65, seed=1, snapshots=None):
        rng = np.random.default_rng(seed)
        if self.init == 'full':
            s = rng.integers(0, self.Q, (L, L, L), dtype=np.uint8)
        elif self.init == 'block':
            s = np.zeros((L, L, L), dtype=np.uint8)
            c = L // 2
            h = self.seed_size // 2
            s[c - h:c + h + 1, c - h:c + h + 1, c - h:c + h + 1] = \
                rng.integers(0, self.Q, (2 * h + 1,) * 3, dtype=np.uint8)
        else:
            raise ValueError(self.init)
        nbr = box26 if self.neighborhood == 'moore' else von6
        for t in range(self.T):
            if self.vacuum:
                qs = range(1, self.Q)
            else:
                qs = range(self.Q)
            counts = np.stack([nbr(s == q) for q in qs])
            own = counts[s.astype(np.int64) - (0 if not self.vacuum else 1),
                         np.arange(L)[:, None, None], np.arange(L)[None, :, None],
                         np.arange(L)[None, None, :]] if not self.vacuum else None
            best = counts.max(0)
            tie = counts == best[None]
            tie = tie & (rng.random(counts.shape) < 0.5)
            s2 = (tie.argmax(0) + (0 if not self.vacuum else 1)).astype(np.uint8)
            if self.vacuum:
                s2[s == 0] = 0
            if self.noise > 0:
                if self.vacuum:
                    m = (s > 0) & (rng.random(s.shape) < self.noise)
                    s2[m] = rng.integers(1, self.Q, int(m.sum()), dtype=np.uint8)
                else:
                    m = rng.random(s.shape) < self.noise
                    s2[m] = rng.integers(0, self.Q, int(m.sum()), dtype=np.uint8)
            s = s2
        return s


class Tumor3D:
    """Рост «опухоли» с питательным полем: блоб с некротическим ядром.

    Клетки делятся в пустых соседей при достатке питания, потребляют его;
    при голодании становятся некротическими. Источник питания — стенки куба.
    """
    family = 'tumor'

    def __init__(self, T=40, D=0.2, cons=0.4, div=0.3, death=0.05,
                 pmax=0.6, name=None):
        self.T, self.D, self.cons, self.div, self.death = T, D, cons, div, death
        self.pmax = pmax
        self.name = name or f'tumor-T{T}-D{D}-c{cons}-v{div}'

    def run(self, L=65, seed=1):
        rng = np.random.default_rng(seed)
        s = np.zeros((L, L, L), np.uint8)
        c = L // 2
        s[c - 1:c + 2, c - 1:c + 2, c - 1:c + 2] = 1
        N = np.full((L, L, L), 0.8, np.float32)
        k = np.array([1, -2, 1], np.float32)
        idx = np.arange(L)
        for t in range(self.T):
            lap = sum(convolve1d(N, k, axis=ax, mode='wrap') for ax in range(3))
            N = np.clip(N + self.D * lap - self.cons * (s == 1) * N, 0, 1)
            # источник на стенках
            N[0, :, :] = N[-1, :, :] = 1.0
            N[:, 0, :] = N[:, -1, :] = 1.0
            N[:, :, 0] = N[:, :, -1] = 1.0
            nbr = von6(s == 1)
            empty = (s == 0) & (nbr > 0) & (N > self.div)
            prob = np.clip(0.05 + nbr / 12.0, 0, self.pmax)
            born = empty & (rng.random(s.shape) < prob)
            s[born] = 1
            s[(s == 1) & (N < self.death)] = 2
        return s


class Invasion3D:
    """Инвазионная перколяция: кластер растёт по наименьшим порогам.

    Каждой клетке дан случайный порог; фронт добавляет соседа с минимальным
    порогом. Получается ветвящаяся сеть. alpha < 1 усиливает ветвление.
    """
    family = 'invasion'

    def __init__(self, N=12000, alpha=1.0, bias=0.0, name=None):
        self.N, self.alpha, self.bias = N, alpha, bias
        self.name = name or f'invasion-N{N}-a{alpha}-b{bias}'

    def run(self, L=65, seed=1):
        rng = np.random.default_rng(seed)
        thr = rng.random((L, L, L)).astype(np.float32) ** self.alpha
        x, y, z = np.indices((L, L, L))
        thr *= 1.0 + self.bias * (x / L)
        c = L // 2
        cl = np.zeros((L, L, L), bool)
        inb = np.zeros((L, L, L), bool)
        cl[c, c, c] = True
        heap = []
        for d in DIRS:
            q = (c + d[0], c + d[1], c + d[2])
            if 0 <= q[0] < L and 0 <= q[1] < L and 0 <= q[2] < L:
                inb[q] = True
                heapq.heappush(heap, (float(thr[q]), q))
        n = 1
        while heap and n < self.N:
            _, p = heapq.heappop(heap)
            inb[p] = False
            if cl[p]:
                continue
            cl[p] = True
            n += 1
            for d in DIRS:
                q = (p[0] + d[0], p[1] + d[1], p[2] + d[2])
                if 0 <= q[0] < L and 0 <= q[1] < L and 0 <= q[2] < L \
                        and not cl[q] and not inb[q]:
                    inb[q] = True
                    heapq.heappush(heap, (float(thr[q]), q))
        return cl.astype(np.uint8)


class Bond3D:
    """Рост графа связей: активные узлы выпускают рёбра в соседей.

    Состояние — 6 бит направлений на ячейку. Сеть может ветвиться и
    замыкаться в петли; визуализируется как граф в 3D.
    """
    family = 'bond'

    def __init__(self, T=60, p=0.35, loops=True, bias=0.0, max_bonds=8000, name=None):
        self.T, self.p, self.loops, self.bias = T, p, loops, bias
        self.max_bonds = max_bonds
        self.name = name or f'bond-T{T}-p{p}-l{int(loops)}'

    def run(self, L=65, seed=1):
        rng = np.random.default_rng(seed)
        bonds = np.zeros((6, L, L, L), bool)
        c = L // 2
        for d in range(6):
            q = (c + DIRS[d][0], c + DIRS[d][1], c + DIRS[d][2])
            bonds[d, c, c, c] = True
            bonds[OPP[d], q[0], q[1], q[2]] = True
        for t in range(self.T):
            deg = bonds.sum(0)
            active = (deg >= 1) & (deg <= 2)
            h = hash3(L, t, salt=seed)
            grow = active & ((h % 1000) < self.p * 1000)
            choose = (h // 1000) % 6
            for d in range(6):
                m = grow & (choose == d)
                if not m.any():
                    continue
                # цель свободна или (если loops) уже занята, но связи нет
                tdeg = np.roll(deg, tuple(-x for x in DIRS[d]), axis=(0, 1, 2))
                if self.loops:
                    ok = (tdeg >= 0) & ~np.roll(bonds[OPP[d]], tuple(-x for x in DIRS[d]), axis=(0, 1, 2))
                else:
                    ok = tdeg == 0
                m = m & ok
                if not m.any():
                    continue
                bonds[d][m] = True
                tm = np.roll(m, tuple(-x for x in DIRS[d]), axis=(0, 1, 2))
                bonds[OPP[d]][tm] = True
            if int(bonds.sum()) >= 2 * self.max_bonds:
                break
        return bonds


def potts_walls(s: np.ndarray) -> np.ndarray:
    w = np.zeros(s.shape, bool)
    for ax in range(3):
        for sh in (1, -1):
            w |= s != np.roll(s, sh, axis=ax)
    return w.astype(np.uint8)


def bonds_to_mask(bonds: np.ndarray) -> np.ndarray:
    """Превращает граф связей в воксельную маску (узлы + середины рёбер)."""
    m = bonds.any(0).astype(np.uint8)
    for d in range(6):
        m |= bonds[d].astype(np.uint8)
    return m


def selftest():
    L = 21
    p = Potts3D(Q=4, T=5, init='block', seed_size=9).run(L)
    assert p.shape == (L, L, L)
    t = Tumor3D(T=5).run(L)
    assert set(np.unique(t)) <= {0, 1, 2}
    i = Invasion3D(N=200).run(L)
    assert 0 < i.sum() < L ** 3
    b = Bond3D(T=5, p=0.5).run(L)
    assert b.shape == (6, L, L, L) and b.any()
    print('ca3d_morph self-test: OK')


if __name__ == '__main__':
    selftest()
