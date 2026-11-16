# -*- coding: utf-8 -*-
"""3D-клеточный автомат на инфляционно расширяющейся кубической решётке.

Модель (issue #1):
  * пространство — куб из L^3 ячеек, каждая ячейка хранит B бит;
  * глобальный такт CLK: все ячейки обновляются одновременно;
  * за такт сигнал проходит не более одной ячейки (c = 1 ячейка/такт);
  * куб растёт, добавляя по слою с каждой стороны; начальная скорость роста
    больше c, затем рост затухает;
  * после остановки роста границы замыкаются периодически (тор).

Реализация хранит куб максимального размера L_max и «активную» подкубическую
область L(t), растущую симметрично вокруг зародыша. Снаружи активной области —
вакуум (0). Это позволяет менять L без перевыделения памяти.

Правила намеренно НЕ строятся как 3D-аналоги «Жизни»: основной объект —
B-битное слово ячейки, а правило комбинирует внутренние биты со сдвигами.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from scipy.ndimage import convolve1d

# --------------------------------------------------------------------------
#  Соседние суммы
# --------------------------------------------------------------------------


def box26(a: np.ndarray, mode: str = 'constant') -> np.ndarray:
    """Сумма по 3x3x3 (26 соседей Мура + сама ячейка)."""
    k = np.ones(3, dtype=np.int32)
    out = a.astype(np.int32, copy=False)
    for ax in range(3):
        out = convolve1d(out, k, axis=ax, mode=mode)
    return out


def von6(a: np.ndarray, mode: str = 'constant') -> np.ndarray:
    """Сумма по 6 фон-Неймановским соседям (без самой ячейки)."""
    k = np.array([1, 0, 1], dtype=np.int32)
    out = np.zeros(a.shape, dtype=np.int32)
    aa = a.astype(np.int32, copy=False)
    for ax in range(3):
        out += convolve1d(aa, k, axis=ax, mode=mode)
    return out


def _bits_pow2(states: int) -> np.ndarray:
    """Информационный вес состояния = число единичных бит в его записи."""
    return np.array([bin(i).count('1') for i in range(states)], dtype=np.int32)


# --------------------------------------------------------------------------
#  Инфляция
# --------------------------------------------------------------------------


@dataclass
class InflationLaw:
    """Закон роста куба L(t) (нечётное L, симметрично вокруг центра).

    kind:
      * ``exp``    — R(t) = R_max (1 - exp(-t/tau)), старт со скоростью v0*c;
      * ``power``  — R(t) = R_max (t/t_end)^p, p < 1 (замедление);
      * ``linear`` — R(t) = R_max t/t_end (постоянная скорость);
      * ``step``   — мгновенный рост до L_max при t = 0 (предел без инфляции).
    """
    kind: str
    L_max: int
    v0: float = 4.0       # начальная скорость роста радиуса, ячеек/такт (в единицах c)
    p: float = 0.5
    L0: int = 3

    def __post_init__(self) -> None:
        if self.L_max % 2 == 0:
            self.L_max += 1
        if self.L0 % 2 == 0:
            self.L0 += 1
        self.R_max = (self.L_max - 1) // 2
        self.R0 = (self.L0 - 1) // 2
        if self.kind == 'step':
            self.t_end = 0.0
        elif self.kind == 'exp':
            self.tau = max(self.R_max - self.R0, 1.0) / max(self.v0, 1e-9)
            self.t_end = self.tau * np.log(200.0)  # R = 0.995 R_max
        elif self.kind == 'power':
            self.t_end = self.p * max(self.R_max - self.R0, 1.0) / max(self.v0, 1e-9)
        elif self.kind == 'linear':
            self.t_end = max(self.R_max - self.R0, 1.0) / max(self.v0, 1e-9)
        else:
            raise ValueError(f'unknown inflation law: {self.kind}')

    def R(self, t: float) -> float:
        if self.kind == 'step' or t >= self.t_end:
            return float(self.R_max)
        if self.kind == 'exp':
            r = (self.R_max - self.R0) * (1.0 - np.exp(-t / self.tau))
        elif self.kind == 'power':
            r = (self.R_max - self.R0) * (t / self.t_end) ** self.p
        else:
            r = (self.R_max - self.R0) * (t / self.t_end)
        return self.R0 + r

    def L(self, t: float) -> int:
        r = int(round(self.R(t)))
        r = min(max(r, self.R0), self.R_max)
        return 2 * r + 1

    def speed(self, t: float) -> float:
        """Скорость роста радиуса в ячейках за такт (в единицах c)."""
        dt = 1e-3
        return (self.R(t + dt) - self.R(t)) / dt


# --------------------------------------------------------------------------
#  Правила
# --------------------------------------------------------------------------

#: индексы направлений: +x, -x, +y, -y, +z, -z
DIRS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
DIR_NAMES = ('+x', '-x', '+y', '-y', '+z', '-z')


def shift(a: np.ndarray, d: int, mode: str) -> np.ndarray:
    """Сдвиг поля на одну ячейку в направлении d (d — индекс в DIRS)."""
    return np.roll(a, DIRS[d], axis=(0, 1, 2))


def rotl(a: np.ndarray, k: int) -> np.ndarray:
    """Циклический сдвиг битов внутри B-битного слова (k влево)."""
    k %= 8
    if k == 0:
        return a.copy()
    return ((a << k) | (a >> (8 - k))) & 0xFF


class Rule:
    """Базовый класс правила. Состояние — целое, активная область — 3D массив."""

    name = 'rule'
    family = 'base'
    states = 2
    c_max = 1.0            # максимальная скорость сигнала, ячеек/такт
    needs_prev = False
    dtype = np.uint8
    description = ''
    formula = ''

    def __init__(self) -> None:
        self.bits = _bits_pow2(self.states).astype(np.int32)

    def initial(self, L0: int, rng: np.random.Generator, kind: str = 'block3',
                density: float = 1.0, value: int | None = None) -> np.ndarray:
        """Зародыш в кубе L0 (нечётный), по умолчанию — плотный блок."""
        a = np.zeros((L0, L0, L0), dtype=self.dtype)
        c = L0 // 2
        top = (self.states - 1) if value is None else value
        if kind in ('point', 'block1'):
            a[c, c, c] = top
        elif kind == 'block3':
            a[c - 1:c + 2, c - 1:c + 2, c - 1:c + 2] = top
        elif kind == 'block5':
            a[c - 2:c + 3, c - 2:c + 3, c - 2:c + 3] = top
        elif kind in ('random3', 'random5'):
            n = 3 if kind == 'random3' else 5
            m = rng.random((n, n, n)) < density
            fr = self.states if self.states > 2 else 2
            vals = rng.integers(0, fr, (n, n, n), dtype=np.uint8).astype(self.dtype)
            block = np.where(m, vals, np.zeros_like(vals)).astype(self.dtype)
            h = n // 2
            a[c - h:c + h + 1, c - h:c + h + 1, c - h:c + h + 1] = block
        elif kind == 'phase3':
            vals = rng.permutation(np.arange(self.states, dtype=np.uint8)).astype(self.dtype)
            block = np.resize(vals, (3, 3, 3))
            a[c - 1:c + 2, c - 1:c + 2, c - 1:c + 2] = block
        else:
            raise ValueError(kind)
        return a

    def step(self, a: np.ndarray, mode: str, prev: np.ndarray | None = None) -> np.ndarray:
        raise NotImplementedError


class Sprout3D(Rule):
    """SPROUT: рост из B-битного слова за счёт поворота битов и сдвигов.

        s'(x) = ROT(s(x)) XOR (s(x) OP s(x-d0)) XOR s(x-d1) XOR s(x-d2)

    Три направления d0,d1,d2 (различные), OP ∈ {AND, OR, XOR}, поворот
    ROT ∈ {влево, вправо}. Внутренняя структура слова работает как «часы»:
    именно она позволяет расти из одной клетки.
    """

    family = 'sprout'
    states = 256
    c_max = 1.0

    def __init__(self, d=(0, 2, 4), op: str = 'and', rot: int = 1, name: str | None = None):
        self.d = tuple(d)
        self.op = op
        self.rot = rot
        self.name = name or f'sprout-{op}-{"".join(str(x) for x in d)}-r{rot}'
        self.formula = (f's\' = ROT{self.rot}(s) XOR (s {op.upper()} s<<{DIR_NAMES[d[0]]}) '
                        f'XOR s<<{DIR_NAMES[d[1]]} XOR s<<{DIR_NAMES[d[2]]}')
        self.description = 'нелинейный рост B-битного слова: поворот битов + AND/OR + сдвиги'
        super().__init__()

    def step(self, a, mode, prev=None):
        r = rotl(a, self.rot)
        t = r ^ shift(a, self.d[1], mode) ^ shift(a, self.d[2], mode)
        x = shift(a, self.d[0], mode)
        if self.op == 'and':
            t ^= a & x
        elif self.op == 'or':
            t ^= a | x
        elif self.op == 'xor':
            t ^= a ^ x
        else:
            raise ValueError(self.op)
        return t


class Helix3D(Rule):
    """HELIX: линейное правило с «винтовым» переносом информации.

        s'(x) = ROT(s(x)) XOR s(x-1,y,z) XOR ROT^-1(s(x,y+1,z)) XOR s(x,y,z-1)

    Линейно над GF(2): сумма двух решений — решение (суперпозиция).
    """

    family = 'helix'
    states = 256
    c_max = 1.0

    def __init__(self, name: str = 'helix'):
        self.name = name
        self.formula = "s' = ROT1(s) XOR s<<+x XOR ROT7(s<<-y) XOR s<<+z"
        self.description = 'линейное над GF(2) правило: биты вращаются при переносе'
        super().__init__()

    def step(self, a, mode, prev=None):
        return rotl(a, 1) ^ shift(a, 0, mode) ^ rotl(shift(a, 3, mode), 7) ^ shift(a, 4, mode)


class Topple3D(Rule):
    """TOPPLE: параллельная песочница на B битах.

        q = s // 6;   s' = s - 6q + сумма q по 6 соседям

    Сумма s (число бит-зёрен) сохраняется точно. Зародыш — одна клетка,
    несущая N бит; правило перераспределяет их, пока слой не станет < 6.
    """

    family = 'topple'
    states = 65536
    dtype = np.uint16
    c_max = 1.0

    def __init__(self, threshold: int = 6, name: str = 'topple-6'):
        self.threshold = threshold
        self.name = name
        self.formula = "q = s div 6;  s' = s - 6q + Σ_neighbors q"
        self.description = 'консервативная параллельная песочница: N бит в одной клетке'
        self.states = 65536
        super().__init__()
        self.bits = np.arange(self.states, dtype=np.int32)

    def initial(self, L0, rng, kind='point', density=1.0, value=None):
        a = np.zeros((L0, L0, L0), dtype=self.dtype)
        c = L0 // 2
        a[c, c, c] = self.states - 1 if value is None else value
        return a

    def step(self, a, mode, prev=None):
        q = (a // self.threshold).astype(np.int32)
        out = a.astype(np.int32) - self.threshold * q + von6(q, mode)
        return np.clip(out, 0, self.states - 1).astype(self.dtype)


class SecondOrder(Rule):
    """Обратимое правило второго порядка: s_{t+1} = F(s_t) XOR s_{t-1}."""

    family = 'reversible'
    needs_prev = True

    def __init__(self, base: Rule, name: str | None = None):
        self.base = base
        self.states = base.states
        self.dtype = base.dtype
        self.c_max = base.c_max
        self.name = name or f'2nd-{base.name}'
        self.formula = f's(t+1) = F(s(t)) XOR s(t-1),  F = {base.name}'
        self.description = f'обратимое второго порядка на базе {base.name}'
        super().__init__()

    def initial(self, *args, **kwargs):
        return self.base.initial(*args, **kwargs)

    def step(self, a, mode, prev=None):
        f = self.base.step(a, mode)
        if prev is None:
            prev = np.zeros_like(a)
        return f ^ prev


class Cyclic3D(Rule):
    """Циклический КА (эталон семейства Гринберга–Хастингса)."""

    family = 'cyclic'

    def __init__(self, states: int = 4, neighbors: str = 'von6', name: str | None = None):
        self.states = states
        self.neighbors = neighbors
        self.name = name or f'cyclic{states}-{neighbors}'
        self.c_max = 1.0 if neighbors == 'von6' else np.sqrt(3.0)
        self.description = (f'циклический КА с {states} фазами: фаза j сдвигается в j+1, '
                            f'если рядом есть сосед в фазе j+1')
        self.formula = "s' = s+1, если есть сосед s+1 (mod S)"
        self.dtype = np.uint8
        super().__init__()

    def step(self, a, mode, prev=None):
        out = a.copy()
        for j in range(self.states):
            nxt = (j + 1) % self.states
            s = von6(a == nxt, mode) if self.neighbors == 'von6' else box26(a == nxt, mode) - (a == nxt)
            out[(a == j) & (s > 0)] = nxt
        return out


class Life3D(Rule):
    """3D-аналог «Жизни» — только контрольное семейство для сравнения."""

    family = 'life'
    states = 2
    c_max = np.sqrt(3.0)

    def __init__(self, birth: Iterable[int], survive: Iterable[int], name: str | None = None):
        self.birth = tuple(sorted(birth))
        self.survive = tuple(sorted(survive))
        self.name = name or f'B{",".join(map(str, self.birth))}/S{",".join(map(str, self.survive))}'
        self._birth = np.zeros(27, dtype=bool)
        self._surv = np.zeros(27, dtype=bool)
        self._birth[list(self.birth)] = True
        self._surv[list(self.survive)] = True
        self.description = ('тоталистическое правило 3D-Жизни: рождение при '
                            f'{self.birth}, выживание при {self.survive} соседях из 26')
        self.formula = "s' = B[n] при s=0, S[n] при s=1 (n — число соседей из 26)"
        super().__init__()

    def step(self, a, mode, prev=None):
        n = box26(a, mode) - a
        out = np.where(a == 0, self._birth[n], self._surv[n])
        return out.astype(np.uint8)


# --------------------------------------------------------------------------
#  Симуляция
# --------------------------------------------------------------------------


class Simulation:
    def __init__(self, rule: Rule, law: InflationLaw, seed_kind: str = 'block3',
                 density: float = 0.5, seed: int = 1, L0: int | None = None,
                 value: int | None = None):
        self.rule = rule
        self.law = law
        self.L_max = law.L_max
        self.center = self.L_max // 2
        rng = np.random.default_rng(seed)
        L0 = L0 or law.L0
        if L0 % 2 == 0:
            L0 += 1
        self.L = L0
        self.t = 0
        self.grid = np.zeros((self.L_max,) * 3, dtype=rule.dtype)
        sub = rule.initial(L0, rng, seed_kind, density, value)
        self._view()[:] = sub
        self.prev = np.zeros_like(self.grid)
        if rule.needs_prev:
            self.prev[self._idx()] = 0

    # ---- геометрия ----
    def _sl(self):
        r = self.L // 2
        return slice(self.center - r, self.center + r + 1)

    def _idx(self):
        s = self._sl()
        return (s, s, s)

    def _view(self):
        return self.grid[self._idx()]

    def _prev_view(self):
        return self.prev[self._idx()]

    def periodic(self) -> bool:
        return self.t >= self.law.t_end

    # ---- шаг ----
    def step(self) -> None:
        mode = 'wrap' if self.periodic() else 'constant'
        a = self._view()
        if self.rule.needs_prev:
            new = self.rule.step(a, mode, self._prev_view())
        else:
            new = self.rule.step(a, mode)
        self.prev[self._idx()] = a
        self.grid[self._idx()] = new
        self.t += 1
        self.L = self.law.L(self.t)

    def run(self, T: int, snapshot: Callable | None = None, every: int = 1) -> list:
        snaps = []
        for _ in range(T):
            self.step()
            if snapshot is not None and (self.t % every == 0):
                snaps.append(snapshot(self))
        return snaps

    # ---- метрики ----
    def state(self) -> np.ndarray:
        return self._view().copy()

    def info_bits(self, a: np.ndarray | None = None) -> int:
        a = self._view() if a is None else a
        return int(self.rule.bits[a].sum())

    def occupancy(self, a: np.ndarray | None = None) -> float:
        a = self._view() if a is None else a
        return float(np.count_nonzero(a) / a.size)


# --------------------------------------------------------------------------
#  Самопроверки
# --------------------------------------------------------------------------


def _selftest() -> None:
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, (9, 9, 9)).astype(np.uint8)
    # 1) box26 = сумма 27 ячеек
    brute = np.zeros_like(a, dtype=np.int32)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                brute += np.roll(a, (i - 1, j - 1, k - 1), axis=(0, 1, 2))
    assert np.array_equal(box26(a, 'wrap'), brute)
    # 2) von6 = 6 соседей
    brute6 = sum(np.roll(a, s, axis=ax) for ax in range(3) for s in (1, -1))
    assert np.array_equal(von6(a, 'wrap'), brute6)

    # 3) причинность: фронт не быстрее c
    L = 65
    for rule, c in [(Cyclic3D(4), 1.0), (Life3D([1], [6]), np.sqrt(3.0))]:
        law = InflationLaw('step', L)
        sim = Simulation(rule, law, seed_kind='point', L0=3, value=1)
        sim.run(20)
        a = sim.state()
        idx = np.argwhere(a > 0)
        ctr = np.array(a.shape) // 2
        r = np.sqrt(((idx - ctr) ** 2).sum(1)).max()
        assert r <= c * sim.t + 1e-9, (rule.name, r, c * sim.t)

    # 4) обратимость второго порядка
    base = Sprout3D()
    rule = SecondOrder(base)
    law = InflationLaw('step', 33)
    sim = Simulation(rule, law, seed_kind='block3', L0=33)
    states = [sim.state()]
    for _ in range(8):
        sim.step()
        states.append(sim.state())
    ok = True
    for t in range(7, 0, -1):
        a = states[t]
        f = base.step(a, 'wrap')
        back = states[t + 1] ^ f
        ok &= np.array_equal(back, states[t - 1])
    assert ok, 'second-order rule is not reversible'

    # 6) TOPPLE сохраняет сумму бит
    law = InflationLaw('step', 65)
    rule = Topple3D()
    sim = Simulation(rule, law, seed_kind='point', L0=65, value=20000)
    s0 = sim.info_bits()
    sim.run(40)
    assert sim.info_bits() == s0, 'TOPPLE does not conserve bits'

    # 7) инфляция: L монотонен, старт быстрее c, финиш L_max
    for kind in ['exp', 'power', 'linear', 'step']:
        law = InflationLaw(kind, 65, v0=4.0)
        Ls = [law.L(t) for t in range(0, 120)]
        assert Ls[-1] == 65
        assert all(b >= a for a, b in zip(Ls, Ls[1:])), kind
        assert law.speed(0.5) >= 1.0 or kind == 'step', (kind, law.speed(0.5))
    print('ca3d self-test: OK')


if __name__ == '__main__':
    _selftest()
