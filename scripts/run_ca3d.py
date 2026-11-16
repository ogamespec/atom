# -*- coding: utf-8 -*-
"""Эксперименты issue #1: 3D-клеточный автомат на инфляционной решётке.

Стадии:
  1. Подсчёт пространства правил (комбинаторика + лемма Бёрнсайда).
  2. Зародыш: что делает 1 клетка — полный перебор 729 тоталистических
     одноуровневых правил 3D-Жизни (контроль, не основа исследования).
  3. Свип грамматики SPROUT: 720 правил (направления x операция x поворот).
  4. Глубокие прогоны 6 семейств: метрики, кадры, GIF, срезы, масштабы.
  5. Сравнение законов инфляции.
  6. Графики, report.json, проверки.

Запуск:  python scripts/run_ca3d.py            (всё, ~5 минут)
         python scripts/run_ca3d.py --fast     (без тяжёлых стадий)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zlib
from collections import Counter, deque
from multiprocessing import Pool
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from src.ca3d import (DIR_NAMES, DIRS, Cyclic3D, Helix3D, InflationLaw, Life3D,
                      SecondOrder, Simulation, Sprout3D, Topple3D, box26, von6)
from src.ca3d_metrics import evaluate
from src.ca3d_render import frame, render_voxels, slice_mosaic

OUT = ROOT / 'results' / 'ca3d'
FRAMES = OUT / 'frames'
OUT.mkdir(parents=True, exist_ok=True)
FRAMES.mkdir(exist_ok=True)

REPORT: dict = {'meta': {}, 'checks': []}
CHECKS: list = []


def check(name: str, ok: bool, value) -> None:
    CHECKS.append({'name': name, 'ok': bool(ok), 'value': value})
    print(('PASS  ' if ok else 'FAIL  ') + name + f'   ->   {value}', flush=True)


# ==========================================================================
#  1. Пространство правил
# ==========================================================================


def _cube_groups() -> tuple[list[np.ndarray], list[np.ndarray]]:
    """(24 собственных вращения, 48 вращений и отражений) куба."""
    import itertools
    proper, full = [], []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            M = np.zeros((3, 3), int)
            for i, p in enumerate(perm):
                M[i, p] = signs[i]
            full.append(M)
            if round(np.linalg.det(M)) == 1:
                proper.append(M)
    return proper, full


def _cycle_counts(group: list[np.ndarray]) -> list[int]:
    pos = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
    cycles = []
    for M in group:
        seen, c = set(), 0
        for idx, p in enumerate(pos):
            if idx in seen:
                continue
            c += 1
            q = tuple(p)
            while True:
                j = int(np.where((pos == q).all(1))[0][0])
                if j in seen:
                    break
                seen.add(j)
                q = tuple(M @ q)
        cycles.append(c)
    return cycles


def _burnside(cycles: list[int], power: str) -> float:
    """log10 (1/|G|) Σ 2^(2^c)  (power='rule') или (1/|G|) Σ 2^c (power='config')."""
    if power == 'rule':
        logs = [2.0 ** c * math.log10(2) for c in cycles]
    else:
        logs = [c * math.log10(2) for c in cycles]
    mx = max(logs)
    return mx + math.log10(sum(10 ** (v - mx) for v in logs)) - math.log10(len(cycles))


def rule_space_counts() -> dict:
    print('== 1. Пространство правил ==', flush=True)
    log2 = math.log10(2)
    proper, full = _cube_groups()
    cyc24, cyc48 = _cycle_counts(proper), _cycle_counts(full)
    orb24 = _burnside(cyc24, 'config')
    orb48 = _burnside(cyc48, 'config')
    rule_orb24 = _burnside(cyc24, 'rule')
    rule_orb48 = _burnside(cyc48, 'rule')
    log_general = (2 ** 27) * log2
    log_b8 = 8 * (2 ** 216) * log2
    log_b8_6 = 8 * (2 ** 56) * log2
    counts = {
        'neighborhood_cells': 27,
        'general_binary': {'log10': log_general, 'digits': int(log_general) + 1,
                           'formula': '2^(2^27)'},
        'six_neighbour_binary': {'value': str(2 ** 128), 'log10': 128 * log2,
                                 'formula': '2^(2^7): 6 соседей + сама ячейка'},
        'totalistic_binary_bs': {'value': str(2 ** 54), 'log10': 54 * log2,
                                 'formula': 'B и S — по 2^27 подмножеств'},
        'totalistic_binary_fn': {'value': str(2 ** 54), 'log10': 54 * log2,
                                 'formula': 'f(s,n): 2·27 входов'},
        'single_totalistic': {'value': str(2 ** 27), 'log10': 27 * log2,
                              'formula': 'f(k), k=0..26'},
        'bays_intervals': {'value': '142884', 'log10': math.log10(142884),
                           'formula': 'C(28,2)^2 — интервальные правила (a,b/c,d)'},
        'b8_27': {'log10': log_b8, 'digits': int(log_b8) + 1,
                  'formula': '256^(256^27) = 2^(8*2^216)'},
        'b8_6': {'log10': log_b8_6, 'digits': int(log_b8_6) + 1,
                 'formula': '256^(256^7) = 2^(8*2^56)'},
        'block_bijections_binary': {'log10': math.lgamma(257) / math.log(10),
                                    'formula': '(2^8)! = 256! на блок 2x2x2'},
        'linear_stencils': {'value': str(2 ** 27), 'log10': 27 * log2},
        'invertible_linear': {'value': '27', 'note': 'только сдвиги (мономы) на бесконечной решётке'},
        'symmetry': {
            'proper_order': len(proper), 'full_order': len(full),
            'cycles_24': cyc24, 'cycles_48': cyc48,
            'config_orbits_24': int(round(10 ** orb24)),
            'config_orbits_48': int(round(10 ** orb48)),
            'invariant_rules_24_log10': (10 ** orb24) * log2,
            'invariant_rules_48_log10': (10 ** orb48) * log2,
            'rule_orbits_24_log10': rule_orb24, 'rule_orbits_48_log10': rule_orb48,
        },
        'second_order_reversible': {'same_as': 'general_binary',
                                    'note': 'любое F даёт обратимое s(t+1)=F(s(t)) XOR s(t-1)'},
    }
    print(f"  general 2^(2^27): 10^{log_general:.3e} ({int(log_general)+1} цифр)")
    print(f"  инвариантных (48): 10^{(10**orb48)*log2:.0f}; (24): 10^{(10**orb24)*log2:.0f}")
    print(f"  орбит правил (48): 10^{rule_orb48:.3f}; (24): 10^{rule_orb24:.3f}")
    print(f"  B=8, 27 ячеек: 10^{log_b8:.3e} цифр")
    REPORT['counts'] = counts
    return counts


# ==========================================================================
#  2. Зародыш: 729 правил, точка
# ==========================================================================


def _life_point_worker(args):
    b, s = args
    L, T = 33, 40
    birth = np.zeros(27, bool)
    surv = np.zeros(27, bool)
    birth[b] = True
    surv[s] = True
    a = np.zeros((L, L, L), np.uint8)
    a[L // 2, L // 2, L // 2] = 1
    pops = []
    for t in range(T):
        n = box26(a) - a
        a = np.where(a == 0, birth[n], surv[n]).astype(np.uint8)
        pops.append(int(a.sum()))
    occ = float((a > 0).mean())
    r = 0.0
    if a.any():
        idx = np.argwhere(a > 0) - L // 2
        r = float(np.sqrt((idx ** 2).sum(1)).max())
    # классификация
    if a.sum() == 0:
        cls = 'смерть'
    elif occ > 0.55:
        cls = 'взрыв/игниция'
    elif max(pops[-10:]) - min(pops[-10:]) == 0:
        cls = 'застыло'
    else:
        cls = 'рост'
    return {'B': b, 'S': s, 'pop': int(a.sum()), 'occ': occ,
            'radius': r, 'class': cls, 'hist': pops[-1]}


def seed_study() -> dict:
    print('== 2. Зародыш 1x1x1: 729 правил 3D-Жизни (контроль) ==', flush=True)
    jobs = [(b, s) for b in range(27) for s in range(27)]
    t0 = time.time()
    with Pool(24) as p:
        res = p.map(_life_point_worker, jobs, chunksize=8)
    cnt = Counter(r['class'] for r in res)
    # только B0 (игниция вакуума) и B1 (оболочка из 26 клеток) дают ненулевой рост
    nonzero = [r for r in res if r['pop'] > 0]
    nontrivial = [r for r in res if r['class'] == 'рост']
    by_B = Counter(r['B'] for r in nonzero)
    print(f'  {len(jobs)} правил за {time.time()-t0:.1f} с: {dict(cnt)}')
    print(f'  ненулевых {len(nonzero)}, из них B=0: {by_B.get(0,0)}, B=1: {by_B.get(1,0)}')
    b0 = sum(1 for r in res if r['B'] == 0)
    b1 = sum(1 for r in res if r['B'] == 1)
    single = sum(1 for r in res if r['B'] >= 2 and r['S'] == 0)
    by_rule = {
        'игниция вакуума (B=0)': b0,
        'оболочка 26 клеток (B=1)': b1,
        'одна застывшая клетка (B≥2, S=0)': single,
        'смерть (остальные)': len(jobs) - b0 - b1 - single,
    }
    data = {'n_rules': len(jobs), 'categories': dict(cnt), 'by_rule': by_rule,
            'nonzero': len(nonzero), 'growing': len(nontrivial),
            'by_B_nonzero': {str(k): v for k, v in sorted(by_B.items())},
            'grid': [[0] * 27 for _ in range(27)]}
    for r in res:
        if r['B'] == 0:
            code = 2          # игниция вакуума
        elif r['B'] == 1:
            code = 3          # оболочка 26 клеток
        elif r['S'] == 0:
            code = 1          # одна застывшая клетка
        else:
            code = 0          # смерть
        data['grid'][r['B']][r['S']] = code
    data['examples'] = {
        'B1/S6': next((r for r in res if r['B'] == 1 and r['S'] == 6), None),
        'B0/S17': next((r for r in res if r['B'] == 0 and r['S'] == 17), None),
    }
    # сохраняем только компактные данные
    data['examples'] = {k: ({kk: vv for kk, vv in v.items() if kk != 'hist'} if v else None)
                        for k, v in data['examples'].items()}
    REPORT['seed'] = data
    # карта
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    cmap = matplotlib.colors.ListedColormap(['#1b1b1b', '#2b6cb0', '#c05621', '#2f855a'])
    ax.imshow(np.array(data['grid']).T, origin='lower', cmap=cmap, vmin=0, vmax=3,
              interpolation='nearest')
    ax.set_xlabel('B — число соседей для рождения')
    ax.set_ylabel('S — число соседей для выживания')
    ax.set_title('Зародыш 1×1×1: исход 729 правил B/S (3D-Жизнь, контроль)')
    handles = [matplotlib.patches.Patch(color=c, label=l) for c, l in
               zip(['#1b1b1b', '#2b6cb0', '#c05621', '#2f855a'],
                   ['смерть', 'одна застывшая клетка', 'игниция вакуума (B=0)', 'оболочка (B=1)'])]
    ax.legend(handles=handles, loc='upper right', fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / 'f2_seed_map.png', dpi=150)
    plt.close(fig)
    return data


# ==========================================================================
#  3. Свип грамматики SPROUT
# ==========================================================================


def _sprout_worker(args):
    dirs, op, rot = args
    L, T = 65, 96
    rule = Sprout3D(d=dirs, op=op, rot=rot)
    law = InflationLaw('exp', L, v0=5.0, L0=7)
    sim = Simulation(rule, law, seed_kind='block3', L0=7)
    acts = []
    prev = sim.state()
    for _ in range(T):
        sim.step()
        cur = sim.state()
        if prev.shape == cur.shape:
            acts.append(float((prev != cur).mean()))
        prev = cur
    a = sim.state()
    act = float(np.mean(acts[-12:])) if acts else 0.0
    m = evaluate(a, rule, T, activity=act, with_clusters=True)
    keep = ('occupancy', 'entropy_norm', 'compression', 'box_dim', 'n_clusters',
            'max_cluster', 'activity', 'persistence', 'richness', 'class',
            'bits_total', 'radius_90')
    rec = {k: (float(m[k]) if isinstance(m[k], (int, float, np.floating)) else m[k])
           for k in keep if k in m}
    rec.update({'dirs': list(dirs), 'op': op, 'rot': rot,
                'name': rule.name, 'L': int(sim.L)})
    return rec


def sprout_sweep() -> dict:
    print('== 3. Свип SPROUT: 720 правил ==', flush=True)
    jobs = []
    import itertools
    for dirs in itertools.permutations(range(6), 3):
        for op in ('and', 'or', 'xor'):
            for rot in (1, 7):
                jobs.append((dirs, op, rot))
    t0 = time.time()
    with Pool(24) as p:
        res = p.map(_sprout_worker, jobs, chunksize=4)
    print(f'  {len(res)} правил за {time.time()-t0:.1f} с')
    alive = [r for r in res if r['occupancy'] > 1e-6]
    by_cls = Counter(r['class'] for r in res)
    print('  классы:', dict(by_cls))
    top = sorted(res, key=lambda r: -r['richness'])[:12]
    data = {'n_rules': len(res), 'classes': dict(by_cls),
            'n_alive': len(alive), 'top': top,
            'occupancy_top': float(np.percentile([r['occupancy'] for r in res], 99)),
            'results': [{k: r[k] for k in ('name', 'op', 'rot', 'dirs', 'occupancy',
                                          'richness', 'activity', 'box_dim', 'class',
                                          'n_clusters', 'compression')} for r in res]}
    REPORT['sweep'] = data
    # график: занятость vs богатство
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    colors = {'and': '#e53e3e', 'or': '#3182ce', 'xor': '#38a169'}
    for op in ('and', 'or', 'xor'):
        pts = [r for r in res if r['op'] == op]
        ax.scatter([max(r['occupancy'], 1e-5) for r in pts],
                   [r['richness'] for r in pts], s=14, alpha=0.55,
                   c=colors[op], label=f'OP = {op.upper()}')
    ax.set_xscale('log')
    ax.set_xlabel('занятость (доля ненулевых ячеек)')
    ax.set_ylabel('эвристический индекс богатства')
    ax.set_title('Свип 720 правил SPROUT: структура против заполнения')
    ax.legend()
    for r in top[:5]:
        ax.annotate(r['name'].replace('sprout-', ''), (max(r['occupancy'], 1e-5), r['richness']),
                    fontsize=7, xytext=(4, 3), textcoords='offset points')
    fig.tight_layout()
    fig.savefig(OUT / 'f3_sweep.png', dpi=150)
    plt.close(fig)
    return data


# ==========================================================================
#  4. Глубокие прогоны
# ==========================================================================


def deep_rules() -> list[tuple[str, object, dict, str]]:
    return [
        ('helix', Helix3D(), dict(seed_kind='random5', density=0.5), 'линейное хиральное'),
        ('sprout', Sprout3D(d=(0, 2, 4), op='and', rot=1), dict(seed_kind='block3'), 'нелинейный рост'),
        ('sprout_or', Sprout3D(d=(0, 3, 4), op='or', rot=1), dict(seed_kind='block3'), 'рост через OR'),
        ('topple', Topple3D(), dict(seed_kind='point', value=30000), 'консервативная песочница'),
        ('cyclic4', Cyclic3D(4), dict(seed_kind='phase3'), 'циклический КА (эталон)'),
        ('twin', SecondOrder(Sprout3D()), dict(seed_kind='block3'), 'обратимое 2-го порядка'),
    ]


def _crop_to_content(g: np.ndarray, margin: int = 3):
    """Обрезает поле до куба вокруг занятой области (для финальных кадров)."""
    idx = np.argwhere(g > 0)
    if len(idx) == 0:
        return g, g.shape[0]
    lo, hi = idx.min(0), idx.max(0)
    c = (lo + hi) // 2
    side = int(max(hi - lo) + 2 * margin)
    side = max(side, 9)
    side = min(side, g.shape[0])
    lo = np.maximum(c - side // 2, 0)
    hi = np.minimum(lo + side, g.shape[0])
    lo = hi - side
    return g[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], side


def run_deep(fast: bool = False) -> dict:
    print('== 4. Глубокие прогоны ==', flush=True)
    L_max = 97 if fast else 321
    T = 128 if fast else 64
    v0 = 40.0
    checkpoints = [0, 8, 16, 24, 32, 48, T]
    frame_times = [0, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, T]
    if fast:
        frame_times = [0, 4, 8, 16, 32, 64, 96, T]
    out = {}
    for name, rule, kw, desc in deep_rules():
        t0 = time.time()
        law = InflationLaw('exp', L_max, v0=v0, L0=7)
        sim = Simulation(rule, law, **kw)
        bits_series, act_series, front_series = [], [], []
        metrics = []
        frames_saved = []
        prev = sim.state()
        next_cp = 0
        for t in range(T + 1):
            if t in checkpoints:
                cur = sim.state()
                cc = cur.shape[0] // 2
                rr = min(int(t) + 1, cc)
                causal = cur[cc - rr:cc + rr + 1, cc - rr:cc + rr + 1, cc - rr:cc + rr + 1]
                m = evaluate(causal, rule, t,
                             activity=float(np.mean(act_series[-16:])) if act_series else 0.0)
                m['full_occupancy'] = float((cur > 0).mean())
                m['real_time_s'] = None
                metrics.append(m)
            if t in frame_times:
                st = sim.state()
                if name == 'cyclic4':
                    g = st
                    norm = 3.0
                else:
                    g = rule.bits[st]
                    norm = 8.0 if rule.states == 256 else float(np.percentile(g[g > 0], 99)) if g.any() else 1.0
                img = render_voxels(g, size=448, canvas_L=sim.L,
                                    region='full',
                                    cmap='twilight' if name == 'cyclic4' else 'inferno',
                                    value_norm=norm,
                                    front_radius=min(t, sim.L / 2),
                                    front_time=t, inset=True,
                                    legend_label=('фаза' if name == 'cyclic4' else
                                                  ('зёрна' if name == 'topple' else 'бит')),
                                    label=f'{name}  L={sim.L}  t={t}')
                fn = FRAMES / f'{name}_{t:03d}.png'
                Image.fromarray(img).save(fn)
                frames_saved.append((t, fn))
            if t == T:
                break
            prev = sim.state()
            sim.step()
            cur = sim.state()
            if prev.shape == cur.shape:
                act_series.append(float((prev != cur).mean()))
            else:
                act_series.append(0.0)
            bits_series.append(int(rule.bits[cur].sum()))
            c = cur.shape[0] // 2
            occ_idx = np.argwhere(cur > 0)
            if len(occ_idx):
                rr = np.sqrt(((occ_idx - c) ** 2).sum(1)).max()
            else:
                rr = 0.0
            front_series.append(float(rr))
        final = sim.state()
        # срез-мозаика и финальный крупный кадр
        slice_mosaic(final, OUT / f'd_{name}_slices.png', n=6,
                     cmap='twilight' if name == 'cyclic4' else 'inferno',
                     title=f'{name}: срезы z, t={T}, L={sim.L}')
        if name == 'cyclic4':
            gfin, norm = final, 3.0
        else:
            gfin = rule.bits[final]
            norm = 8.0 if rule.states == 256 else float(np.percentile(gfin[gfin > 0], 99)) if gfin.any() else 1.0
        gview = gfin
        side = gfin.shape[0]
        region = 'full'
        big = render_voxels(gview, size=640, canvas_L=side, region=region,
                            cmap='twilight' if name == 'cyclic4' else 'inferno',
                            value_norm=norm,
                            front_radius=min(T, side / 2), front_time=T, inset=True,
                            legend_label=('фаза' if name == 'cyclic4' else
                                          ('зёрна' if name == 'topple' else 'бит')),
                            label=f'{name}  L={sim.L}  t={T}')
        Image.fromarray(big).save(OUT / f'd_{name}_final.png')
        big_cut = render_voxels(gview, size=640, canvas_L=side, cutaway=True,
                                cmap='twilight' if name == 'cyclic4' else 'inferno',
                                value_norm=norm,
                                front_radius=min(T, side / 2), front_time=T,
                                legend_label=('фаза' if name == 'cyclic4' else
                                              ('зёрна' if name == 'topple' else 'бит')),
                                label=f'{name} (cutaway)  L={sim.L}  t={T}')
        Image.fromarray(big_cut).save(OUT / f'd_{name}_cut.png')
        # «микроскоп»: блок 24x24x24, где каждая ячейка — отдельный кубик
        Z = 24
        if gfin.shape[0] >= Z:
            from scipy.ndimage import uniform_filter
            occf = (gfin > 0).astype(np.float32)
            dens = uniform_filter(occf, size=Z, mode='constant')
            ctr = np.array(np.unravel_index(int(np.argmax(dens)), dens.shape))
            lo = np.clip(ctr - Z // 2, 0, gfin.shape[0] - Z)
            zoom = gfin[lo[0]:lo[0] + Z, lo[1]:lo[1] + Z, lo[2]:lo[2] + Z]
            zimg = render_voxels(zoom, size=560, canvas_L=Z, cell=0.78, grid_step=4,
                                 region='full',
                                 cmap='twilight' if name == 'cyclic4' else 'inferno',
                                 value_norm=norm,
                                 label=f'{name}  микроскоп {Z}³  t={T}')
            Image.fromarray(zimg).save(OUT / f'd_{name}_zoom.png')
        # вращение камеры: 24 кадра
        rot = []
        for k in range(24):
            az = 12 + 58 * k / 23
            im = render_voxels(gview, size=384, canvas_L=side, azim=az, elev=22,
                               region=region,
                               cmap='twilight' if name == 'cyclic4' else 'inferno',
                               value_norm=norm,
                               front_radius=min(T, side / 2), front_time=T,
                               inset=True, label=f'{name}  L={sim.L}  t={T}')
            rot.append(Image.fromarray(im))
        rot[0].save(OUT / f'd_{name}_rotate.gif', save_all=True,
                    append_images=rot[1:], duration=110, loop=0)
        # масштабная серия: центральные вырезки L, L/2, L/4, L/8
        scales = []
        side = final.shape[0]
        for frac in (1.0, 0.5, 0.25, 0.125):
            s = max(9, int(side * frac) // 2 * 2 + 1)
            if s > side:
                s = side
            o = (side - s) // 2
            crop = final[o:o + s, o:o + s, o:o + s]
            if name == 'cyclic4':
                gc, nn = crop, 3.0
            else:
                gc = rule.bits[crop]
                nn = 8.0 if rule.states == 256 else float(np.percentile(gc[gc > 0], 99)) if gc.any() else 1.0
            img = render_voxels(gc, size=320, canvas_L=s, cutaway=False,
                                region='full',
                                cmap='twilight' if name == 'cyclic4' else 'inferno',
                                value_norm=nn, label=f'{s} ячеек')
            scales.append((s, Image.fromarray(img)))
        fig, axes = plt.subplots(2, 2, figsize=(9, 9))
        for ax, (s, im) in zip(axes.ravel(), scales):
            ax.imshow(im)
            ax.set_title(f'масштаб: {s} ячеек/сторона', fontsize=9)
            ax.axis('off')
        fig.suptitle(f'{name}: структура на разных масштабах (t={T})', fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / f'd_{name}_scales.png', dpi=140)
        plt.close(fig)
        # GIF из кадров
        gif_frames = [Image.open(fn).convert('RGB').resize((384, 384), Image.LANCZOS)
                      for _, fn in frames_saved]
        gif_frames[0].save(OUT / f'd_{name}.gif', save_all=True,
                           append_images=gif_frames[1:], duration=170, loop=0)
        if name == 'cyclic4':
            picks = [8, 16, 32, 64]
            fig, axes = plt.subplots(2, 2, figsize=(8.6, 8.6))
            for ax, tt in zip(axes.ravel(), picks):
                f = FRAMES / f'{name}_{tt:03d}.png'
                if f.exists():
                    ax.imshow(Image.open(f))
                    ax.set_title(f't = {tt}', fontsize=9)
                ax.axis('off')
            fig.suptitle('CYCLIC4: фронт r = c·t и фазовая волна в центре',
                         fontsize=11)
            fig.tight_layout()
            fig.savefig(OUT / 'f_front.png', dpi=150)
            plt.close(fig)
        out[name] = {
            'description': desc, 'metrics': metrics,
            'bits_series': bits_series, 'activity_series': act_series,
            'front_series': front_series, 'frames': [t for t, _ in frames_saved],
            'L_max': L_max, 'T': T, 'c_max': rule.c_max,
            'run_time_s': round(time.time() - t0, 1),
        }
        print(f'  {name:8s} {time.time()-t0:6.1f}s  L={sim.L}  '
              f'bits={metrics[-1]["bits_total"]}  occ={metrics[-1]["occupancy"]:.4f}  '
              f'rich={metrics[-1]["richness"]:.2f}  class={metrics[-1]["class"]}', flush=True)
    REPORT['deep'] = out
    return out


# ==========================================================================
#  5. Законы инфляции
# ==========================================================================


def inflation_study() -> dict:
    print('== 5. Законы инфляции ==', flush=True)
    L_max, T = 97, 128
    out = {}
    laws = [('exp', 6.0), ('power', 5.0), ('linear', 5.0), ('step', 0.0), ('linear_slow', 0.25)]
    for kind, v0 in laws:
        if kind == 'step':
            law = InflationLaw('step', L_max, L0=7)
        else:
            law = InflationLaw('linear' if kind == 'linear_slow' else kind, L_max,
                               v0=v0, L0=7)
        rule = Sprout3D(d=(0, 2, 4), op='and', rot=1)
        sim = Simulation(rule, law, seed_kind='block3', L0=7)
        Ls, dens, acts, fronts = [], [], [], []
        prev = sim.state()
        for t in range(T + 1):
            cur = sim.state()
            Ls.append(sim.L)
            dens.append(float((cur > 0).mean()))
            if t > 0 and prev.shape == cur.shape:
                acts.append(float((prev != cur).mean()))
            idx = np.argwhere(cur > 0)
            fronts.append(float(np.sqrt(((idx - cur.shape[0] // 2) ** 2).sum(1)).max())
                          if len(idx) else 0.0)
            if t == T:
                break
            prev = cur
            sim.step()
        img = frame(sim.state(), sim.L, T, size=448, canvas_L=L_max)
        Image.fromarray(img).save(OUT / f'i_{kind}.png')
        out[kind] = {'L': Ls, 'density': dens, 'activity': acts, 'front': fronts,
                     't_end': law.t_end, 'v0': law.speed(0.01),
                     'L_final': sim.L, 'final_occ': dens[-1],
                     'state_hash': zlib.crc32(sim.state().tobytes())}
        print(f'  {kind:6s} t_end={law.t_end:5.1f} v0={law.speed(0.01):5.2f}c '
              f'L_final={sim.L} occ={dens[-1]:.4f}', flush=True)
    REPORT['inflation'] = out
    # график
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    colors = {'exp': '#e53e3e', 'power': '#3182ce', 'linear': '#38a169', 'step': '#805ad5', 'linear_slow': '#d69e2e'}
    for kind in out:
        axes[0].plot(out[kind]['L'], color=colors[kind], label=kind)
        axes[1].plot(out[kind]['front'], color=colors[kind], label=kind)
    axes[0].plot([0, T], [0, T], 'k--', lw=1, label='c·t (световой конус)')
    axes[0].set_xlabel('такт t')
    axes[0].set_ylabel('L — сторона куба')
    axes[0].set_title('Рост куба: разные законы инфляции')
    axes[1].plot([0, T], [0, T], 'k--', lw=1)
    axes[1].set_xlabel('такт t')
    axes[1].set_ylabel('радиус фронта структуры')
    axes[1].set_title('Фронт структуры против светового конуса')
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'f5_inflation.png', dpi=150)
    plt.close(fig)
    return out


# ==========================================================================
#  6. Сводные графики и отчёт
# ==========================================================================




def closed_experiment() -> dict:
    """Контрольный «закрытый» куб: фронт доходит до стенки и замыкается."""
    print('== 4b. Закрытый куб: эффект замыкания ==', flush=True)
    L, T = 97, 160
    law = InflationLaw('exp', L, v0=4.0, L0=7)
    sim = Simulation(Helix3D(), law, seed_kind='random5', density=0.5)
    sim.run(T)
    g = Helix3D().bits[sim.state()]
    img = render_voxels(g, size=640, canvas_L=sim.L,
                        front_radius=min(T, sim.L / 2), front_time=T,
                        legend_label='бит',
                        label=f'closed helix  L={sim.L}  t={T}')
    Image.fromarray(img).save(OUT / 'd_closed_helix.png')
    data = {'L': sim.L, 'T': T, 'occupancy': float((sim.state() > 0).mean()),
            'bits': sim.info_bits(), 'image': 'd_closed_helix.png'}
    REPORT['closed'] = data
    print(f'  closed helix L={sim.L} bits={data["bits"]} occ={data["occupancy"]:.4f}',
          flush=True)
    return data


def summary_figures() -> None:
    print('== 6. Сводные графики ==', flush=True)
    # временные ряды
    deep = REPORT['deep']
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for name, d in deep.items():
        ts = [m['t'] for m in d['metrics']]
        axes[0, 0].plot(ts, [m['occupancy'] for m in d['metrics']], marker='o', ms=3, label=name)
        axes[0, 1].plot(ts, [m['richness'] for m in d['metrics']], marker='o', ms=3, label=name)
        axes[1, 0].plot(ts, [m['n_clusters'] for m in d['metrics']], marker='o', ms=3, label=name)
        axes[1, 1].plot(ts, [m['bits_total'] for m in d['metrics']], marker='o', ms=3, label=name)
    axes[0, 0].set_yscale('log'); axes[0, 0].set_ylabel('занятость')
    axes[0, 1].set_ylabel('богатство')
    axes[1, 0].set_yscale('log'); axes[1, 0].set_ylabel('число кластеров')
    axes[1, 1].set_yscale('log'); axes[1, 1].set_ylabel('суммарные биты')
    for ax in axes.ravel():
        ax.set_xlabel('такт t'); ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle('Глубокие прогоны: занятость, богатство, кластеры, информация')
    fig.tight_layout()
    fig.savefig(OUT / 'f4_series.png', dpi=150)
    plt.close(fig)

    # световой конус
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, d in deep.items():
        ax.plot(range(len(d['front_series'])), d['front_series'], lw=1.2, label=name)
    ax.plot([0, max(len(d['front_series']) for d in deep.values())],
            [0, max(len(d['front_series']) for d in deep.values())], 'k--', label='c·t')
    ax.set_xlabel('такт t'); ax.set_ylabel('радиус фронта')
    ax.set_title('Причинность: фронт структуры не обгоняет c·t')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'f9_lightcone.png', dpi=150)
    plt.close(fig)

    # галерея
    names = [n for n in deep if (OUT / f'd_{n}_final.png').exists()][:6]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, n in zip(axes.ravel(), names):
        ax.imshow(Image.open(OUT / f'd_{n}_final.png'))
        ax.set_title(f'{n}: {deep[n]["description"]}', fontsize=9)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(OUT / 'f6_gallery.png', dpi=140)
    plt.close(fig)

    # подсчёт правил
    c = REPORT['counts']
    labels = ['2-сост., 6 соседей', 'B/S (2^54)', '2-сост., 27 ячеек',
              'симметр. 27 ячеек', 'B=8, 6 соседей', 'B=8, 27 ячеек']
    vals = [c['six_neighbour_binary']['log10'], c['totalistic_binary_bs']['log10'],
            c['general_binary']['log10'], c['symmetry']['invariant_rules_48_log10'],
            c['b8_6']['log10'], c['b8_27']['log10']]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.barh(labels, vals, color=['#3182ce', '#38a169', '#e53e3e', '#dd6b20', '#805ad5', '#d53f8c'])
    ax.set_xscale('log')
    ax.set_xlabel('log10 числа правил (лог. шкала)')
    ax.set_title('Сколько бывает правил: уровни описания')
    for i, v in enumerate(vals):
        ax.text(v * 1.25, i, f'10^{v:.3g}', va='center', fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'f1_counts.png', dpi=150)
    plt.close(fig)


def finalize() -> None:
    REPORT['checks'] = CHECKS
    REPORT['meta'] = {
        'issue': 'https://github.com/ogamespec/atom/issues/1',
        'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
        'python': sys.version.split()[0],
        'numpy': np.__version__,
    }
    (OUT / 'report.json').write_text(json.dumps(REPORT, ensure_ascii=False, indent=1),
                                     encoding='utf-8')
    n_ok = sum(1 for c in CHECKS if c['ok'])
    print(f'== report.json: {n_ok}/{len(CHECKS)} проверок PASS ==', flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--fast', action='store_true', help='меньше размеры и без свипа')
    args = ap.parse_args()
    t0 = time.time()
    rule_space_counts()
    check('2^128 точно посчитано', 2 ** 128 == 340282366920938463463374607431768211456, str(2 ** 7))
    check('Бёрнсайд: 24 вращения', REPORT['counts']['symmetry']['proper_order'] == 24, 24)
    if not args.fast:
        seed_study()
        parts = REPORT['seed']['by_rule']
        check('классификация зародыша: сумма = 729',
              sum(parts.values()) == 729, parts)
        sprout_sweep()
    # плодородность одной клетки для собственных правил
    law_pt = InflationLaw('step', 65, L0=65)
    sim_pt = Simulation(Sprout3D(), law_pt, seed_kind='point', L0=65)
    sim_pt.run(60)
    check('SPROUT растёт из одной B-битной клетки', sim_pt.info_bits() > 100,
          sim_pt.info_bits())
    sim_hp = Simulation(Helix3D(), law_pt, seed_kind='point', L0=65)
    sim_hp.run(60)
    check('HELIX растёт из одной B-битной клетки', sim_hp.info_bits() > 100,
          sim_hp.info_bits())
    run_deep(fast=args.fast)
    closed_experiment()
    inflation_study()
    summary_figures()
    # проверки по результатам
    deep = REPORT['deep']
    check('TOPPLE сохраняет биты', deep['topple']['bits_series'][-1] == deep['topple']['bits_series'][0],
          deep['topple']['bits_series'][-1])
    # обратимость TWIN на фиксированном кубе
    base = Sprout3D()
    twin = SecondOrder(base)
    law_fix = InflationLaw('step', 17, L0=17)
    st = Simulation(twin, law_fix, seed_kind='block3', L0=17)
    hist = [st.state()]
    for _ in range(8):
        st.step()
        hist.append(st.state())
    ok_rev = True
    for tt in range(7, 0, -1):
        f = base.step(hist[tt], 'wrap')
        ok_rev &= np.array_equal(hist[tt + 1] ^ f, hist[tt - 1])
    check('TWIN обратим (8 шагов, обратный ход)', ok_rev, 8)
    ok_cone = all(max(d['front_series']) <= d['c_max'] * d['T'] + 2.5 for d in deep.values())
    check('фронт не обгоняет c_max·t', ok_cone,
          {n: round(max(d['front_series']) / d['T'], 2) for n, d in deep.items()})
    check('инфляция: старт быстрее c', REPORT['inflation']['exp']['v0'] > 1.0,
          round(REPORT['inflation']['exp']['v0'], 2))
    law_chk = InflationLaw('exp', 97, v0=6.0, L0=7)
    ok_wall = all(law_chk.R(t) > t for t in range(1, int(law_chk.t_end) + 1))
    check('стенка не догоняет свет всю быструю инфляцию', ok_wall,
          round(law_chk.t_end, 1))
    check('куб вырос до L_max', REPORT['inflation']['exp']['L_final'] == 97,
          REPORT['inflation']['exp']['L_final'])
    fast = ['exp', 'power', 'linear', 'step']
    hashes = {k: REPORT['inflation'][k]['state_hash'] for k in fast}
    check('быстрые законы дают одинаковое состояние', len(set(hashes.values())) == 1, hashes)
    check('медленный закон даёт другое состояние',
          REPORT['inflation']['linear_slow']['state_hash'] not in hashes.values(),
          REPORT['inflation']['linear_slow']['state_hash'])
    finalize()
    print(f'Готово за {time.time()-t0:.1f} с')


if __name__ == '__main__':
    main()
