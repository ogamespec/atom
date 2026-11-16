# -*- coding: utf-8 -*-
"""Вторая часть: 100 правил для сетей, пены и блобов.

Стадии:
  1. Сгенерировать 100 правил четырёх семейств.
  2. Прогнать скрининг L=65, измерить морфологию.
  3. Отсеять мёртвые/туманные/тривиальные, выбрать лучшие сети, пену и блобы.
  4. Глубокие прогоны и визуализации.
  5. report.json + проверки.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field, asdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ca3d_morph import (Bond3D, Invasion3D, Potts3D, Tumor3D, bonds_to_mask,
                            potts_walls, von6)
from src.ca3d_shape import (bond_graph_metrics, largest_domain, shape_metrics)

OUT = ROOT / 'results' / 'ca3d2'
OUT.mkdir(parents=True, exist_ok=True)
REPORT: dict = {'meta': {}, 'checks': []}
CHECKS: list = []


def check(name, ok, value):
    CHECKS.append({'name': name, 'ok': bool(ok), 'value': value})
    print(('PASS  ' if ok else 'FAIL  ') + name + f'   ->   {value}', flush=True)


@dataclass
class Spec:
    sid: str
    family: str
    title: str
    params: dict
    T: int
    L: int = 65
    seed: int = 1


def build_rules() -> list[Spec]:
    rules = []
    # --- Potts: пена/домены ---
    k = 0
    for Q in (4, 6, 8, 12, 16):
        for noise in (0.02, 0.05, 0.1):
            for T in (40, 80):
                if k >= 25:
                    break
                k += 1
                init = 'block' if k % 2 == 0 else 'full'
                params = dict(Q=Q, T=T, noise=noise, init=init)
                if init == 'block':
                    params['seed_size'] = 41
                rules.append(Spec(f'P{k:03d}', 'potts',
                                  f'Potts Q={Q}, шум={noise}, T={T}, {init}',
                                  params, T))
    # --- Invasion: ветвящиеся сети ---
    k = 0
    for N in (5000, 9000, 14000, 20000):
        for alpha in (0.5, 0.8, 1.0, 1.3):
            for bias in (0.0, 0.3):
                if k >= 25:
                    break
                k += 1
                rules.append(Spec(f'I{k:03d}', 'invasion', f'Invasion N={N}, α={alpha}, bias={bias}',
                                  dict(N=N, alpha=alpha, bias=bias), 0))
    # --- Bond: графовые сети ---
    k = 0
    for T in (30, 50, 80):
        for p in (0.2, 0.35, 0.5):
            for loops in (True, False):
                if k >= 20:
                    break
                k += 1
                rules.append(Spec(f'B{k:03d}', 'bond', f'Bond T={T}, p={p}, loops={int(loops)}',
                                  dict(T=T, p=p, loops=loops), T))
    for p in (0.1, 0.65, 0.8, 0.95, 0.15, 0.45, 0.7):
        k += 1
        rules.append(Spec(f'B{k:03d}', 'bond', f'Bond T=50, p={p}, loops=1',
                          dict(T=50, p=p, loops=True), 50))
    # --- Growth: Eden-блобы (15) ---
    k = 0
    for T in (12, 20, 30):
        for dens in (0.3, 0.5, 0.7):
            for kk in (1, 2):
                if k >= 15:
                    break
                k += 1
                rules.append(Spec(f'E{k:03d}', 'eden', f'Eden T={T}, ρ={dens}, k={kk}',
                                  dict(T=T, density=dens, k=kk), T))
    # --- Growth: опухоль (10) ---
    tumors = [
        dict(T=25, D=0.2, cons=0.4, div=0.3, death=0.05, pmax=0.6),
        dict(T=40, D=0.2, cons=0.4, div=0.3, death=0.05, pmax=0.6),
        dict(T=40, D=0.35, cons=0.2, div=0.2, death=0.02, pmax=0.9),
        dict(T=40, D=0.1, cons=0.5, div=0.35, death=0.08, pmax=0.5),
        dict(T=60, D=0.25, cons=0.3, div=0.25, death=0.04, pmax=0.7),
        dict(T=30, D=0.15, cons=0.6, div=0.4, death=0.1, pmax=0.5),
        dict(T=50, D=0.4, cons=0.15, div=0.15, death=0.01, pmax=0.95),
        dict(T=50, D=0.05, cons=0.4, div=0.3, death=0.06, pmax=0.5),
        dict(T=35, D=0.3, cons=0.35, div=0.28, death=0.05, pmax=0.65),
        dict(T=70, D=0.2, cons=0.45, div=0.32, death=0.07, pmax=0.55),
    ]
    for j, p in enumerate(tumors, 1):
        rules.append(Spec(f'T{j:03d}', 'tumor', f'Tumor T={p["T"]}, D={p["D"]}, c={p["cons"]}',
                          p, p['T']))
    return rules[:100]


def _eden(p, L, seed):
    rng = np.random.default_rng(seed)
    s = np.zeros((L, L, L), np.uint8)
    c = L // 2
    h = 1
    m = rng.random((2 * h + 1,) * 3) < p['density']
    s[c - h:c + h + 1, c - h:c + h + 1, c - h:c + h + 1] = m
    for t in range(p['T']):
        n = von6(s)
        s = ((s == 1) | ((s == 0) & (n >= p['k']))).astype(np.uint8)
    return s


def _run(spec: Spec):
    t0 = time.time()
    L = spec.L
    out = {'sid': spec.sid, 'family': spec.family, 'title': spec.title,
           'params': spec.params, 'T': spec.T, 'L': L}
    if spec.family == 'potts':
        p = Potts3D(**spec.params).run(L, spec.seed)
        walls = potts_walls(p)
        dom = largest_domain(p, spec.params['Q'])
        mn = shape_metrics(walls)
        mb = shape_metrics(dom)
        out.update({'occ': float((p > 0).mean()), 'wall_frac': float(walls.mean()),
                    'network': mn['network'], 'blob': mb['blob'],
                    'percolates': mn['percolates'], 'sv': mn['sv'],
                    'skel_len': mn['skel_len'], 'branches': mn['branches'],
                    'largest': mb['largest'], 'compactness': mb['compactness'],
                    'n_clusters': mn['n_clusters'], 'euler': mn['euler'],
                    'foam': bool(0.04 < walls.mean() < 0.4 and mn['branches'] > 150),
                    'kind': 'potts'})
    elif spec.family == 'invasion':
        cl = Invasion3D(**spec.params).run(L, spec.seed)
        m = shape_metrics(cl)
        out.update({'occ': m['occupancy'], 'network': m['network'], 'blob': m['blob'],
                    'percolates': m['percolates'], 'sv': m['sv'], 'skel_len': m['skel_len'],
                    'branches': m['branches'], 'largest': m['largest'],
                    'compactness': m['compactness'], 'n_clusters': m['n_clusters'],
                    'euler': m['euler'], 'foam': False, 'kind': 'invasion'})
    elif spec.family == 'bond':
        b = Bond3D(**spec.params).run(L, spec.seed)
        g = bond_graph_metrics(b)
        m = shape_metrics(bonds_to_mask(b), with_skeleton=False)
        graph_score = (0.5 * min(g['mean_degree'] / 3.2, 1)
                       + 0.5 * min(g['branch_nodes'] / max(g['nodes'], 1) / 0.6, 1))
        graph_score *= min(g['nodes'] / 300.0, 1.0)
        out.update({'occ': m['occupancy'], 'network': max(m['network'], graph_score),
                    'blob': m['blob'], 'percolates': m['percolates'], 'sv': m['sv'],
                    'skel_len': 0, 'branches': g['branch_nodes'], 'largest': g['nodes'],
                    'compactness': 0.0, 'n_clusters': 1, 'euler': 0,
                    'foam': False, 'nodes': g['nodes'], 'edges': g['edges'],
                    'mean_degree': g['mean_degree'], 'kind': 'bond'})
    elif spec.family == 'eden':
        s = _eden(spec.params, L, spec.seed)
        m = shape_metrics(s > 0)
        out.update({'occ': m['occupancy'], 'network': m['network'], 'blob': m['blob'],
                    'percolates': m['percolates'], 'sv': m['sv'], 'skel_len': m['skel_len'],
                    'branches': m['branches'], 'largest': m['largest'],
                    'compactness': m['compactness'], 'n_clusters': m['n_clusters'],
                    'euler': m['euler'], 'foam': False, 'kind': 'eden'})
    else:
        st = Tumor3D(**spec.params).run(L, spec.seed)
        m = shape_metrics(st > 0)
        out.update({'occ': m['occupancy'], 'network': m['network'], 'blob': m['blob'],
                    'percolates': m['percolates'], 'sv': m['sv'], 'skel_len': m['skel_len'],
                    'branches': m['branches'], 'largest': m['largest'],
                    'compactness': m['compactness'], 'n_clusters': m['n_clusters'],
                    'euler': m['euler'], 'foam': False,
                    'necrotic': float((st == 2).mean()), 'kind': 'tumor'})
    out['alive'] = bool(out['occ'] > 1e-4 and out.get('largest', 0) > 100)
    if spec.family == 'bond':
        out['alive'] = bool(out.get('nodes', 0) > 100)
    out['fog'] = bool(out['occ'] > 0.6)
    out['structured'] = bool(out['network'] > 0.55 or out['blob'] > 0.55 or out['foam'])
    out['verdict'] = ('туман' if out['fog'] else
                      ('мёртвое' if not out['alive'] else
                       ('сеть' if out['network'] >= out['blob'] and out['network'] > 0.55 else
                        ('блоб' if out['blob'] > 0.55 else 'слабое'))))
    out['time_s'] = round(time.time() - t0, 2)
    return out


def main():
    t0 = time.time()
    rules = build_rules()
    print(f'== 100 правил: {len(rules)} ==', flush=True)
    with Pool(24) as pool:
        res = pool.map(_run, rules, chunksize=1)
    alive = [r for r in res if r['alive']]
    struct = [r for r in alive if r['structured']]
    print(f"  живых {len(alive)}, структурных {len(struct)}, "
          f"время {time.time()-t0:.1f} с")
    from collections import Counter
    print('  вердикты:', dict(Counter(r['verdict'] for r in res)))
    REPORT['rules'] = res
    REPORT['funnel'] = {'total': len(res), 'alive': len(alive), 'structured': len(struct)}
    REPORT['meta'] = {'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
                      'python': sys.version.split()[0], 'numpy': np.__version__}
    REPORT['checks'] = CHECKS
    check('всего правил = 100', len(res) == 100, len(res))
    check('есть живые структуры', len(struct) >= 10, len(struct))
    check('есть сети и блобы',
          sum(r['verdict'] == 'сеть' for r in res) >= 3 and
          sum(r['verdict'] == 'блоб' for r in res) >= 3,
          {'сети': sum(r['verdict'] == 'сеть' for r in res),
           'блобы': sum(r['verdict'] == 'блоб' for r in res)})
    (OUT / 'screening.json').write_text(json.dumps(REPORT, ensure_ascii=False, indent=1),
                                        encoding='utf-8')
    print(f'== screening.json: {sum(c["ok"] for c in CHECKS)}/{len(CHECKS)} PASS ==')


if __name__ == '__main__':
    main()
