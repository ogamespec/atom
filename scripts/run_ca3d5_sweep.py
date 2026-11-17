# -*- coding: utf-8 -*-
"""Часть 5: перебор 30 правил заряженного газа.

Одна клетка-зародыш, начальная асимметрия заряда f = P/(P+N), bias-режимы,
rep, β, шум, K, экранирование, анти-кулон, обмен. Метрики: сеть, блоб,
слои/заполнение; отбор лучших и глубокая визуализация.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
from scipy import ndimage

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ca3d_coulomb import run_one
from src.ca3d_shape import shape_metrics

OUT = ROOT / 'results' / 'ca3d5'
OUT.mkdir(parents=True, exist_ok=True)
L, T, N = 65, 20, 16_000_000
S = {'rules': []}


def cluster_stats(mask):
    if not mask.any():
        return 0, 0.0
    lab, n = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(lab.ravel())[1:]
    big = int((sizes > 30).sum())
    frac = float(sizes.max() / max(mask.sum(), 1)) if n else 0.0
    return big, frac


def moments(p, n):
    x, y, z = np.indices((L, L, L))
    c = (L - 1) / 2
    rho = p.astype(np.float64) - n.astype(np.float64)
    D = [float((rho * (x - c)).sum() / N), float((rho * (y - c)).sum() / N),
         float((rho * (z - c)).sum() / N)]
    Q = float((rho * (x - c) ** 2).sum() / N)
    return D, Q


def walls_of(p, n):
    sign = np.sign(p - n)
    w = np.zeros((L, L, L), bool)
    for ax in range(3):
        for sh in (1, -1):
            w |= sign != np.roll(sign, sh, axis=ax)
    w &= (p + n) > 0
    return w


def evaluate(tag, kw):
    t0 = time.time()
    p, n, phi, hist = run_one(L, T, N, **kw)
    occ = (p + n) > 0
    pm, nm = p > 0, n > 0
    m = shape_metrics(occ)
    mp = shape_metrics(pm)
    mn = shape_metrics(nm)
    walls = walls_of(p, n)
    mw = shape_metrics(walls)
    bp, bf = cluster_stats(pm)
    bn, nf = cluster_stats(nm)
    D, Q = moments(p, n)
    net = min(1.0, m['network'] + 0.5 * mw['network'])
    blob = max(mp['blob'], mn['blob'])
    occf = float(occ.mean())
    if occf > 0.40:
        cls = 'слой/заполнение'
    elif net > 0.45 and occf < 0.35:
        cls = 'сеть'
    elif blob > 0.70 and max(bf, nf) > 0.55:
        cls = 'блоб'
    else:
        cls = 'облако'
    rec = {'tag': tag, **{k: v for k, v in kw.items()},
           'P': int(p.sum()), 'Nbits': int(n.sum()),
           'total': int(p.sum() + n.sum()), 'cells': int(occ.sum()),
           'occ': round(occf, 3),
           'p_clusters': bp, 'n_clusters': bn,
           'largest_p': round(bf, 2), 'largest_n': round(nf, 2),
           'network': round(m['network'], 2), 'wall_network': round(mw['network'], 2),
           'net_score': round(net, 2), 'blob': round(blob, 2),
           'walls': round(float(walls.mean()), 3),
           'D': round(float(np.linalg.norm(D)), 2), 'Dx': round(D[0], 2),
           'Qxx': round(Q, 2), 'class': cls,
           'run_s': round(time.time() - t0, 2)}
    S['rules'].append(rec)
    print(f"  {tag:5s} {cls:12s} net={rec['net_score']:.2f} blob={rec['blob']:.2f} "
          f"occ={rec['occ']:.2f} pcl={bp} ncl={bn} Dx={rec['Dx']} Qxx={rec['Qxx']} "
          f"f={kw.get('charge_frac',0.5)} {kw.get('bias_mode','x')}", flush=True)
    return rec


def main():
    t0 = time.time()
    # контроли + случайный перебор
    rules = [
        ('S00', dict(charge_frac=0.5, bias=0.0, rep=0.0, beta=0.5, bias_mode='none')),
        ('S01', dict(charge_frac=0.5, bias=0.4, rep=1.0, beta=0.5, bias_mode='x')),
        ('S02', dict(charge_frac=0.5, bias=0.2, rep=1.0, beta=0.5, bias_mode='x', noise=0.8, seed_num=2)),
        ('S03', dict(charge_frac=0.8, bias=0.0, rep=0.5, beta=0.5, bias_mode='none')),
    ]
    rng = np.random.default_rng(7)
    modes = ['none', 'x', 'diag', 'radial', 'random']
    for i in range(26):
        mode = modes[int(rng.integers(len(modes)))]
        bias = 0.0 if mode == 'none' else float(rng.choice([0.2, 0.5]))
        kw = dict(charge_frac=float(rng.choice([0.5, 0.65, 0.8, 0.95])),
                  bias=bias, rep=float(rng.choice([0.0, 0.5, 1.5])),
                  beta=float(rng.choice([0.2, 0.5, 0.9])), bias_mode=mode,
                  noise=float(rng.choice([0.0, 0.4, 0.8])),
                  K=(None if rng.random() < 0.6 else 30000),
                  screening=float(rng.choice([0.0, 0.15])),
                  anti=bool(rng.random() < 0.25),
                  exchange=bool(rng.random() < 0.35),
                  seed_num=int(rng.integers(1, 5)))
        rules.append((f'S{i+4:02d}', kw))
    for tag, kw in rules:
        evaluate(tag, kw)
    S['meta'] = {'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
                 'python': sys.version.split()[0], 'numpy': np.__version__,
                 'L': L, 'T': T, 'N': N, 'run_s': round(time.time() - t0, 1)}
    (OUT / 'sweep.json').write_text(json.dumps(S, ensure_ascii=False, indent=1), encoding='utf-8')
    nets = [r for r in S['rules'] if r['class'] == 'сеть']
    blobs = [r for r in S['rules'] if r['class'] == 'блоб']
    print(f"== {len(S['rules'])} правил; сетей {len(nets)}, блобов {len(blobs)}, "
          f"слоёв {sum(r['class']=='слой/заполнение' for r in S['rules'])}; "
          f"{time.time()-t0:.1f} с ==", flush=True)


if __name__ == '__main__':
    main()
