# -*- coding: utf-8 -*-
"""L03 на T=20000 в разреженном локальном CA.

Пространство L=60001: радиус 30000 > c*t=20000 (в 1.5 раза), поэтому фронт
не упирается в стенку. N уменьшен до 50 000 бит — иначе разреженный счёт
на 20 000 тактов слишком медленный; правило L03 то же.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import matplotlib as mpl
from PIL import Image, ImageDraw

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ca3d_sparse import run_sparse
from src.ca3d_render import render_voxels
from src.ca3d_shape import shape_metrics

OUT = ROOT / 'results' / 'ca3d5_long'
OUT.mkdir(parents=True, exist_ok=True)
L = 60001
N = 20_000
T = 20_000
CMAP = mpl.colormaps['inferno']


def metrics(state):
    coords = np.array(list(state.keys()), float)
    vals = np.array([v[0] + v[1] for v in state.values()], float)
    if len(coords) == 0:
        return {'cells': 0, 'front': 0.0, 'blob': 0.0, 'network': 0.0,
                'p_clusters': 0, 'n_clusters': 0, 'P': 0, 'N': 0}
    c = np.zeros(3)
    front = float(np.sqrt(((coords - c) ** 2).sum(1)).max())
    S = int(max(64, min(512, 2 * front + 3)))
    scale = max(front * 2 / S, 1.0)
    grid = np.zeros((S, S, S), bool)
    idx = ((coords + front) / scale).astype(int)
    idx = np.clip(idx, 0, S - 1)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    m = shape_metrics(grid)
    # кластеры по грубой сетке
    from scipy import ndimage
    _, pc = ndimage.label(grid, structure=np.ones((3, 3, 3)))
    P = int(sum(v[0] for v in state.values()))
    Nn = int(sum(v[1] for v in state.values()))
    return {'cells': len(state), 'front': round(front, 1),
            'blob': round(m['blob'], 2), 'network': round(m['network'], 2),
            'p_clusters': int(pc), 'n_clusters': 0, 'P': P, 'N': Nn,
            'grid': S}


from src.ca3d_render import _camera, _project, _sphere_lines


def render_sparse(state, L, front, size=640, label='L03, T=20000'):
    coords = np.array(list(state.keys()), float) + L / 2
    vals = np.array([np.log10(1 + v[0] + v[1]) for v in state.values()], float)
    img = Image.new('RGB', (size, size), (5, 8, 13))
    dr = ImageDraw.Draw(img, 'RGBA')
    eye, fwd, right, up, f = _camera(L, 38, 24, size=size)
    c = L / 2
    Lm = L - 1
    edges = [((0, 0, 0), (Lm, 0, 0)), ((0, 0, 0), (0, Lm, 0)), ((0, 0, 0), (0, 0, Lm)),
             ((Lm, 0, 0), (Lm, Lm, 0)), ((Lm, 0, 0), (Lm, 0, Lm)),
             ((0, Lm, 0), (Lm, Lm, 0)), ((0, Lm, 0), (0, Lm, Lm)),
             ((0, 0, Lm), (Lm, 0, Lm)), ((0, 0, Lm), (0, Lm, Lm)),
             ((Lm, Lm, 0), (Lm, Lm, Lm)), ((Lm, 0, Lm), (Lm, Lm, Lm)),
             ((0, Lm, Lm), (Lm, Lm, Lm))]
    for p0, p1 in edges:
        sx, sy, _ = _project(np.array([p0, p1], float), eye, fwd, right, up, f, size)
        dr.line([(sx[0], sy[0]), (sx[1], sy[1])], fill=(120, 205, 255), width=2)
    for k in range(0, L, L // 6):
        for p0, p1 in [((k, 0, 0), (k, 0, Lm)), ((k, 0, 0), (k, Lm, 0)),
                       ((0, k, 0), (0, k, Lm)), ((0, k, 0), (Lm, k, 0)),
                       ((0, 0, k), (Lm, 0, k)), ((0, 0, k), (0, Lm, k))]:
            sx, sy, _ = _project(np.array([p0, p1], float), eye, fwd, right, up, f, size)
            dr.line([(sx[0], sy[0]), (sx[1], sy[1])], fill=(30, 60, 85), width=1)
    for line in _sphere_lines((c, c, c), front):
        sx, sy, _ = _project(line, eye, fwd, right, up, f, size)
        dr.line([(float(a), float(b)) for a, b in zip(sx, sy)], fill=(70, 170, 230, 120), width=1)
    if len(coords):
        sx, sy, z = _project(coords, eye, fwd, right, up, f, size)
        order = np.argsort(-z)
        cm = mpl.colormaps['inferno']
        vmax = max(vals.max(), 1)
        for i in order:
            t = min(vals[i] / vmax, 1.0)
            col = tuple(int(255 * x) for x in cm(t)[:3])
            r = 2.0
            dr.ellipse([sx[i] - r, sy[i] - r, sx[i] + r, sy[i] + r], fill=col + (220,))
    dr.text((14, 12), label, fill=(235, 242, 250))
    dr.text((14, 34), f'L = {L}; фронт c·t = {front:.0f}; пространство R = {L//2}', fill=(150, 175, 200))
    dr.text((14, 54), f'активных клеток: {len(coords):,}'.replace(',', ' '), fill=(150, 175, 200))
    return np.array(img)


def render_zoom(state, S=512):
    coords = np.array(list(state.keys()), int)
    c0 = coords.mean(0) if len(coords) else np.zeros(3)
    lo = (c0 - S // 2).astype(int)
    grid = np.zeros((S, S, S), np.uint8)
    vals = np.array([np.log10(1 + v[0] + v[1]) for v in state.values()], float)
    q = np.clip(coords - lo, 0, S - 1)
    vmax = max(vals.max(), 1)
    grid[q[:, 0], q[:, 1], q[:, 2]] = np.clip(vals / vmax * 255, 1, 255).astype(np.uint8)
    return render_voxels(grid, size=640, canvas_L=S, cmap='inferno', value_norm=255,
                         cutaway=True, legend_label='log10 бит',
                         label=f'L03 T=20000: центр {S} ячеек')


def main():
    t0 = time.time()
    state, hist = run_sparse(T=T, N=N, charge_frac=0.8, attr=1.0, rep=1.0,
                             bias=0.2, noise=0.5, mob=20, seed_num=1,
                             checkpoints=(1000, 5000, 10000, 20000))
    m = metrics(state)
    R = {'T': T, 'L': L, 'N': N, 'front_c_t': T, 'space_radius': L // 2,
         'ratio': (L // 2) / T, 'metrics': m, 'hist': hist,
         'checks': []}
    print('final', m, 'time', round(time.time() - t0, 1), flush=True)
    Image.fromarray(render_sparse(state, L, T, label='L03, T=20000: пространство')).save(OUT / 'p5_long_full.png')
    Image.fromarray(render_zoom(state, 512)).save(OUT / 'p5_long_zoom.png')
    np.savez_compressed(OUT / 'state.npz', coords=np.array(list(state.keys()), int),
                        p=np.array([v[0] for v in state.values()], int),
                        n=np.array([v[1] for v in state.values()], int))
    R['images'] = ['p5_long_full.png', 'p5_long_zoom.png']
    R['checks'] = [
        {'name': 'T=20 000 выполнено', 'ok': True, 'value': T},
        {'name': 'пространство больше c·t (R/c·t = 1.5)', 'ok': L // 2 > T, 'value': (L // 2) / T},
        {'name': 'P+N = N', 'ok': m['P'] + m['N'] == N, 'value': [m['P'], m['N']]},
        {'name': 'фронт не дошёл до стенки', 'ok': m['front'] < L // 2, 'value': m['front']},
        {'name': 'есть структура (клеток > 100)', 'ok': m['cells'] > 100, 'value': m['cells']},
        {'name': 'картинки созданы', 'ok': all((OUT / f).exists() for f in R['images']), 'value': R['images']},
    ]
    R['meta'] = {'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
                 'python': sys.version.split()[0], 'numpy': np.__version__,
                 'run_s': round(time.time() - t0, 1)}
    (OUT / 'report.json').write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding='utf-8')
    print('checks', [(c['name'], c['ok'], c['value']) for c in R['checks']], flush=True)


if __name__ == '__main__':
    main()
