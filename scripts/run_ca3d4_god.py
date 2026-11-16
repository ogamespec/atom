# -*- coding: utf-8 -*-
"""Стохастические правила части 4: «Бог играет в кости»."""
from __future__ import annotations
import json, sys, time, zlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ca3d_cloud import density, run_god, total
from src.ca3d_render import render_voxels
from src.ca3d_shape import shape_metrics

OUT = ROOT / 'results' / 'ca3d4'
L, T, N = 161, 64, 16_000_000
G = {'runs': [], 'checks': []}
MODES = [('G01', 'split', '3 случайных направления, по 2/7'),
         ('G02', 'rot', 'случайный поворот радиальных весов'),
         ('G03', 'swap', 'радиальный поток + случайный обмен'),
         ('G04', 'weights', 'случайные целые веса 0..2')]


def check(name, ok, value):
    G['checks'].append({'name': name, 'ok': bool(ok), 'value': value})
    print(('PASS  ' if ok else 'FAIL  ') + name + f'   ->   {value}', flush=True)


def front(a):
    idx = np.argwhere(a > 0)
    if len(idx) == 0:
        return 0.0
    c = (L - 1) / 2
    return float(np.sqrt(((idx - c) ** 2).sum(1)).max())


def main():
    t0 = time.time()
    for sid, mode, desc in MODES:
        for seed in (1, 2, 3):
            t1 = time.time()
            a, hist = run_god(L, T, N, mode, seed)
            d = density(a)
            m = shape_metrics(a > 0)
            rec = {'sid': f'{sid}-s{seed}', 'mode': mode, 'desc': desc, 'seed': seed,
                   'bits': total(a), 'cells': int(np.count_nonzero(a)),
                   'front': front(a), 'max_log': float(d.max()),
                   'compression': float(len(zlib.compress(np.round(d * 10).astype(np.uint8).tobytes(), 6)) / d.size),
                   'network': m['network'], 'blob': m['blob'],
                   'run_s': round(time.time() - t1, 1), 'image': None}
            G['runs'].append(rec)
            print(f"  {rec['sid']} {mode:8s} seed={seed} bits={rec['bits']} "
                  f"cells={rec['cells']} front={rec['front']:.1f} "
                  f"maxlog={rec['max_log']:.2f} {rec['run_s']}s", flush=True)
    # картинки по одному сиду на режим
    for sid, mode, desc in MODES:
        r = next(x for x in G['runs'] if x['sid'] == f'{sid}-s1')
        a, _ = run_god(L, T, N, mode, 1)
        d = density(a) ** 0.5
        img = render_voxels(d, size=640, canvas_L=L, front_radius=min(T, L / 2),
                            front_time=T, inset=True, cmap='inferno', value_norm=3.0,
                            legend_label='log10 бит',
                            label=f"{sid}: {mode} (seed 1)")
        fn = f'p4g_{mode}.png'
        Image.fromarray(img).save(OUT / fn)
        r['image'] = fn
    # «кости»: один режим, три сида
    pics = []
    for seed in (1, 2, 3):
        a, _ = run_god(L, T, N, 'split', seed)
        d = density(a) ** 0.5
        img = render_voxels(d, size=420, canvas_L=L, front_radius=min(T, L / 2),
                            front_time=T, cmap='inferno', value_norm=3.0,
                            legend_label='log10 бит', label=f'split seed={seed}')
        pics.append(Image.fromarray(img))
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6))
    for ax, im, seed in zip(axes, pics, (1, 2, 3)):
        ax.imshow(im); ax.set_title(f'seed = {seed}', fontsize=10); ax.axis('off')
    fig.suptitle('«Бог играет в кости»: одно правило, три разных мира', fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / 'p4g_dice.png', dpi=140)
    plt.close(fig)

    check('12 стохастических прогонов', len(G['runs']) == 12, len(G['runs']))
    check('сумма бит сохраняется во всех',
          all(r['bits'] == N for r in G['runs']),
          sorted({r['bits'] for r in G['runs']}))
    check('фронт не обгоняет c·t', all(r['front'] <= T + 1 for r in G['runs']),
          round(max(r['front'] for r in G['runs']), 1))
    check('разные сиды дают разные картины',
          len({(r['cells'], round(r['front'], 1)) for r in G['runs'] if r['mode'] == 'split'}) >= 2,
          [(r['cells'], round(r['front'], 1)) for r in G['runs'] if r['mode'] == 'split'])
    check('картинки созданы',
          all((OUT / r['image']).exists() for r in G['runs'] if r['image']), 4)
    G['meta'] = {'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
                 'python': sys.version.split()[0], 'numpy': np.__version__,
                 'run_s': round(time.time() - t0, 1)}
    (OUT / 'god.json').write_text(json.dumps(G, ensure_ascii=False, indent=1),
                                  encoding='utf-8')
    print(f"== god.json: {sum(c['ok'] for c in G['checks'])}/{len(G['checks'])} PASS, "
          f"{time.time()-t0:.1f} с ==", flush=True)


if __name__ == '__main__':
    main()
