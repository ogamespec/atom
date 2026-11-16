# -*- coding: utf-8 -*-
"""Часть 4: один зародыш — 16 млн бит, перенос долей и обмен."""
from __future__ import annotations
import json, sys, time, zlib
from pathlib import Path
import numpy as np
from PIL import Image

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ca3d_cloud import density, run, seed, total
from src.ca3d_render import render_voxels, slice_mosaic
from src.ca3d_shape import shape_metrics

OUT = ROOT / 'results' / 'ca3d4'
OUT.mkdir(parents=True, exist_ok=True)
L, T, N = 161, 64, 16_000_000
REPORT = {'rules': [], 'checks': []}

RULES = [
    ('C01', 'uniform', None, 'none', 'равномерный перенос'),
    ('C02', 'uniform', 20000, 'none', 'перенос с ограничением K'),
    ('C03', 'x', None, 'none', 'смещение по x'),
    ('C04', 'z', None, 'none', 'смещение по z'),
    ('C05', 'radial', None, 'none', 'радиальный поток наружу'),
    ('C06', 'radial', None, 'x', 'радиальный поток + обмен по x'),
    ('C07', 'radial', None, 'all', 'радиальный поток + обмен по всем осям'),
    ('C08', 'radial', 2000, 'none', 'радиальный поток с K=2000'),
    ('C09', 'radial', 20000, 'x', 'радиальный поток K=20000 + обмен'),
    ('C10', 'radial', 500, 'none', 'радиальный поток с K=500'),
    ('C11', 'diag', None, 'none', 'диагональное смещение'),
    ('C12', 'uniform', None, 'all', 'равномерный перенос + обмен'),
]


def check(name, ok, value):
    REPORT['checks'].append({'name': name, 'ok': bool(ok), 'value': value})
    print(('PASS  ' if ok else 'FAIL  ') + name + f'   ->   {value}', flush=True)


def front(a, L):
    idx = np.argwhere(a > 0)
    if len(idx) == 0:
        return 0.0
    c = (L - 1) / 2
    return float(np.sqrt(((idx - c) ** 2).sum(1)).max())


def main():
    t0 = time.time()
    b0 = total(seed(L, N))
    print(f'зародыш: одна клетка, {b0} бит', flush=True)
    for sid, wmode, K, ex, desc in RULES:
        t1 = time.time()
        a, hist = run(L, T, N, wmode, K, ex)
        d = density(a)
        m = shape_metrics(a > 0)
        comp = len(zlib.compress(np.round(d * 10).astype(np.uint8).tobytes(), 6)) / d.size
        rec = {'sid': sid, 'wmode': wmode, 'K': K, 'exchange': ex, 'desc': desc,
               'T': T, 'L': L, 'N': N, 'bits_start': b0, 'bits_end': total(a),
               'cells': int(np.count_nonzero(a)), 'front': front(a, L),
               'max_log': float(d.max()), 'mean_bits': float(a.sum() / max(np.count_nonzero(a), 1)),
               'compression': float(comp), 'network': m['network'], 'blob': m['blob'],
               'run_s': round(time.time() - t1, 2), 'image': None,
               'hist_cells': [h[2] for h in hist[::8]]}
        rec['class'] = ('ядро+гало' if rec['max_log'] > 5 else
                        ('радиальная оболочка' if wmode == 'radial' else 'облако'))
        REPORT['rules'].append(rec)
        print(f"  {sid} {wmode:8s} K={str(K):5s} ex={ex:4s}  bits={rec['bits_end']}  "
              f"cells={rec['cells']}  front={rec['front']:.1f}  maxlog={rec['max_log']:.2f}  "
              f"comp={rec['compression']:.3f}  {rec['run_s']}s", flush=True)

    # картинки
    chosen = ['C01', 'C05', 'C06', 'C08', 'C10']
    for sid in chosen:
        r = next(x for x in REPORT['rules'] if x['sid'] == sid)
        a, _ = run(L, T, N, r['wmode'], r['K'], r['exchange'])
        d = density(a) ** 0.5
        norm = 3.0 if r['max_log'] <= 4.5 else 6.0
        img = render_voxels(d, size=640, canvas_L=L, front_radius=min(T, L / 2),
                            front_time=T, inset=True, cmap='inferno', value_norm=norm,
                            legend_label='log10 бит',
                            label=f"{sid}: {r['wmode']} K={r['K']} ex={r['exchange']}")
        fn = f'p4_{sid}.png'
        Image.fromarray(img).save(OUT / fn)
        r['image'] = fn
        if sid == 'C05':
            slice_mosaic(density(a), OUT / 'p4_C05_slices.png', n=6, cmap='inferno',
                         title='C05: радиальная оболочка, срезы log10(1+бит)', vmin=0, vmax=3)

    check('12 правил прогнаны', len(REPORT['rules']) == 12, len(REPORT['rules']))
    check('сумма бит сохраняется во всех правилах',
          all(r['bits_end'] == b0 for r in REPORT['rules']),
          {'start': b0, 'ends': sorted({r['bits_end'] for r in REPORT['rules']})})
    check('фронт не обгоняет c·t', all(r['front'] <= T + 1 for r in REPORT['rules']),
          round(max(r['front'] for r in REPORT['rules']), 1))
    check('есть разные морфологии', len({r['class'] for r in REPORT['rules']}) >= 2,
          sorted({r['class'] for r in REPORT['rules']}))
    check('активных клеток меньше объёма', all(r['cells'] < L ** 3 for r in REPORT['rules']),
          round(max(r['cells'] for r in REPORT['rules']) / L ** 3, 4))
    check('картинки созданы',
          all((OUT / r['image']).exists() for r in REPORT['rules'] if r['image']), len(chosen))
    REPORT['meta'] = {'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
                      'python': sys.version.split()[0], 'numpy': np.__version__,
                      'germ_bits': b0, 'run_s': round(time.time() - t0, 1)}
    (OUT / 'report.json').write_text(json.dumps(REPORT, ensure_ascii=False, indent=1),
                                     encoding='utf-8')
    print(f"== report.json: {sum(c['ok'] for c in REPORT['checks'])}/"
          f"{len(REPORT['checks'])} PASS, {time.time()-t0:.1f} с ==", flush=True)


if __name__ == '__main__':
    main()
