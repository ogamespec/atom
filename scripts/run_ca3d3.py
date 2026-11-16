# -*- coding: utf-8 -*-
"""Третья часть: консервативные битовые правила из зародыша.

12 правил (поворот/перестановка байтов/поле направлений), зародыш 15^3
по 64 бита = 216 000 единичных бит. Проверяется точное сохранение popcount
и причинность; лучшие правила рендерятся в полный куб с фронтом.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ca3d_bits import dense, front_radius, run, seed, total_bits
from src.ca3d_render import render_voxels
from src.ca3d_shape import shape_metrics

OUT = ROOT / 'results' / 'ca3d3'
OUT.mkdir(parents=True, exist_ok=True)
REPORT = {'rules': [], 'checks': []}
L, T, SIZE = 161, 48, 15

RULES = [
    ('R01', 0,  'id',    False, 'прямые лучи: без поворота'),
    ('R02', 1,  'id',    False, 'клубок: поворот на 1 бит'),
    ('R03', 2,  'id',    False, 'поворот на 2'),
    ('R04', 3,  'id',    False, 'поворот на 3'),
    ('R05', 5,  'id',    False, 'поворот на 5'),
    ('R06', 8,  'id',    False, 'поворот на байт'),
    ('R07', 16, 'id',    False, 'поворот на 16'),
    ('R08', 32, 'id',    False, 'поворот на 32'),
    ('R09', 1,  'swap',  False, 'поворот 1 + перестановка байтов'),
    ('R10', 8,  'swap',  False, 'поворот 8 + перестановка байтов'),
    ('R11', 0,  'id',    True,  'поле направлений'),
    ('R12', 8,  'id',    True,  'поворот 8 + поле направлений'),
]


def check(name, ok, value):
    REPORT['checks'].append({'name': name, 'ok': bool(ok), 'value': value})
    print(('PASS  ' if ok else 'FAIL  ') + name + f'   ->   {value}', flush=True)


def main():
    t0 = time.time()
    g0 = seed(L, SIZE)
    b0 = total_bits(g0)
    print(f'зародыш: {len(g0)} клеток, {b0} бит', flush=True)
    for sid, rot, pk, field, desc in RULES:
        t1 = time.time()
        g, hist = run(L, T, SIZE, rot, pk, seed_num=1, field=field, track=True)
        d = dense(g, L)
        m = shape_metrics(d > 0)
        rec = {
            'sid': sid, 'rot': rot, 'perm': pk, 'field': field, 'desc': desc,
            'T': T, 'L': L, 'seed_size': SIZE,
            'bits_start': b0, 'bits_end': total_bits(g),
            'cells': len(g), 'front': front_radius(g, L),
            'max_bits': int(d.max()), 'mean_bits': float(d.sum() / max(len(g), 1)),
            'occupancy': float((d > 0).mean()),
            'clusters': m['n_clusters'], 'network': m['network'], 'blob': m['blob'],
            'largest': m['largest'], 'run_s': round(time.time() - t1, 2),
            'hist_cells': [h[1] for h in hist[::8]],
            'hist_front': [round(h[3], 1) for h in hist[::8]],
            'image': None,
        }
        # классификация по морфологии
        if rec['front'] > 0.75 * T and rec['max_bits'] <= 16:
            rec['class'] = 'пучки/лучи'
        elif rec['front'] < 0.45 * T:
            rec['class'] = 'клубок'
        else:
            rec['class'] = 'облако'
        REPORT['rules'].append(rec)
        print(f"  {sid} rot={rot} {pk} field={int(field)}  bits={rec['bits_end']}  "
              f"cells={rec['cells']}  front={rec['front']:.1f}  {rec['class']}  "
              f"{rec['run_s']}s", flush=True)

    # рендер выбранных правил
    chosen = ['R01', 'R02', 'R04', 'R09', 'R12']
    for sid in chosen:
        r = next(x for x in REPORT['rules'] if x['sid'] == sid)
        g, _ = run(L, T, SIZE, r['rot'], r['perm'], seed_num=1, field=r['field'])
        d = dense(g, L)
        norm = 16 if r['max_bits'] <= 16 else 64
        img = render_voxels(d, size=640, canvas_L=L,
                            front_radius=min(T + 8, L / 2), front_time=T + 7,
                            front_label='фронт c·t+r₀',
                            inset=True, cmap='inferno', value_norm=norm,
                            legend_label='бит/ячейка',
                            label=f"{sid}: rot={r['rot']} {r['perm']}  L={L}  T={T}")
        fn = f'p3_{sid}.png'
        Image.fromarray(img).save(OUT / fn)
        r['image'] = fn

    check('все 12 правил прогнаны', len(REPORT['rules']) == 12, len(REPORT['rules']))
    check('popcount сохраняется во всех правилах',
          all(r['bits_end'] == b0 for r in REPORT['rules']),
          {'start': b0, 'ends': sorted({r['bits_end'] for r in REPORT['rules']})})
    check('фронт не обгоняет c·t + радиус зародыша',
          all(r['front'] <= T + 8 for r in REPORT['rules']),
          round(max(r['front'] for r in REPORT['rules']), 1))
    check('пустые клетки не обрабатываются (активных < объёма)',
          all(r['cells'] < L ** 3 for r in REPORT['rules']),
          round(max(r['cells'] for r in REPORT['rules']) / L ** 3, 4))
    check('есть разные морфологии',
          len({r['class'] for r in REPORT['rules']}) >= 2,
          sorted({r['class'] for r in REPORT['rules']}))
    check('картинки созданы',
          all((OUT / r['image']).exists() for r in REPORT['rules'] if r['image']),
          len(chosen))
    REPORT['meta'] = {'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
                      'python': sys.version.split()[0], 'numpy': np.__version__,
                      'seed_bits': b0, 'run_time_s': round(time.time() - t0, 1)}
    (OUT / 'report.json').write_text(json.dumps(REPORT, ensure_ascii=False, indent=1),
                                     encoding='utf-8')
    print(f"== report.json: {sum(c['ok'] for c in REPORT['checks'])}/"
          f"{len(REPORT['checks'])} PASS, {time.time()-t0:.1f} с ==", flush=True)


if __name__ == '__main__':
    main()
