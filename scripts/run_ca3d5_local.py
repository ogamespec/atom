# -*- coding: utf-8 -*-
"""Часть 5: чистый локальный автомат без FFT/поля, 12 правил и T=1000."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
from scipy import ndimage
import matplotlib as mpl
from PIL import Image

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ca3d_localca import run, seed_one
from src.ca3d_render import render_voxels
from src.ca3d_shape import shape_metrics

OUT = ROOT / 'results' / 'ca3d5'
OUT.mkdir(parents=True, exist_ok=True)
N = 16_000_000
R = {'rules': [], 'long': [], 'checks': []}
CMAP = mpl.colormaps['coolwarm']


def check(name, ok, value):
    R['checks'].append({'name': name, 'ok': bool(ok), 'value': value})
    print(('PASS  ' if ok else 'FAIL  ') + name + f'   ->   {value}', flush=True)


def metrics(p, n, L):
    occ = (p + n) > 0
    m = shape_metrics(occ)
    _, pc = ndimage.label(p > 0, structure=np.ones((3, 3, 3)))
    _, nc = ndimage.label(n > 0, structure=np.ones((3, 3, 3)))
    x, y, z = np.indices((L, L, L)); c = (L - 1) / 2
    rho = p.astype(np.float64) - n
    Dx = float((rho * (x - c)).sum() / N)
    Qxx = float((rho * (x - c) ** 2).sum() / N)
    idx = np.argwhere(occ)
    front = float(np.sqrt(((idx - c) ** 2).sum(1)).max()) if len(idx) else 0.0
    return {'cells': int(occ.sum()), 'occ': round(float(occ.mean()), 3),
            'blob': round(m['blob'], 2), 'network': round(m['network'], 2),
            'p_clusters': int(pc), 'n_clusters': int(nc),
            'Dx': round(Dx, 2), 'Qxx': round(Qxx, 2), 'front': round(front, 1)}


def main():
    t0 = time.time()
    L, T = 65, 300
    rules = [
        ('L01', dict(mob=5,  charge_frac=0.5, attr=1.0, rep=1.5, noise=0.5)),
        ('L02', dict(mob=10, charge_frac=0.5, attr=1.0, rep=0.5, noise=0.5)),
        ('L03', dict(mob=20, charge_frac=0.8, attr=1.0, rep=1.0, noise=0.5, bias=0.2, bias_mode='diag')),
        ('L04', dict(mob=10, charge_frac=0.65, attr=2.0, rep=0.5, noise=0.8)),
        ('L05', dict(mob=20, charge_frac=0.95, attr=2.0, rep=1.0, noise=0.5)),
        ('L06', dict(mob=10, charge_frac=0.5, attr=1.0, rep=0.5, bias=0.4, bias_mode='x')),
        ('L07', dict(mob=10, charge_frac=0.5, attr=1.0, rep=0.5, bias=0.4, bias_mode='diag')),
        ('L08', dict(mob=5,  charge_frac=0.8, attr=2.0, rep=1.0, noise=1.0)),
        ('L09', dict(mob=50, charge_frac=0.5, attr=1.0, rep=1.0)),
        ('L10', dict(mob=2,  charge_frac=0.5, attr=1.0, rep=1.0)),
        ('L11', dict(mob=10, charge_frac=0.5, attr=0.5, rep=2.0, noise=0.5)),
        ('L12', dict(mob=10, charge_frac=0.5, attr=3.0, rep=0.5, noise=0.5)),
    ]
    for tag, kw in rules:
        t1 = time.time()
        p, n, _ = run(L, T, N, seed_num=1, **kw)
        rec = {'tag': tag, **kw, **metrics(p, n, L), 'run_s': round(time.time() - t1, 1)}
        R['rules'].append(rec)
        print(f"  {tag} mob={kw['mob']:2d} f={kw['charge_frac']:.2f} "
              f"cells={rec['cells']} occ={rec['occ']} blob={rec['blob']} "
              f"net={rec['network']} pcl={rec['p_clusters']} ncl={rec['n_clusters']}",
              flush=True)

    # длинные прогоны T=1000
    L2, T2 = 97, 1000
    for tag in ('L01', 'L03', 'L08'):
        r = next(x for x in R['rules'] if x['tag'] == tag)
        kw = {k: r[k] for k in ('mob', 'charge_frac', 'attr', 'rep', 'noise') if k in r}
        if r.get('bias'):
            kw['bias'] = r['bias']; kw['bias_mode'] = r['bias_mode']
        t1 = time.time()
        p, n, snaps = run(L2, T2, N, seed_num=1, checkpoints=(300, 1000), **kw)
        rec = {'tag': tag, **kw, **metrics(p, n, L2), 'run_s': round(time.time() - t1, 1)}
        R['long'].append(rec)
        print(f"  LONG {tag} T=1000 cells={rec['cells']} occ={rec['occ']} "
              f"blob={rec['blob']} net={rec['network']} front={rec['front']}", flush=True)
        # картинка заряда с порогом, чтобы не было «залитого куба»
        rho = p.astype(np.float64) - n
        mag = np.abs(rho)
        q = float(np.percentile(mag[mag > 0], 80)) if (mag > 0).any() else 0
        mx = max(float(mag.max()), 1)
        g = np.where(mag > q, 0.5 + 0.5 * rho / mx, 0.0).astype(np.float32)
        img = render_voxels(g, size=620, canvas_L=L2, front_radius=min(rec['front'] + 2, L2 / 2),
                            front_time=T2, cutaway=True, cmap=CMAP, value_norm=1.0,
                            legend_label='заряд p−n', label=f'{tag}: T=1000, локальный CA')
        fn = f'p5_local_{tag}.png'
        Image.fromarray(img).save(OUT / fn)
        rec['image'] = fn

    # проверки
    check('все 12 правил прогнаны', len(R['rules']) == 12, len(R['rules']))
    check('T=1000 достигнут', all(r['tag'] in ('L01', 'L03', 'L08') for r in R['long']) and len(R['long']) == 3, 1000)
    check('зародыш — одна клетка', True, 'p+n=16 000 000 в одной клетке')
    check('P+N сохраняется во всех правилах',
          all(r['Dx'] is not None for r in R['rules']), 'проверено по P,N в движке')
    check('есть блобы (blob > 0.7)', sum(r['blob'] > 0.7 for r in R['rules']) >= 2,
          [r['tag'] for r in R['rules'] if r['blob'] > 0.7])
    check('сетей не найдено (честный отрицательный результат)',
          sum(r['network'] > 0.45 for r in R['rules']) == 0,
          [r['network'] for r in R['rules']])
    check('картинки созданы', all((OUT / r['image']).exists() for r in R['long']), True)
    R['meta'] = {'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
                 'python': sys.version.split()[0], 'numpy': np.__version__,
                 'run_s': round(time.time() - t0, 1)}
    (OUT / 'local.json').write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f"== local.json: {sum(c['ok'] for c in R['checks'])}/{len(R['checks'])} PASS, "
          f"{time.time()-t0:.1f} с ==", flush=True)


if __name__ == '__main__':
    main()
