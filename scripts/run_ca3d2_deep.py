# -*- coding: utf-8 -*-
"""Глубокие прогоны лучших правил второй части и визуализации."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ca3d_morph import (Bond3D, Invasion3D, Potts3D, Tumor3D, bonds_to_mask,
                            potts_walls, von6)
from src.ca3d_net import graph_from_bonds, graph_from_mask, render_graph
from src.ca3d_render import render_voxels, slice_mosaic
from src.ca3d_shape import bond_graph_metrics, shape_metrics

OUT = ROOT / 'results' / 'ca3d2'
OUT.mkdir(parents=True, exist_ok=True)
SCREEN = json.loads((OUT / 'screening.json').read_text(encoding='utf-8'))
DEEP: dict = {'selected': [], 'checks': []}


def check(name, ok, value):
    DEEP['checks'].append({'name': name, 'ok': bool(ok), 'value': value})
    print(('PASS  ' if ok else 'FAIL  ') + name + f'   ->   {value}', flush=True)


def crop(a, margin=3):
    idx = np.argwhere(a > 0)
    if len(idx) == 0:
        return a, a.shape[0]
    lo, hi = idx.min(0), idx.max(0)
    c = (lo + hi) // 2
    side = int(max(hi - lo) + 2 * margin)
    side = max(9, min(side, a.shape[0]))
    lo2 = np.maximum(c - side // 2, 0)
    hi2 = np.minimum(lo2 + side, a.shape[0])
    lo2 = hi2 - side
    return a[lo2[0]:hi2[0], lo2[1]:hi2[1], lo2[2]:hi2[2]], side


def full_render(g, label, cmap, norm, t_eff, legend, L=129):
    return render_voxels(g, size=640, canvas_L=L,
                         front_radius=min(t_eff, L / 2), front_time=t_eff,
                         inset=True, cmap=cmap, value_norm=norm,
                         legend_label=legend, label=label)


def by_sid(sid):
    for r in SCREEN['rules']:
        if r['sid'] == sid:
            return r
    raise KeyError(sid)


def main():
    L = 161
    selected = []
    # 1) INVASION — сеть
    t0 = time.time()
    inv = Invasion3D(N=15000, alpha=0.5).run(L)
    m = shape_metrics(inv > 0)
    Image.fromarray(full_render(inv, f'INVASION  L={L}', 'inferno', 1.0, 50,
                                'ячейки')).save(OUT / 'n_invasion.png')
    n, e = graph_from_mask(inv > 0)
    Image.fromarray(render_graph(n, e, size=640, canvas_L=L, show_nodes=False,
                                 edge_width=2, front_radius=50,
                                 label='INVASION: скелет сети')).save(OUT / 'n_invasion_skel.png')
    slice_mosaic(inv, OUT / 'n_invasion_slices.png', n=6, cmap='inferno',
                 title=f'INVASION: срезы, L={L}', vmin=0, vmax=1)
    art = {**by_sid('I025'), 'metrics': m, 'deep_L': L, 'nodes': len(n), 'edges': len(e),
           'image': 'n_invasion.png', 'skel': 'n_invasion_skel.png',
           'slices': 'n_invasion_slices.png'}
    selected.append(art)
    print('invasion', round(time.time()-t0, 1), m['network'], len(n), len(e))

    # 2) POTTS — пена
    t0 = time.time()
    # Пена по построению заполняет всё пространство; для центрального
    # фрагмента вырезаем куб 41^3 из центра и окружаем пустым пространством.
    p = Potts3D(Q=8, T=40, init='full', noise=0.05).run(L)
    walls_full = potts_walls(p)
    o = (L - 41) // 2
    frag = walls_full[o:o + 41, o:o + 41, o:o + 41]
    walls = np.zeros_like(walls_full)
    walls[o:o + 41, o:o + 41, o:o + 41] = frag
    mw = shape_metrics(frag)
    mw['wall_frac_fragment'] = float(frag.mean())
    from matplotlib.colors import ListedColormap as _LC
    tab = matplotlib.colormaps['tab10']
    fcmap = _LC(['#05070a'] + [tab(i) for i in range(8)])
    gfoam = np.where(walls > 0, -1, p.astype(np.int16))
    p_frag = p[o:o + 41, o:o + 41, o:o + 41]
    wr = frag
    gfoam = np.where(wr > 0, -1, p_frag.astype(np.int16))
    slice_mosaic(gfoam, OUT / 'n_foam_slices.png', n=6, cmap=fcmap,
                 title='POTTS: центральный фрагмент 41³ — фазы (цвет) и стенки (чёрный)',
                 vmin=-1, vmax=7)
    Image.fromarray(full_render(np.where(walls > 0, 1, 0).astype(np.uint8),
                                f'POTTS: пена  L={L}', 'winter', 1.0, 40,
                                'стенки')).save(OUT / 'n_foam.png')
    art = {**by_sid('P015'), 'metrics': mw, 'wall_frac': float(frag.mean()),
           'deep_L': L, 'image': 'n_foam.png', 'slices': 'n_foam_slices.png',
           'note': 'центральный фрагмент 41^3'}
    selected.append(art)
    print('foam', round(time.time()-t0, 1), mw['network'], float(walls.mean()))

    # 3) EDEN — блоб
    t0 = time.time()
    rng = np.random.default_rng(1)
    ed = np.zeros((L, L, L), np.uint8)
    c = L // 2
    ed[c-1:c+2, c-1:c+2, c-1:c+2] = (rng.random((3, 3, 3)) < 0.7)
    for t in range(30):
        ed = ((ed == 1) | ((ed == 0) & (von6(ed) >= 1))).astype(np.uint8)
    m = shape_metrics(ed > 0)
    Image.fromarray(full_render(ed, f'EDEN: блоб  L={L}', 'inferno', 1.0, 30,
                                'ячейки')).save(OUT / 'b_eden.png')
    art = {**by_sid('E011'), 'metrics': m, 'deep_L': L, 'image': 'b_eden.png'}
    selected.append(art)
    print('eden', round(time.time()-t0, 1), m['blob'], m['largest'])

    # 4) TUMOR — блоб с некрозом
    t0 = time.time()
    tm = Tumor3D(T=40, D=0.05, cons=0.4, div=0.3, death=0.06, pmax=0.5).run(L)
    m = shape_metrics(tm > 0)
    cmap = ListedColormap(['#05070a', '#ff8c42', '#7a1f1f'])
    Image.fromarray(full_render(tm, f'TUMOR: блоб  L={L}', cmap, 2.0, 40,
                                'опухоль')).save(OUT / 'b_tumor.png')
    slice_mosaic(tm, OUT / 'b_tumor_slices.png', n=6, cmap=cmap,
                 title=f'TUMOR: срезы (оранжевый — опухоль, тёмный — некроз), L={L}',
                 vmin=0, vmax=2)
    art = {**by_sid('T008'), 'metrics': m, 'deep_L': L, 'image': 'b_tumor.png',
           'slices': 'b_tumor_slices.png', 'necrotic': float((tm == 2).mean())}
    selected.append(art)
    print('tumor', round(time.time()-t0, 1), m['blob'], int((tm > 0).sum()))

    # 5) BOND — графовая сеть (дополнительно)
    t0 = time.time()
    b = Bond3D(T=120, p=0.2, loops=True).run(L)
    g = bond_graph_metrics(b)
    n, e = graph_from_bonds(b)
    Image.fromarray(render_graph(n, e, size=640, canvas_L=L, show_nodes=False,
                                 edge_width=2, front_radius=60,
                                 label='BOND: граф связей')).save(OUT / 'n_bond.png')
    art = {**by_sid('B013'), 'metrics': {'nodes': g['nodes'], 'edges': g['edges'],
                                        'mean_degree': g['mean_degree'],
                                        'branches': g['branch_nodes']},
           'deep_L': L, 'image': 'n_bond.png'}
    selected.append(art)
    print('bond', round(time.time()-t0, 1), g['nodes'], g['edges'])

    # funnel + scatter
    f = SCREEN['funnel']
    fig, ax = plt.subplots(figsize=(7, 3.6))
    vals = [f['total'], f['alive'], f['structured'], len(selected)]
    labels = ['все правила', 'живые', 'структурные', 'отобрано']
    ax.bar(labels, vals, color=['#4a5568', '#3182ce', '#38a169', '#dd6b20'])
    for i, v in enumerate(vals):
        ax.text(i, v + 1, str(v), ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel('правил')
    ax.set_title('Воронка отсева: 100 → сеть/блоб')
    fig.tight_layout()
    fig.savefig(OUT / 'funnel.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    fam_color = {'potts': '#805ad5', 'invasion': '#e53e3e', 'bond': '#3182ce',
                 'eden': '#38a169', 'tumor': '#dd6b20'}
    for fam, col in fam_color.items():
        xs = [r for r in SCREEN['rules'] if r['family'] == fam]
        ax.scatter([r['network'] for r in xs], [r['blob'] for r in xs],
                   s=16, alpha=0.6, c=col, label=fam)
    for r in selected:
        ax.scatter(r['network'], r['blob'], s=90, facecolors='none',
                   edgecolors='white', linewidths=1.2)
    ax.set_xlabel('network score'); ax.set_ylabel('blob score')
    ax.set_title('100 правил: сеть против блоба (белые кружки — отобранные)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'morph_scatter.png', dpi=150)
    plt.close(fig)

    DEEP['selected'] = selected
    DEEP['funnel'] = f
    DEEP['checks'] = DEEP['checks']
    check('100 правил в скрининге', len(SCREEN['rules']) == 100, len(SCREEN['rules']))
    check('отобрано >= 4 правил', len(selected) >= 4, len(selected))
    check('есть сеть (скелет) и блоб', any('skel' in r for r in selected) and
          any(r['family'] == 'eden' for r in selected), True)
    check('инвазионная сеть не пуста', selected[0]['metrics']['largest'] > 1000,
          selected[0]['metrics']['largest'])
    check('блоб Eden компактен', selected[2]['metrics']['compactness'] > 0.2,
          round(selected[2]['metrics']['compactness'], 2))
    (OUT / 'deep.json').write_text(json.dumps(DEEP, ensure_ascii=False, indent=1),
                                   encoding='utf-8')
    print(f'== deep.json: {sum(c["ok"] for c in DEEP["checks"])}/{len(DEEP["checks"])} PASS ==')


if __name__ == '__main__':
    main()
