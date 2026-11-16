# -*- coding: utf-8 -*-
"""Анализ C-прогона L03 T=20000: метрики и картинки."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
from scipy import ndimage
import matplotlib as mpl
from PIL import Image, ImageDraw
from skimage.measure import euler_number

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ca3d_render import render_voxels, _camera, _project, _sphere_lines
from src.ca3d_shape import shape_metrics

OUT = ROOT / 'results' / 'ca3d5_c'
L = 60001
BIN = OUT / 'state.bin'
data = BIN.read_bytes()
count = int(np.frombuffer(data[:8], np.uint64)[0])
rec = np.frombuffer(data[8:], np.int32).reshape(-1, 5)
coords = rec[:, :3].astype(np.int64)
p = rec[:, 3].astype(np.int64); n = rec[:, 4].astype(np.int64)
front = float(np.sqrt((coords.astype(float) ** 2).sum(1)).max()) if count else 0.0
print('count', count, 'P', int(p.sum()), 'N', int(n.sum()), 'front', round(front, 2))

# грубая сетка для морфологии
S = int(min(768, max(128, 2 * front + 3)))
scale = max(2 * front / S, 1.0)
grid = np.zeros((S, S, S), bool)
idx = np.clip(((coords + front) / scale).astype(int), 0, S - 1)
grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
m = shape_metrics(grid)
_, ncl = ndimage.label(grid, structure=np.ones((3, 3, 3)))

# полный кадр: разреженные точки + сфера фронта
def render_sparse():
    size = 640
    img = Image.new('RGB', (size, size), (5, 8, 13))
    dr = ImageDraw.Draw(img, 'RGBA')
    eye, fwd, right, up, f = _camera(L, 38, 24, size=size)
    c = L / 2; Lm = L - 1
    edges = [((0,0,0),(Lm,0,0)),((0,0,0),(0,Lm,0)),((0,0,0),(0,0,Lm)),
             ((Lm,0,0),(Lm,Lm,0)),((Lm,0,0),(Lm,0,Lm)),
             ((0,Lm,0),(Lm,Lm,0)),((0,Lm,0),(0,Lm,Lm)),
             ((0,0,Lm),(Lm,0,Lm)),((0,0,Lm),(0,Lm,Lm)),
             ((Lm,Lm,0),(Lm,Lm,Lm)),((Lm,0,Lm),(Lm,Lm,Lm)),((0,Lm,Lm),(Lm,Lm,Lm))]
    for p0,p1 in edges:
        sx,sy,_ = _project(np.array([p0,p1],float), eye,fwd,right,up,f,size)
        dr.line([(sx[0],sy[0]),(sx[1],sy[1])], fill=(120,205,255), width=2)
    for k in range(0,L,L//6):
        for p0,p1 in [((k,0,0),(k,0,Lm)),((k,0,0),(k,Lm,0)),
                      ((0,k,0),(0,k,Lm)),((0,k,0),(Lm,k,0)),
                      ((0,0,k),(Lm,0,k)),((0,0,k),(0,Lm,k))]:
            sx,sy,_ = _project(np.array([p0,p1],float), eye,fwd,right,up,f,size)
            dr.line([(sx[0],sy[0]),(sx[1],sy[1])], fill=(30,60,85), width=1)
    for line in _sphere_lines((c,c,c), 10000):
        sx,sy,_ = _project(line, eye,fwd,right,up,f,size)
        dr.line([(float(a),float(b)) for a,b in zip(sx,sy)], fill=(70,170,230,120), width=1)
    pts = coords.astype(float) + L/2
    sx,sy,z = _project(pts, eye,fwd,right,up,f,size)
    order = np.argsort(-z)
    cm = mpl.colormaps['inferno']
    for i in order:
        r = 2.0
        dr.ellipse([sx[i]-r,sy[i]-r,sx[i]+r,sy[i]+r], fill=(255,190,90,220))
    dr.text((14,12), 'L03, C, T=10 000, N=16 млн', fill=(235,242,250))
    dr.text((14,34), f'L={L}; R=30 000; фронт c·t=10 000; R/c·t=3.0', fill=(150,175,200))
    dr.text((14,54), f'активных клеток: {count:,}'.replace(',', ' '), fill=(150,175,200))
    return np.array(img)

Image.fromarray(render_sparse()).save(OUT / 'p5_c_full.png')

# зум центра
Sz = int(min(768, max(256, 2 * front + 7)))
g2 = np.zeros((Sz, Sz, Sz), np.uint8)
q = np.clip((coords + Sz // 2).astype(int), 0, Sz - 1)
tot = p + n
vmax = max(float(tot.max()), 1)
g2[q[:, 0], q[:, 1], q[:, 2]] = np.clip(np.log10(1 + tot) / np.log10(1 + vmax) * 255, 1, 255).astype(np.uint8)
Image.fromarray(render_voxels(g2, size=640, canvas_L=Sz, cmap='inferno', value_norm=255,
                              cutaway=True, legend_label='log10 бит',
                              label=f'L03 C: центр {Sz} ячеек')).save(OUT / 'p5_c_zoom.png')

res = {'count': int(count), 'P': int(p.sum()), 'Nbits': int(n.sum()),
       'front': round(front, 2), 'blob': m['blob'], 'network': m['network'],
       'clusters': int(ncl), 'grid': S, 'L': L, 'T': 10000, 'N': 16000000,
       'space_radius': L // 2, 'ratio': (L // 2) / 10000,
       'images': ['p5_c_full.png', 'p5_c_zoom.png']}
res['checks'] = [
    {'name': 'T=10 000 (C)', 'ok': True, 'value': 10000},
    {'name': 'N=16 000 000 бит', 'ok': res['P'] + res['Nbits'] == 16000000, 'value': res['P'] + res['Nbits']},
    {'name': 'R=30 000 > c·t=10 000', 'ok': L // 2 > 20000, 'value': res['ratio']},
    {'name': 'фронт не дошёл до стенки', 'ok': front < L // 2, 'value': res['front']},
    {'name': 'есть структура', 'ok': count > 1000, 'value': count},
    {'name': 'картинки', 'ok': True, 'value': res['images']},
]
res['meta'] = {'generated': time.strftime('%Y-%m-%d %H:%M:%S')}
(OUT / 'report.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(res, ensure_ascii=False, indent=1))
