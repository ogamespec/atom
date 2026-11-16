# -*- coding: utf-8 -*-
"""Рендер сетей: графы связей и скелеты кластеров в 3D."""
from __future__ import annotations

import numpy as np
import matplotlib
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager
from skimage.morphology import skeletonize

from src.ca3d_render import _camera, _project, _sphere_lines, CMAP


def _font(sz=14):
    try:
        return ImageFont.truetype(font_manager.findfont('DejaVu Sans'), sz)
    except Exception:
        return ImageFont.load_default()


def _cube(dr, canvas_L, eye, fwd, right, up, f, size, color=(60, 110, 150, 180)):
    L = canvas_L - 1
    edges = [((0, 0, 0), (L, 0, 0)), ((0, 0, 0), (0, L, 0)), ((0, 0, 0), (0, 0, L)),
             ((L, 0, 0), (L, L, 0)), ((L, 0, 0), (L, 0, L)),
             ((0, L, 0), (L, L, 0)), ((0, L, 0), (0, L, L)),
             ((0, 0, L), (L, 0, L)), ((0, 0, L), (0, L, L)),
             ((L, L, 0), (L, L, L)), ((L, 0, L), (L, L, L)), ((0, L, L), (L, L, L))]
    for p0, p1 in edges:
        sx, sy, _ = _project(np.array([p0, p1], float), eye, fwd, right, up, f, size)
        dr.line([(sx[0], sy[0]), (sx[1], sy[1])], fill=color, width=2)
    step = max(8, canvas_L // 8)
    gcol = (35, 65, 90, 130)
    for k in range(0, canvas_L, step):
        for p0, p1 in [((k, 0, 0), (k, 0, L)), ((k, 0, 0), (k, L, 0)),
                       ((0, k, 0), (0, k, L)), ((0, k, 0), (L, k, 0)),
                       ((0, 0, k), (L, 0, k)), ((0, 0, k), (0, L, k))]:
            sx, sy, _ = _project(np.array([p0, p1], float), eye, fwd, right, up, f, size)
            dr.line([(sx[0], sy[0]), (sx[1], sy[1])], fill=gcol, width=1)


def render_graph(nodes: np.ndarray, edges: np.ndarray, size: int = 640,
                 canvas_L: int | None = None, label: str | None = None,
                 node_color=(255, 190, 90), edge_color=(120, 200, 255),
                 azim: float = 38, elev: float = 24, max_edges: int = 40000,
                 show_nodes: bool = True, edge_width: int = 1,
                 front_radius: float | None = None) -> np.ndarray:
    """nodes: (N,3) координаты; edges: (E,2) индексы. Рисует 3D-граф."""
    canvas_L = canvas_L or (int(nodes.max()) + 2 if len(nodes) else 2)
    img = Image.new('RGB', (size, size), (5, 8, 13))
    dr = ImageDraw.Draw(img, 'RGBA')
    eye, fwd, right, up, f = _camera(canvas_L, azim, elev, size=size)
    _cube(dr, canvas_L, eye, fwd, right, up, f, size)
    if front_radius:
        c = (canvas_L - 1) / 2
        for line in _sphere_lines((c, c, c), min(front_radius, canvas_L / 2 - 0.5)):
            sx, sy, _ = _project(line, eye, fwd, right, up, f, size)
            dr.line([(float(a), float(b)) for a, b in zip(sx, sy)],
                    fill=(70, 160, 220, 90), width=1)
    if len(nodes) == 0 or len(edges) == 0:
        return np.array(img)
    sx, sy, z = _project(nodes.astype(float), eye, fwd, right, up, f, size)
    if len(edges) > max_edges:
        edges = edges[::int(np.ceil(len(edges) / max_edges))]
    midz = 0.5 * (z[edges[:, 0]] + z[edges[:, 1]])
    order = np.argsort(-midz)
    edges = edges[order]
    for a, b in edges:
        d = 0.55 + 0.45 * (1 - (0.5 * (z[a] + z[b]) - z.min()) / max(z.max() - z.min(), 1e-9))
        col = tuple(int(c * d) for c in edge_color) + (185,)
        dr.line([(sx[a], sy[a]), (sx[b], sy[b])], fill=col, width=edge_width)
    if show_nodes:
        for i in range(len(nodes)):
            r = 1.2
            dr.ellipse([sx[i] - r, sy[i] - r, sx[i] + r, sy[i] + r],
                       fill=node_color + (230,))
    if label:
        dr.text((12, 10), label, fill=(230, 240, 250), font=_font(15))
        dr.text((12, 32), f'узлов: {len(nodes):,}  рёбер: {len(edges):,}'.replace(',', ' '),
                fill=(150, 172, 198), font=_font(12))
    return np.array(img)


def graph_from_bonds(bonds: np.ndarray):
    L = bonds.shape[1]
    idx = -np.ones((L, L, L), int)
    coords = np.argwhere(bonds.any(0))
    idx[tuple(coords.T)] = np.arange(len(coords))
    edges = []
    for d, (dx, dy, dz) in enumerate(((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                                      (0, -1, 0), (0, 0, 1), (0, 0, -1))):
        if d % 2 == 1:
            continue
        src = np.argwhere(bonds[d])
        for x, y, z in src:
            q = (x + dx, y + dy, z + dz)
            if 0 <= q[0] < L and 0 <= q[1] < L and 0 <= q[2] < L and idx[q] >= 0:
                edges.append((idx[x, y, z], idx[q]))
    return coords, np.array(edges, int)


def graph_from_mask(mask: np.ndarray):
    """Скелет кластера как граф (6-связность)."""
    sk = skeletonize(mask.astype(bool))
    coords = np.argwhere(sk)
    idx = -np.ones(mask.shape, int)
    idx[tuple(coords.T)] = np.arange(len(coords))
    edges = []
    dirs = [(1, 0, 0), (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1),
            (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
            (0, 1, 0), (0, 1, 1), (0, 1, -1), (0, 0, 1)]
    for dx, dy, dz in dirs:
        src = np.argwhere(sk)
        for x, y, z in src:
            q = (x + dx, y + dy, z + dz)
            if 0 <= q[0] < mask.shape[0] and 0 <= q[1] < mask.shape[1] and 0 <= q[2] < mask.shape[2] and idx[q] >= 0:
                edges.append((idx[x, y, z], idx[q]))
    return coords, np.array(edges, int)
