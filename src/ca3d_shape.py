# -*- coding: utf-8 -*-
"""Морфологические метрики: отличают сеть (network) от блоба (blob)."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import label, generate_binary_structure, binary_erosion
from skimage.morphology import skeletonize
from skimage.measure import euler_number

STRUCT6 = generate_binary_structure(3, 1)


def _n6(a):
    return sum(np.roll(a, s, axis=ax) for ax in range(3) for s in (1, -1))


def largest_component(mask):
    lab, n = label(mask, structure=STRUCT6)
    if n == 0:
        return np.zeros_like(mask), 0, 0
    sizes = np.bincount(lab.ravel())[1:]
    k = int(sizes.argmax()) + 1
    return lab == k, int(sizes.max()), int(n)


def percolates(mask):
    """Кластер соединяет противоположные стенки хотя бы по одной оси."""
    if not mask.any():
        return False
    out = False
    for ax in range(3):
        a = np.moveaxis(mask, ax, 0)
        out |= bool(a[0].any() and a[-1].any() and
                    label(a, structure=STRUCT6)[0].max() and
                    _connected_faces(a))
    return out


def _connected_faces(a):
    lab, n = label(a, structure=STRUCT6)
    if n == 0:
        return False
    left = set(np.unique(lab[0])) - {0}
    right = set(np.unique(lab[-1])) - {0}
    return bool(left & right)


def shape_metrics(mask, with_skeleton=True):
    mask = np.asarray(mask, bool)
    vol = int(mask.sum())
    total = mask.size
    if vol == 0:
        return {'occupancy': 0.0, 'n_clusters': 0, 'largest': 0, 'largest_frac': 0.0,
                'surface': 0, 'sv': 0.0, 'compactness': 0.0, 'percolates': False,
                'skel_len': 0, 'branches': 0, 'endpoints': 0, 'euler': 0,
                'network': 0.0, 'blob': 0.0, 'class': 'пусто'}
    lc, largest, ncl = largest_component(mask)
    surf = int((mask & ~binary_erosion(mask, STRUCT6)).sum())
    sv = surf / max(vol, 1)
    coords = np.argwhere(lc)
    center = coords.mean(0)
    r = float(np.sqrt(((coords - center) ** 2).sum(1)).max()) if len(coords) else 0.0
    compact = largest / (4.0 / 3.0 * np.pi * max(r, 1e-9) ** 3)
    per = percolates(lc)
    if with_skeleton and vol < 400000:
        sk = skeletonize(mask)
        skel_len = int(sk.sum())
        nb = _n6(sk)
        branches = int((sk & (nb > 2)).sum())
        ends = int((sk & (nb == 1)).sum())
    else:
        skel_len = branches = ends = 0
    try:
        euler = int(euler_number(mask, connectivity=1))
    except Exception:
        euler = 0
    # эвристические шкалы
    net = 0.0
    if per:
        net += 0.45
    net += 0.25 * min(sv / 0.55, 1.0)
    net += 0.20 * min((skel_len / max(vol, 1)) / 0.45, 1.0)
    net += 0.10 * min(branches / 80.0, 1.0)
    if vol / total > 0.35:
        net *= 0.55
    blob = 0.55 * min(compact / 0.65, 1.0) + 0.30 * min(largest / max(vol, 1), 1.0)
    blob += 0.15 * max(0.0, 1.0 - sv / 0.45)
    if largest / max(mask.size, 1) > 0.45:
        blob *= 0.4
    if largest < 300:
        blob *= 0.5
    return {'occupancy': vol / total, 'n_clusters': ncl, 'largest': largest,
            'largest_frac': largest / vol, 'surface': surf, 'sv': sv,
            'compactness': compact, 'percolates': bool(per),
            'skel_len': skel_len, 'branches': branches, 'endpoints': ends,
            'euler': euler, 'network': float(np.clip(net, 0, 1)),
            'blob': float(np.clip(blob, 0, 1)),
            'class': 'сеть' if net >= blob else 'блоб'}


def largest_domain(state, Q):
    """Самый большой домен (по фазе) для Potts."""
    best = np.zeros(state.shape, bool)
    best_n = 0
    for q in range(Q):
        m, n = label(state == q, structure=STRUCT6)
        if n == 0:
            continue
        sizes = np.bincount(m.ravel())[1:]
        k = int(sizes.argmax()) + 1
        if sizes[k - 1] > best_n:
            best_n = int(sizes[k - 1])
            best = m == k
    return best


def bond_graph_metrics(bonds):
    """Метрики графа связей (6,L,L,L)."""
    nodes = int(bonds.any(0).sum())
    edges = int(bonds.sum()) // 2
    deg = bonds.sum(0)
    branch = int((deg > 2).sum())
    leaves = int((deg == 1).sum())
    return {'nodes': nodes, 'edges': edges, 'branch_nodes': branch,
            'leaves': leaves, 'mean_degree': float(deg[deg > 0].mean()) if nodes else 0.0}
