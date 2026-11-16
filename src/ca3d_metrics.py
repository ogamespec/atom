# -*- coding: utf-8 -*-
"""Метрики структурного богатства 3D-конфигураций клеточного автомата.

Все метрики считаются по активной области (кубу L(t)) и, где это возможно,
нормируются так, чтобы быть сравнимыми между правилами и размерами.
"""
from __future__ import annotations

import zlib

import numpy as np
from scipy import ndimage

STRUCT6 = ndimage.generate_binary_structure(3, 1)   # 6-связность


def shannon_entropy(a: np.ndarray, states: int) -> float:
    """Энтропия распределения состояний, бит (нормировка — в metrics)."""
    counts = np.bincount(a.ravel(), minlength=states).astype(np.float64)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def compression_ratio(a: np.ndarray) -> float:
    """len(zlib)/len(raw): 1 — несжимаемый шум, ~0 — порядок."""
    raw = a.tobytes()
    if not raw:
        return 1.0
    return len(zlib.compress(raw, 6)) / len(raw)


def cluster_stats(a: np.ndarray) -> dict:
    occ = a > 0
    lab, n = ndimage.label(occ, structure=STRUCT6)
    if n == 0:
        return {'n_clusters': 0, 'max_cluster': 0, 'mean_cluster': 0.0, 'sizes': []}
    sizes = np.bincount(lab.ravel())[1:]
    return {
        'n_clusters': int(n),
        'max_cluster': int(sizes.max()),
        'mean_cluster': float(sizes.mean()),
        'sizes': sizes.astype(np.int64),
    }


def box_dimension(a: np.ndarray, max_scale: int | None = None) -> float:
    """Фрактальная размерность методом box-counting (наклон log N / log(1/s))."""
    occ = a > 0
    L = occ.shape[0]
    max_scale = max_scale or max(4, L // 4)
    scales, counts = [], []
    s = 2
    while s <= max_scale and s <= L:
        pad = (-L) % s
        b = np.pad(occ, ((0, pad), (0, pad), (0, pad)), constant_values=False)
        m = b.shape[0] // s
        blocks = b.reshape(m, s, m, s, m, s).any(axis=(1, 3, 5))
        n = int(blocks.sum())
        if n > 0:
            scales.append(s)
            counts.append(n)
        s *= 2
    if len(scales) < 2:
        return float('nan')
    x = np.log(1.0 / np.array(scales, float))
    y = np.log(np.array(counts, float))
    return float(np.polyfit(x, y, 1)[0])


def radial_stats(a: np.ndarray, center: int | None = None) -> dict:
    """Радиальный профиль занятости и радиус фронта."""
    occ = a > 0
    L = occ.shape[0]
    c = center if center is not None else L // 2
    idx = np.argwhere(occ)
    if len(idx) == 0:
        return {'front_radius': 0, 'radius_90': 0, 'radial_profile': []}
    r = np.sqrt(((idx - np.array([c, c, c])) ** 2).sum(1))
    hist, edges = np.histogram(r, bins=np.arange(0, L, 1.0))
    # нормировка на объём сферического слоя
    rr = 0.5 * (edges[:-1] + edges[1:])
    shell = 4.0 * np.pi * np.maximum(rr, 0.5) ** 2
    prof = hist / shell
    cum = np.cumsum(hist) / max(hist.sum(), 1)
    radius_90 = float(rr[np.searchsorted(cum, 0.9)]) if len(rr) else 0.0
    return {
        'front_radius': float(r.max()),
        'radius_90': radius_90,
        'radial_profile': prof.tolist(),
    }


def activity_fraction(prev: np.ndarray, cur: np.ndarray) -> float:
    """Доля изменившихся ячеек (0..1)."""
    if prev.shape != cur.shape or prev.size == 0:
        return 0.0
    return float(np.count_nonzero(prev != cur) / prev.size)


def persistence_fn(prev: np.ndarray, cur: np.ndarray) -> float:
    """Доля занятых сейчас ячеек, занятых и раньше (устойчивость структуры)."""
    now = cur > 0
    if now.sum() == 0:
        return 0.0
    return float(((cur > 0) & (prev > 0)).sum() / now.sum())


def evaluate(a: np.ndarray, rule, t: int, prev: np.ndarray | None = None,
             with_clusters: bool = True, activity: float | None = None,
             persistence: float | None = None) -> dict:
    """Полный набор метрик для одного снимка."""
    states = rule.states
    occ = float(np.count_nonzero(a) / a.size)
    H = shannon_entropy(a, states)
    comp = compression_ratio(a)
    out = {
        't': int(t),
        'L': int(a.shape[0]),
        'states_used': int(len(np.unique(a))),
        'occupancy': occ,
        'entropy_bits': max(H, 0.0),
        'entropy_norm': max(H, 0.0) / np.log2(states),
        'compression': comp,
        'bits_total': int(rule.bits[a].sum()),
        'bits_per_cell': float(rule.bits[a].mean()),
        'active_fraction': float((a > 0).mean()),
    }
    rad = radial_stats(a)
    out.update(rad)
    out['box_dim'] = box_dimension(a)
    if with_clusters:
        cl = cluster_stats(a)
        out.update({k: v for k, v in cl.items() if k != 'sizes'})
        sizes = cl['sizes']
        if len(sizes) >= 5:
            # наклон хвоста распределения кластеров (лог-лог), если он есть
            s = np.sort(sizes)[::-1]
            s = s[s > 0]
            k = max(2, int(0.2 * len(s)))
            if len(s) > k + 1:
                x = np.log(np.arange(1, k + 1))
                y = np.log(s[:k])
                out['cluster_slope'] = float(np.polyfit(x, y, 1)[0])
    if activity is not None:
        out['activity'] = float(activity)
    elif prev is not None and prev.shape == a.shape:
        out['activity'] = activity_fraction(prev, a)
    else:
        out['activity'] = float('nan')
    if persistence is not None:
        out['persistence'] = float(persistence)
    elif prev is not None and prev.shape == a.shape:
        out['persistence'] = persistence_fn(prev, a)
    else:
        out['persistence'] = float('nan')
    out['richness'] = richness(out, states)
    out['class'] = classify(out)
    return out


def richness(m: dict, states: int) -> float:
    """Эвристический индекс богатства структуры (0..1), только для ранжирования.

    Состав: разнообразие состояний, сжимаемость (порядок), иерархия кластеров,
    фрактальность и устойчивая активность. Штрафы: туман (occ > 0.6), заморозка.
    """
    if m['occupancy'] <= 1e-6:
        return 0.0
    diversity = np.clip(m['entropy_norm'], 0, 1)
    structure = np.clip(1.0 - m['compression'], 0, 1)
    hierarchy = np.clip(np.log10(1 + m.get('n_clusters', 0)) / 2.5, 0, 1)
    frac = np.clip((3.0 - (m['box_dim'] if np.isfinite(m['box_dim']) else 3.0)) / 1.5, 0, 1)
    act = np.clip(m.get('activity', 0) / 0.02, 0, 1) if np.isfinite(m.get('activity', np.nan)) else 0.5
    val = 0.32 * diversity + 0.26 * structure + 0.20 * hierarchy + 0.12 * frac + 0.10 * act
    if m['occupancy'] > 0.6:
        val *= 0.55
    if np.isfinite(m.get('activity', np.nan)) and m['activity'] < 1e-5:
        val *= 0.35
    return float(np.clip(val, 0, 1))


def classify(m: dict) -> str:
    if m['occupancy'] <= 1e-6:
        return 'мёртвое'
    if np.isfinite(m.get('activity', np.nan)) and m['activity'] < 1e-5 and m['t'] > 10:
        return 'замороженное'
    if m['occupancy'] > 0.6:
        return 'туман'
    if m['box_dim'] < 2.5 and m['occupancy'] < 0.35:
        return 'фрактальное'
    if m['compression'] < 0.35 and m['occupancy'] < 0.45:
        return 'кристалл'
    return 'хаотическое'
