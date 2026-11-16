# -*- coding: utf-8 -*-
"""Сборка отчёта L03 T=20000."""
from __future__ import annotations
import html, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'ca3d5_long'
D = json.loads((OUT / 'report.json').read_text(encoding='utf-8'))
TPL = (ROOT / 'report' / 'ca3d5_long_template.html').read_text(encoding='utf-8')
IMG = '../results/ca3d5_long'


def e(x):
    return html.escape(str(x))


def metrics_table():
    m = D['metrics']
    rows = [
        ('T', D['T']), ('L', D['L']), ('R = L/2', D['space_radius']),
        ('c·t', D['front_c_t']), ('R / c·t', D['ratio']),
        ('N (бит)', D['N']), ('P', m['P']), ('N-бит', m['N']),
        ('активных клеток', m['cells']), ('фронт структуры', m['front']),
        ('blob', m['blob']), ('network', m['network']), ('кластеров', m['p_clusters']),
    ]
    body = ''.join(f'<tr><td>{e(a)}</td><td class="num">{e(b)}</td></tr>' for a, b in rows)
    return f'<table><thead><tr><th>параметр</th><th class="num">значение</th></tr></thead><tbody>{body}</tbody></table>'


def checkpoints_table():
    rows = ''.join(f'<tr><td class="num">{t}</td><td class="num">{c:,}</td>'.replace(',', ' ') +
                   f'<td class="num">{s:,}</td></tr>'.replace(',', ' ')
                   for t, c, s in D['hist'])
    return ('<table><thead><tr><th class="num">T</th><th class="num">активных клеток</th>'
            '<th class="num">бит</th></tr></thead><tbody>' + rows + '</tbody></table>')


def checks_table():
    rows = ''.join(f'<tr><td>{e(c["name"])}</td>'
                   f'<td><span class="tag {"ok" if c["ok"] else "no"}">'
                   f'{"PASS" if c["ok"] else "FAIL"}</span></td>'
                   f'<td class="num small">{e(c["value"])}</td></tr>' for c in D['checks'])
    return ('<table><thead><tr><th>Проверка</th><th>Статус</th><th>Значение</th></tr></thead>'
            '<tbody>' + rows + '</tbody></table>')


def gallery():
    cards = []
    caps = {'p5_long_full.png': 'Полный куб L=60 001: фронт c·t=20 000, пространство R=30 000.',
            'p5_long_zoom.png': 'Центр: структура L03 после 20 000 тактов.'}
    for fn, cap in caps.items():
        if (OUT / fn).exists():
            cards.append(f'<figure><img loading="lazy" src="{IMG}/{fn}" alt="{e(cap)}">'
                         f'<figcaption><b>{e(fn)}.</b> {cap}</figcaption></figure>')
    return ''.join(cards)


def main():
    out = TPL
    out = out.replace('{{METRICS}}', metrics_table())
    out = out.replace('{{CHECKPOINTS}}', checkpoints_table())
    out = out.replace('{{CHECKS}}', checks_table())
    out = out.replace('{{GALLERY}}', gallery())
    out = out.replace('{{N_CHECKS}}', str(len(D['checks'])))
    out = out.replace('{{GENERATED}}', e(D['meta']['generated']))
    left = sorted(set(re.findall(r'\{\{(\w+)\}\}', out)))
    if left:
        print('не заполнены:', left)
    (ROOT / 'report' / 'ca3d5_long.html').write_text(out, encoding='utf-8')
    print(f'написано report/ca3d5_long.html ({len(out)/1024:.0f} КБ)')


if __name__ == '__main__':
    main()
