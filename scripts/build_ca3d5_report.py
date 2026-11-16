# -*- coding: utf-8 -*-
"""Сборка HTML части 5: локальный CA без FFT."""
from __future__ import annotations
import html, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'ca3d5'
D = json.loads((OUT / 'local.json').read_text(encoding='utf-8'))
TPL = (ROOT / 'report' / 'ca3d5_template.html').read_text(encoding='utf-8')
IMG = '../results/ca3d5'


def e(x):
    return html.escape(str(x))


def rules_table():
    rows = []
    for r in D['rules']:
        rows.append(f'<tr><td><code>{e(r["tag"])}</code></td>'
                    f'<td class="num">{r["mob"]}</td><td class="num">{r["charge_frac"]:.2f}</td>'
                    f'<td class="num">{r["attr"]}</td><td class="num">{r["rep"]}</td>'
                    f'<td class="num">{r.get("noise", 0)}</td>'
                    f'<td class="num">{r["cells"]:,}</td>'.replace(',', ' ') +
                    f'<td class="num">{r["occ"]:.2f}</td><td class="num">{r["blob"]:.2f}</td>'
                    f'<td class="num">{r["network"]:.2f}</td>'
                    f'<td class="num">{r["p_clusters"]}/{r["n_clusters"]}</td></tr>')
    head = ('<table><thead><tr><th>ID</th><th class="num">mob</th><th class="num">f</th>'
            '<th class="num">attr</th><th class="num">rep</th><th class="num">noise</th>'
            '<th class="num">клеток</th><th class="num">occ</th><th class="num">blob</th>'
            '<th class="num">net</th><th class="num">кластеры p/n</th></tr></thead><tbody>')
    return head + ''.join(rows) + '</tbody></table>'


def long_table():
    rows = []
    for r in D['long']:
        rows.append(f'<tr><td><code>{e(r["tag"])}</code></td>'
                    f'<td class="num">{r["cells"]:,}</td>'.replace(',', ' ') +
                    f'<td class="num">{r["occ"]:.2f}</td><td class="num">{r["blob"]:.2f}</td>'
                    f'<td class="num">{r["network"]:.2f}</td>'
                    f'<td class="num">{r["front"]:.1f}</td>'
                    f'<td class="num">{r["Dx"]:.2f}</td><td class="num">{r["Qxx"]:.2f}</td></tr>')
    head = ('<table><thead><tr><th>ID</th><th class="num">клеток</th><th class="num">occ</th>'
            '<th class="num">blob</th><th class="num">net</th><th class="num">фронт</th>'
            '<th class="num">Dx</th><th class="num">Qxx</th></tr></thead><tbody>')
    return head + ''.join(rows) + '</tbody></table>'


def checks_table():
    rows = []
    for c in D['checks']:
        rows.append(f'<tr><td>{e(c["name"])}</td>'
                    f'<td><span class="tag {"ok" if c["ok"] else "no"}">'
                    f'{"PASS" if c["ok"] else "FAIL"}</span></td>'
                    f'<td class="num small">{e(c["value"])}</td></tr>')
    return ('<table><thead><tr><th>Проверка</th><th>Статус</th><th>Значение</th></tr></thead>'
            '<tbody>' + ''.join(rows) + '</tbody></table>')


def gallery():
    caps = {'L01': 'L01: медленный обмен, компактный блоб.',
            'L03': 'L03: асимметрия f=0.8 + diag-bias + шум — перколирующая сеть (net = 0.50).',
            'L08': 'L08: сильный шум, но всё равно компактный блоб.'}
    cards = []
    for r in D['long']:
        if r.get('image'):
            cards.append(f'<figure><img loading="lazy" src="{IMG}/{r["image"]}" alt="{e(r["tag"])}">'
                         f'<figcaption><b>{e(r["tag"])}, T=1000.</b> {caps.get(r["tag"], "")} '
                         f'Клеток: {r["cells"]:,}; blob {r["blob"]:.2f}.'.replace(',', ' ') +
                         '</figcaption></figure>')
    return ''.join(cards)


def main():
    out = TPL
    out = out.replace('{{RULES_TABLE}}', rules_table())
    out = out.replace('{{LONG_TABLE}}', long_table())
    out = out.replace('{{CHECKS}}', checks_table())
    out = out.replace('{{GALLERY}}', gallery())
    out = out.replace('{{N_CHECKS}}', str(len(D['checks'])))
    out = out.replace('{{GENERATED}}', e(D['meta']['generated']))
    out = out.replace('{{PYTHON}}', e(D['meta']['python']))
    left = sorted(set(re.findall(r'\{\{(\w+)\}\}', out)))
    if left:
        print('не заполнены:', left)
    (ROOT / 'report' / 'ca3d5.html').write_text(out, encoding='utf-8')
    print(f'написано report/ca3d5.html ({len(out)/1024:.0f} КБ)')


if __name__ == '__main__':
    main()
