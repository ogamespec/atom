# -*- coding: utf-8 -*-
"""Сборка HTML третьей части: консервативный битовый автомат."""
from __future__ import annotations
import html, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'ca3d3'
D = json.loads((OUT / 'report.json').read_text(encoding='utf-8'))
TPL = (ROOT / 'report' / 'ca3d3_template.html').read_text(encoding='utf-8')
IMG = '../results/ca3d3'


def e(x):
    return html.escape(str(x))


def rules_table():
    rows = []
    for r in D['rules']:
        rows.append(
            f'<tr><td><code>{e(r["sid"])}</code></td><td>{e(r["desc"])}</td>'
            f'<td class="num">{r["rot"]}</td><td>{e(r["perm"])}</td>'
            f'<td class="num">{int(r["field"])}</td><td class="num">{r["cells"]:,}</td>'.replace(',', ' ') +
            f'<td class="num">{r["front"]:.1f}</td><td class="num">{r["max_bits"]}</td>'
            f'<td>{e(r["class"])}</td></tr>')
    return ('<table><thead><tr><th>ID</th><th>правило</th><th class="num">rot</th>'
            '<th>perm</th><th class="num">field</th><th class="num">клеток</th>'
            '<th class="num">фронт</th><th class="num">max бит</th><th>класс</th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>')


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
    caps = {
        'R01': 'R01: поворот 0. Шесть кубов-лучей вдоль осей плюс центральное ядро.',
        'R02': 'R02: поворот на 1 бит. Биты перемешались в компактный клубок.',
        'R04': 'R04: поворот на 3. Промежуточная форма.',
        'R09': 'R09: поворот 1 + перестановка байтов. Более рыхлое облако.',
        'R12': 'R12: поворот 8 + поле направлений. Облако с изогнутыми траекториями.',
    }
    cards = []
    for r in D['rules']:
        if not r.get('image'):
            continue
        cards.append(f'<figure><img loading="lazy" src="{IMG}/{r["image"]}" alt="{e(r["sid"])}">'
                     f'<figcaption><b>{e(r["sid"])}.</b> {e(caps.get(r["sid"], r["desc"]))} '
                     f'Клеток: {r["cells"]:,}; фронт {r["front"]:.1f}; максимум {r["max_bits"]} бит/ячейка.'
                     .replace(',', ' ') + '</figcaption></figure>')
    return '<div class="gallery">' + ''.join(cards) + '</div>'


def main():
    out = TPL
    out = out.replace('{{SEED_BITS}}', f'{D["meta"]["seed_bits"]:,}'.replace(',', ' '))
    out = out.replace('{{N_RULES}}', str(len(D['rules'])))
    out = out.replace('{{N_CHECKS}}', str(len(D['checks'])))
    out = out.replace('{{RULES_TABLE}}', rules_table())
    out = out.replace('{{CHECKS}}', checks_table())
    out = out.replace('{{GALLERY}}', gallery())
    out = out.replace('{{GENERATED}}', e(D['meta']['generated']))
    out = out.replace('{{PYTHON}}', e(D['meta']['python']))
    left = sorted(set(re.findall(r'\{\{(\w+)\}\}', out)))
    if left:
        print('не заполнены:', left)
    (ROOT / 'report' / 'ca3d3.html').write_text(out, encoding='utf-8')
    print(f'написано report/ca3d3.html ({len(out)/1024:.0f} КБ)')


if __name__ == '__main__':
    main()
