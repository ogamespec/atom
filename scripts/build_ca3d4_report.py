# -*- coding: utf-8 -*-
"""Сборка HTML части 4: 16 млн бит, обмен и случайность."""
from __future__ import annotations
import html, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'ca3d4'
DET = json.loads((OUT / 'report.json').read_text(encoding='utf-8'))
GOD = json.loads((OUT / 'god.json').read_text(encoding='utf-8'))
TPL = (ROOT / 'report' / 'ca3d4_template.html').read_text(encoding='utf-8')
IMG = '../results/ca3d4'


def e(x):
    return html.escape(str(x))


def det_table():
    rows = []
    for r in DET['rules']:
        rows.append(
            f'<tr><td><code>{e(r["sid"])}</code></td><td>{e(r["wmode"])}</td>'
            f'<td class="num">{e(r["K"])}</td><td>{e(r["exchange"])}</td>'
            f'<td class="num">{r["cells"]:,}</td>'.replace(',', ' ') +
            f'<td class="num">{r["front"]:.1f}</td><td class="num">{r["max_log"]:.2f}</td>'
            f'<td class="num">{r["compression"]:.3f}</td><td>{e(r["class"])}</td></tr>')
    head = ('<table><thead><tr><th>ID</th><th>веса</th><th class="num">K</th>'
            '<th>обмен</th><th class="num">клеток</th><th class="num">фронт</th>'
            '<th class="num">max log</th><th class="num">сжатие</th><th>класс</th>'
            '</tr></thead><tbody>')
    return head + ''.join(rows) + '</tbody></table>'


def god_table():
    rows = []
    for r in GOD['runs']:
        rows.append(
            f'<tr><td><code>{e(r["sid"])}</code></td><td>{e(r["mode"])}</td>'
            f'<td class="num">{r["seed"]}</td><td class="num">{r["cells"]:,}</td>'.replace(',', ' ') +
            f'<td class="num">{r["front"]:.1f}</td><td class="num">{r["max_log"]:.2f}</td>'
            f'<td class="num">{r["compression"]:.3f}</td><td class="num">{r["run_s"]} с</td></tr>')
    head = ('<table><thead><tr><th>ID</th><th>режим</th><th class="num">сид</th>'
            '<th class="num">клеток</th><th class="num">фронт</th><th class="num">max log</th>'
            '<th class="num">сжатие</th><th class="num">время</th></tr></thead><tbody>')
    return head + ''.join(rows) + '</tbody></table>'


def checks_table():
    allc = DET.get('checks', []) + GOD.get('checks', [])
    rows = []
    for c in allc:
        rows.append(f'<tr><td>{e(c["name"])}</td>'
                    f'<td><span class="tag {"ok" if c["ok"] else "no"}">'
                    f'{"PASS" if c["ok"] else "FAIL"}</span></td>'
                    f'<td class="num small">{e(c["value"])}</td></tr>')
    return ('<table><thead><tr><th>Проверка</th><th>Статус</th><th>Значение</th></tr></thead>'
            '<tbody>' + ''.join(rows) + '</tbody></table>')


def fig(path, cap, num_):
    return (f'<figure><img loading="lazy" src="{IMG}/{path}" alt="{e(cap)}">'
            f'<figcaption><b>Рис. {num_}.</b> {cap}</figcaption></figure>')


def det_gallery():
    caps = {
        'C01': 'C01, равномерный перенос: облако вокруг центра.',
        'C05': 'C05, радиальный поток наружу: лобастая оболочка.',
        'C06': 'C06, радиальный поток + обмен по x: текстура меняется, фронт шире.',
        'C08': 'C08, радиальный поток с K = 2000: плотное ядро и гало.',
        'C10': 'C10, K = 500: ядро ещё компактнее.',
    }
    cards = []
    for r in DET['rules']:
        if not r.get('image'):
            continue
        cards.append(fig(r['image'], f"{e(caps.get(r['sid'], r['desc']))} "
                                  f"Клеток: {r['cells']:,}; фронт {r['front']:.1f}."
                       .replace(',', ' '), '2.' + r['sid'][1:]))
    cards.append(fig('p4_C05_slices.png', 'C05: срезы радиальной оболочки. '
                   'Видно, что это объёмное облако с плотным центром.', '2.6'))
    return '<div class="gallery">' + ''.join(cards) + '</div>'


def god_gallery():
    caps = {
        'split': 'G01: три случайных направления на клетку — ветвящееся облако.',
        'rot': 'G02: случайный поворот радиальных весов — «закрученная» оболочка.',
        'swap': 'G03: радиальный поток + случайный обмен.',
        'weights': 'G04: случайные целые веса 0..2 — рыхлое облако.',
    }
    cards = []
    for mode, cap in caps.items():
        p = OUT / f'p4g_{mode}.png'
        if p.exists():
            cards.append(fig(p.name, cap, '3.' + str(len(cards) + 1)))
    if (OUT / 'p4g_dice.png').exists():
        cards.append(fig('p4g_dice.png', '«Кости»: одно правило split, три сида. Крупные черты похожи, '
                        'детали и число активных клеток — разные.', '3.5'))
    return '<div class="gallery">' + ''.join(cards) + '</div>'


def main():
    out = TPL
    out = out.replace('{{DET_TABLE}}', det_table())
    out = out.replace('{{GOD_TABLE}}', god_table())
    out = out.replace('{{DET_GALLERY}}', det_gallery())
    out = out.replace('{{GOD_GALLERY}}', god_gallery())
    out = out.replace('{{CHECKS}}', checks_table())
    out = out.replace('{{N_CHECKS}}', str(len(DET.get('checks', [])) + len(GOD.get('checks', []))))
    out = out.replace('{{GENERATED}}', e(DET['meta']['generated']))
    out = out.replace('{{PYTHON}}', e(DET['meta']['python']))
    left = sorted(set(re.findall(r'\{\{(\w+)\}\}', out)))
    if left:
        print('не заполнены:', left)
    (ROOT / 'report' / 'ca3d4.html').write_text(out, encoding='utf-8')
    print(f'написано report/ca3d4.html ({len(out)/1024:.0f} КБ)')


if __name__ == '__main__':
    main()
