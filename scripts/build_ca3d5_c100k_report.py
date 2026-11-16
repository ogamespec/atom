# -*- coding: utf-8 -*-
import html, json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results'/'ca3d5_c100k'
D=json.loads((OUT/'report.json').read_text(encoding='utf-8'))
TPL=(ROOT/'report'/'ca3d5_c100k_template.html').read_text(encoding='utf-8')
IMG='../results/ca3d5_c100k'
def e(x): return html.escape(str(x))
def metrics():
    rows=[('N',D['N']),('P',D['P']),('N-бит',D['Nbits']),('активных клеток',D['count']),
          ('фронт структуры',D['front']),('blob',D['blob']),('network',D['network']),
          ('кластеров',D['clusters']),('L',D['L']),('R',D['space_radius']),('R/c·t',D['ratio'])]
    body=''.join(f'<tr><td>{e(a)}</td><td class="num">{e(b)}</td></tr>' for a,b in rows)
    return f'<table><thead><tr><th>параметр</th><th class="num">значение</th></tr></thead><tbody>{body}</tbody></table>'
def checkpoints():
    rows=[]
    for ln in (OUT/'run.log').read_text(encoding='utf-8',errors='replace').splitlines():
        if ln.startswith('cp t='):
            p=dict(x.split('=') for x in ln.split()[1:])
            rows.append(f'<tr><td class="num">{p["t"]}</td><td class="num">{int(p["active"]):,}</td>'.replace(',', ' ')+f'<td class="num">{int(p["bits"]):,}</td>'.replace(',', ' ')+f'<td class="num">{p["front"]}</td></tr>')
    return ('<table><thead><tr><th class="num">T</th><th class="num">активных</th><th class="num">бит</th><th class="num">фронт</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>')
def checks():
    rows=''.join(f'<tr><td>{e(c["name"])}</td><td><span class="tag {"ok" if c["ok"] else "no"}">{"PASS" if c["ok"] else "FAIL"}</span></td><td class="num small">{e(c["value"])}</td></tr>' for c in D['checks'])
    return f'<table><thead><tr><th>Проверка</th><th>Статус</th><th>Значение</th></tr></thead><tbody>{rows}</tbody></table>'
def gallery():
    caps={'p5_c_full.png':'Полный куб L=300 001: R=150 000, фронт c·t=100 000.','p5_c_zoom.png':'Центр: структура после 100 000 тактов.'}
    return ''.join(f'<figure><img loading="lazy" src="{IMG}/{fn}" alt="{e(cap)}"><figcaption><b>{e(fn)}.</b> {cap}</figcaption></figure>' for fn,cap in caps.items() if (OUT/fn).exists())
def main():
    out=TPL
    out=out.replace('{{METRICS}}',metrics()).replace('{{CHECKPOINTS}}',checkpoints()).replace('{{CHECKS}}',checks()).replace('{{GALLERY}}',gallery())
    out=out.replace('{{N_CHECKS}}',str(len(D['checks']))).replace('{{GENERATED}}',e(D['meta']['generated']))
    left=sorted(set(re.findall(r'\{\{(\w+)\}\}',out)))
    if left: print('не заполнены:',left)
    (ROOT/'report'/'ca3d5_c100k.html').write_text(out,encoding='utf-8')
    print('написано report/ca3d5_c100k.html',len(out)//1024,'KB')
if __name__=='__main__': main()
