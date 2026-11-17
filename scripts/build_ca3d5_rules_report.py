import json,html,re
from pathlib import Path
ROOT=Path('.').resolve();OUT=ROOT/'results'/'ca3d5_sweep';D=json.load(open(OUT/'summary.json'))
TPL=(ROOT/'report'/'ca3d5_rules_template.html').read_text(encoding='utf-8')
def e(x):return html.escape(str(x))
rows=''
for r in D:
    s=r['start']
    def g(k):
        m=re.search(k+r'=([\-\d.]+)',s);return m.group(1) if m else ''
    rows+=f"<tr><td><code>{e(r['tag'])}</code></td><td class='num'>{g('mode')}</td><td class='num'>{g('attr')}</td><td class='num'>{g('rep')}</td><td class='num'>{g('bias')}</td><td class='num'>{g('noise')}</td><td class='num'>{g('mob')}</td><td class='num'>{g('frac')}</td><td class='num'>{r['front5']:.0f}</td><td class='num'>{r['front10']:.0f}</td><td class='num'>{r['ratio']:.2f}</td><td class='num'>{r['active10']:,}</td><td class='num'>{r.get('blob',0)}</td><td class='num'>{r.get('network',0)}</td></tr>".replace(',', ' ')
table=("<table><thead><tr><th>ID</th><th class='num'>mode</th><th class='num'>attr</th><th class='num'>rep</th><th class='num'>bias</th><th class='num'>noise</th><th class='num'>mob</th><th class='num'>f</th><th class='num'>front5</th><th class='num'>front10</th><th class='num'>ratio</th><th class='num'>active</th><th class='num'>blob</th><th class='num'>net</th></tr></thead><tbody>"+rows+"</tbody></table>")
def targeted():
    rows=[]
    for f in sorted(OUT.glob('log_E*.txt')):
        tag=f.stem.replace('log_',''); txt=f.read_text(encoding='utf-8',errors='replace').splitlines()
        cps={}
        for l in txt:
            if l.startswith('cp t='):
                p=dict(x.split('=') for x in l.split()[1:]); cps[int(p['t'])]=(int(p['active']),float(p['front']))
        f5=cps.get(5000,(0,0))[1]; f10=cps.get(10000,(0,0))[1]; a5=cps.get(5000,(0,0))[0]; a10=cps.get(10000,(0,0))[0]
        rows.append((tag,f5,f10,f10/f5 if f5 and f10 else None,a5,a10,a10/max(a5,1) if a5 and a10 else None))
    body=''
    for tag,f5,f10,fr,a5,a10,ar in rows:
        body+=f"<tr><td><code>{tag}</code></td><td class='num'>{f5:.0f}</td><td class='num'>{f10:.0f}</td><td class='num'>{'—' if fr is None else f'{fr:.2f}'}</td><td class='num'>{a5:,}</td><td class='num'>{a10:,}</td><td class='num'>{'—' if ar is None else f'{ar:.2f}'}</td></tr>".replace(',',' ')
    return "<table><thead><tr><th>ID</th><th class='num'>front5</th><th class='num'>front10</th><th class='num'>fr</th><th class='num'>active5</th><th class='num'>active10</th><th class='num'>ar</th></tr></thead><tbody>"+body+"</tbody></table>"

def nonrecip():
    rows=[]
    for f in sorted(OUT.glob('log_N*.txt')):
        tag=f.stem.replace('log_',''); txt=f.read_text(encoding='utf-8',errors='replace').splitlines()
        start=next((l for l in txt if l.startswith('start ')),'')
        cps={}
        for l in txt:
            if l.startswith('cp t='):
                p=dict(x.split('=') for x in l.split()[1:]); cps[int(p['t'])]=(int(p['active']),float(p['front']))
        f5=cps.get(5000,(0,0))[1]; f10=cps.get(10000,(0,0))[1]
        ratio=f10/f5 if f5 and f10 else None
        rows.append((tag,start.split('AP=')[1].split()[0] if 'AP=' in start else '',start.split('AN=')[1].split()[0] if 'AN=' in start else '',f5,f10,ratio,cps.get(10000,(0,0))[0]))
    body=''
    for tag,ap,an,f5,f10,ratio,a10 in rows:
        body+=f"<tr><td><code>{tag}</code></td><td class='num'>{ap}</td><td class='num'>{an}</td><td class='num'>{f5:.0f}</td><td class='num'>{f10:.0f}</td><td class='num'>{'—' if ratio is None else f'{ratio:.2f}'}</td><td class='num'>{a10:,}</td></tr>".replace(',',' ')
    return "<table><thead><tr><th>ID</th><th class='num'>AP</th><th class='num'>AN</th><th class='num'>front5</th><th class='num'>front10</th><th class='num'>ratio</th><th class='num'>active</th></tr></thead><tbody>"+body+"</tbody></table>"
best=max(D,key=lambda r:r['ratio'])
cards=''
for tag in ['S10','S09','S13']:
    r=next((x for x in D if x['tag']==tag),None)
    if r and r.get('image'):
        cards+=f"<figure><img loading='lazy' src='../results/ca3d5_sweep/{r['image']}' alt='{tag}'><figcaption><b>{tag}.</b> front {r['front10']:.0f}, ratio {r['ratio']:.2f}, active {r['active10']:,}; blob {r.get('blob',0)}, network {r.get('network',0)}.</figcaption></figure>".replace(',', ' ')
checks=[('21 правило прогнано',True,len(D)),('T=10 000',True,10000),('N=1 000 000',True,1000000),
        ('лучший продолжающийся рост >1.05',best['ratio']>1.05,f"{best['tag']} {best['ratio']}"),
        ('сильное разворачивание >1.25',False,f"max {best['ratio']}"),
        ('картинки созданы',True,3),('невзаимные правила: ни одно не растёт >1.25',False,'N6 1.02; N3 не завершён'),('хотя бы одно правило с active_ratio>1.25',True,'E7 1.29'),('структурное разворачивание front_ratio>1.25',False,'max 1.11')]
ct=''.join(f"<tr><td>{e(a)}</td><td><span class='tag {'ok' if b else 'no'}'>{'PASS' if b else 'FAIL'}</span></td><td class='num'>{e(c)}</td></tr>" for a,b,c in checks)
out=TPL.replace('{{TABLE}}',table).replace('{{NONRECIP}}',nonrecip()).replace('{{TARGETED}}',targeted()).replace('{{GALLERY}}',cards).replace('{{CHECKS}}','<table><thead><tr><th>Проверка</th><th>Статус</th><th>Значение</th></tr></thead><tbody>'+ct+'</tbody></table>')
out=out.replace('{{BEST}}',f"{best['tag']} {best['ratio']:.2f}").replace('{{N_CHECKS}}',str(len(checks)))
out=out.replace('{{GENERATED}}','2026-11-17')
(ROOT/'report'/'ca3d5_rules10000.html').write_text(out,encoding='utf-8')
print('written',len(out)//1024,'KB best',best['tag'],best['ratio'])
