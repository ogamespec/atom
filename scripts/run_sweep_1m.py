# -*- coding: utf-8 -*-
"""Свип правил: N=1 млн, T=10 000, локальный C-CA. Ищем рост/эволюцию."""
from __future__ import annotations
import json, subprocess, sys, time
from pathlib import Path
import numpy as np
from scipy import ndimage
import matplotlib as mpl
from PIL import Image

if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8',errors='replace')
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.ca3d_shape import shape_metrics
from src.ca3d_render import render_voxels

OUT=ROOT/'results'/'ca3d5_sweep'; OUT.mkdir(parents=True,exist_ok=True)
BIN=ROOT/'native'/'ca3d_sweep'
T=10000; N=1_000_000; L=30001
rng=np.random.default_rng(11)
configs=[('S00',1,1.0,1.0,0.2,0.5,20,0.8),
         ('S01',0,0.0,0.0,0.0,0.5,20,0.5),
         ('S02',1,0.5,2.0,0.2,0.5,20,0.8),
         ('S03',1,2.0,0.5,0.2,0.5,20,0.8)]
for i in range(20):
    configs.append((f'S{i+4:02d}', int(rng.integers(0,2)),
                    float(rng.choice([0.0,0.5,1.0,2.0])),
                    float(rng.choice([0.0,0.5,1.0,2.0])),
                    float(rng.choice([0.0,0.2,0.5])),
                    float(rng.choice([0.0,0.3,0.8])),
                    int(rng.choice([5,20,50])),
                    float(rng.choice([0.5,0.8]))))
S={'runs':[]}
def metrics(path,front):
    d=Path(path).read_bytes(); cnt=int(np.frombuffer(d[:8],np.uint64)[0])
    rec=np.frombuffer(d[8:],np.int32).reshape(-1,5)
    if cnt==0: return {'count':0,'front':0,'blob':0,'network':0,'clusters':0}
    coords=rec[:,:3].astype(float); tot=rec[:,3].astype(np.int64)+rec[:,4]
    S_=int(min(256,max(64,2*front+3))); scale=max(2*front/S_,1.0)
    g=np.zeros((S_,S_,S_),bool); q=np.clip(((coords+front)/scale).astype(int),0,S_-1)
    g[q[:,0],q[:,1],q[:,2]]=True
    m=shape_metrics(g); _,cl=ndimage.label(g,structure=np.ones((3,3,3)))
    return {'count':int(cnt),'front':round(float(np.sqrt((coords**2).sum(1)).max()),1),
            'blob':round(m['blob'],2),'network':round(m['network'],2),'clusters':int(cl)}
t0=time.time()
for tag,mode,attr,rep,bias,noise,mob,frac in configs:
    t1=time.time(); state=OUT/f'state_{tag}.bin'; log=OUT/f'log_{tag}.txt'
    cmd=[str(BIN),str(T),str(N),f'{frac}',str(mode),f'{attr}',f'{rep}',f'{bias}',f'{noise}',str(mob),str(state)]
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=1200)
    log.write_text(r.stdout+r.stderr,encoding='utf-8')
    cps={}
    for ln in (r.stdout+r.stderr).splitlines():
        if ln.startswith('cp t='):
            p=dict(x.split('=') for x in ln.split()[1:]); cps[int(p['t'])]=(int(p['active']),float(p['front']))
    fin=metrics(state, cps.get(10000,(0,0))[1])
    f5=cps.get(5000,(fin['count'],fin['front']))[1]; f10=fin['front']
    a5=cps.get(5000,(fin['count'],fin['front']))[0]; a10=fin['count']
    fr=float(f10/max(f5,1e-9)); ar=float(a10/max(a5,1))
    rec={'tag':tag,'mode':mode,'attr':attr,'rep':rep,'bias':bias,'noise':noise,'mob':mob,'frac':frac,
         **fin,'front5':f5,'front10':f10,'active5':a5,'active10':a10,
         'front_ratio':round(fr,3),'active_ratio':round(ar,3),
         'evolves': bool(fr>1.25 or ar>1.25), 'run_s':round(time.time()-t1,1)}
    S['runs'].append(rec)
    print(f"{tag} m={mode} a={attr} r={rep} b={bias} n={noise} mob={mob} f={frac} "
          f"cells={fin['count']} front={f10} fr={fr:.2f} ar={ar:.2f} net={fin['network']} blob={fin['blob']} {'EVOLVES' if rec['evolves'] else ''}",flush=True)
S['meta']={'generated':time.strftime('%Y-%m-%d %H:%M:%S'),'T':T,'N':N,'L':L,'run_s':round(time.time()-t0,1)}
(OUT/'sweep.json').write_text(json.dumps(S,ensure_ascii=False,indent=1),encoding='utf-8')
print('== done',len(S['runs']),'runs',round(time.time()-t0,1),'s; evolves',
      sum(r['evolves'] for r in S['runs']),flush=True)
