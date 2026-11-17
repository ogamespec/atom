# -*- coding: utf-8 -*-
import json, numpy as np
from pathlib import Path
from scipy import ndimage
import matplotlib as mpl
from PIL import Image
import sys
ROOT=Path('.').resolve();sys.path.insert(0,str(ROOT))
from src.ca3d_shape import shape_metrics
from src.ca3d_render import render_voxels
OUT=ROOT/'results'/'ca3d5_sweep'
rows=json.load(open(OUT/'summary_raw.json'))
for r in rows:
    st=OUT/f"state_{r['tag']}.bin"
    if not st.exists(): r.update(blob=0,network=0,clusters=0); continue
    d=st.read_bytes(); cnt=int(np.frombuffer(d[:8],np.uint64)[0]); rec=np.frombuffer(d[8:],np.int32).reshape(-1,5)
    coords=rec[:,:3].astype(float)
    front=float(np.sqrt((coords**2).sum(1)).max()) if cnt else 0
    S=int(min(256,max(64,2*front+3))); scale=max(2*front/S,1.0)
    g=np.zeros((S,S,S),bool); q=np.clip(((coords+front)/scale).astype(int),0,S-1); g[q[:,0],q[:,1],q[:,2]]=True
    m=shape_metrics(g); _,cl=ndimage.label(g,structure=np.ones((3,3,3)))
    r.update(blob=round(m['blob'],2),network=round(m['network'],2),clusters=int(cl))
rows.sort(key=lambda r:(-r.get('ratio',0),-r.get('front10',0)))
json.dump(rows,open(OUT/'summary.json','w'),ensure_ascii=False,indent=1)
# render top-3: best ratio, largest front
sel=[]
for tag in ['S10','S09','S13']:
    r=next((x for x in rows if x['tag']==tag),None)
    if not r: continue
    st=OUT/f'state_{tag}.bin'; d=st.read_bytes(); rec=np.frombuffer(d[8:],np.int32).reshape(-1,5)
    coords=rec[:,:3].astype(int); tot=rec[:,3].astype(np.int64)+rec[:,4]
    front=r['front10']; S=int(2*front)+21; g=np.zeros((S,S,S),np.uint8)
    q=np.clip(coords+S//2,0,S-1); vmax=max(int(tot.max()),1)
    g[q[:,0],q[:,1],q[:,2]]=np.clip(np.log10(1+tot)/np.log10(1+vmax)*255,1,255).astype(np.uint8)
    img=render_voxels(g,size=700,canvas_L=S,cmap='inferno',value_norm=255,cutaway=False,
                      legend_label='log10 бит/ячейка',
                      label=f"{tag}: T=10000, фронт {front:.0f}, ratio {r['ratio']:.2f}")
    Image.fromarray(img).save(OUT/f'p5_{tag}.png'); r['image']=f'p5_{tag}.png'
json.dump(rows,open(OUT/'summary.json','w'),ensure_ascii=False,indent=1)
print('rows',len(rows)); print([(r['tag'],r['ratio'],r['front10'],r['network']) for r in rows[:8]])
