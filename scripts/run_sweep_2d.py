import json,subprocess,time,sys
from pathlib import Path
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize
ROOT=Path('.').resolve();OUT=ROOT/'results'/'ca2d_sweep';OUT.mkdir(exist_ok=True)
BIN=ROOT/'native'/'ca2d';T=10000;N=1_000_000
rng=np.random.default_rng(23)
cfgs=[('S00',1,1,1,1,1,0.2,0.5,20,0.8),
      ('S01',1,2,0,-2,0,0.0,0.3,50,0.8),
      ('S02',0,0,2,0,2,0.0,0.3,50,0.5),
      ('S03',1,-2,-2,-2,-2,0.0,0.5,50,0.5),
      ('S04',0,0,0,0,0,0.5,0.8,100,0.5)]
for i in range(25):
    cfgs.append((f'S{i+5:02d}',int(rng.integers(0,2)),
                 float(rng.choice([0,0.5,1,2,-2])),float(rng.choice([0,0.5,1,2,-2])),
                 float(rng.choice([0,0.5,1,2,-2])),float(rng.choice([0,0.5,1,2,-2])),
                 float(rng.choice([0,0.2,0.5])),float(rng.choice([0,0.3,0.8])),
                 int(rng.choice([5,20,50,100])),float(rng.choice([0.5,0.8]))))
def metrics(path,front):
    d=Path(path).read_bytes();cnt=int(np.frombuffer(d[:8],np.uint64)[0]);rec=np.frombuffer(d[8:],np.int32).reshape(-1,4)
    if cnt==0:return dict(count=0,front=0,blob=0,network=0,clusters=0)
    c=rec[:,:2].astype(float)
    S=int(min(512,max(64,2*front+3)));sc=max(2*front/S,1)
    g=np.zeros((S,S),bool);q=np.clip(((c+front)/sc).astype(int),0,S-1);g[q[:,0],q[:,1]]=True
    lab,n=ndimage.label(g,structure=np.ones((3,3)))
    sizes=np.bincount(lab.ravel())[1:] if n else np.array([0])
    largest=int(sizes.max()) if n else 0; area=int(g.sum())
    per=int((g&~ndimage.binary_erosion(g)).sum())
    compact=4*np.pi*area/max(per*per,1) if area and per else 0
    sk=skeletonize(g); skl=int(sk.sum()); nb=ndimage.convolve(sk.astype(int),np.ones((3,3)),mode='constant'); branches=int((sk&(nb>4)).sum())
    blob=0.6*min(compact/0.8,1)+0.4*min(largest/max(area,1),1)
    net=0.5*min(skl/max(area,1)/0.4,1)+0.3*min(branches/40,1)+0.2*min(n/8,1)
    return dict(count=int(cnt),front=round(float(np.sqrt((c**2).sum(1)).max()),1),blob=round(float(blob),2),network=round(float(net),2),clusters=int(n))
R={'runs':[]};t0=time.time()
for tag,mode,ap,rp,an,rn,bias,noise,mob,frac in cfgs:
    st=OUT/f'state_{tag}.bin';log=OUT/f'log_{tag}.txt'
    r=subprocess.run([str(BIN),str(T),str(N),str(frac),str(mode),str(ap),str(rp),str(an),str(rn),str(bias),str(noise),str(mob),str(st)],capture_output=True,text=True,timeout=600)
    log.write_text(r.stdout+r.stderr,encoding='utf-8');cps={}
    for ln in (r.stdout+r.stderr).splitlines():
        if ln.startswith('cp t='):
            p=dict(x.split('=') for x in ln.split()[1:]);cps[int(p['t'])]=(int(p['active']),float(p['front']))
    fin=metrics(st,cps.get(10000,(0,0))[1])
    f5=cps.get(5000,(0,0))[1];a5=cps.get(5000,(0,0))[0]
    fr=fin['front']/max(f5,1e-9);ar=fin['count']/max(a5,1)
    rec=dict(tag=tag,mode=mode,ap=ap,rp=rp,an=an,rn=rn,bias=bias,noise=noise,mob=mob,frac=frac,
             **fin,front5=f5,front10=fin['front'],active5=a5,active10=fin['count'],
             front_ratio=round(fr,2),active_ratio=round(ar,2),evolves=bool(fr>1.25 or ar>1.25))
    R['runs'].append(rec)
    print(f"{tag} m={mode} AP={ap} RP={rp} AN={an} RN={rn} b={bias} n={noise} mob={mob} f={frac} cells={fin['count']} front={fin['front']} fr={fr:.2f} ar={ar:.2f} net={fin['network']} blob={fin['blob']} {'EVOLVES' if rec['evolves'] else ''}",flush=True)
R['meta']={'run_s':round(time.time()-t0,1),'T':T,'N':N}
(OUT/'sweep.json').write_text(json.dumps(R,ensure_ascii=False,indent=1),encoding='utf-8')
print('done',len(R['runs']),'evolves',sum(r['evolves'] for r in R['runs']),flush=True)
