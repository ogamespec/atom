import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
import matplotlib as mpl
OUT=Path('results/ca2d_gif'); size=360
font=ImageFont.load_default()
def read(p):
 d=p.read_bytes();cnt=int(np.frombuffer(d[:8],np.uint64)[0]);rec=np.frombuffer(d[8:],np.int32).reshape(-1,4)
 return rec[:,:2].astype(float),rec[:,2].astype(np.int64),rec[:,3].astype(np.int64)
def frame(path,mode,T,box):
 c,p,n=read(path);front=float(np.sqrt((c**2).sum(1)).max())
 S=box; sc=max(2*front/S,1)
 if mode=='p1':
  ch=np.zeros((S,S));q=np.clip(((c+front)/sc).astype(int),0,S-1);ch[q[:,0],q[:,1]]=p-n
  sm=ndimage.uniform_filter(ch,5); vmax=max(np.abs(sm).max(),1)
  t=np.clip(0.5+0.5*sm/vmax,0,1); cm=mpl.colormaps['coolwarm']; rgb=(cm(t)[:,:,:3]*255).astype(np.uint8)
  rgb[ch==0]=0
 else:
  dens=np.zeros((S,S));q=np.clip(((c+front)/sc).astype(int),0,S-1);dens[q[:,0],q[:,1]]=np.log10(1+p+n)
  vmax=max(dens.max(),1);t=np.clip(dens/vmax,0,1); cm=mpl.colormaps['inferno']; rgb=(cm(t)[:,:,:3]*255).astype(np.uint8); rgb[dens==0]=0
 im=Image.fromarray(rgb).resize((size,size),Image.NEAREST); dr=ImageDraw.Draw(im)
 dr.text((6,6),f'T={T}',fill=(255,255,255),font=font)
 return im
for mode,name in [('p1','p1_domains.gif'),('p6','p6_web.gif')]:
 frames=[]; box=220 if mode=='p1' else 560
 for T in range(1000,10001,1000):
  frames.append(frame(OUT/f'{mode}_{T}.bin',mode,T,box))
 frames[0].save(OUT/name,save_all=True,append_images=frames[1:],duration=450,loop=0)
 # монтаж 4 кадра
 picks=[frames[0],frames[2],frames[5],frames[9]]
 m=Image.new('RGB',(size*4,size),(0,0,0))
 for i,im in enumerate(picks): m.paste(im,(i*size,0))
 m.save(OUT/name.replace('.gif','_montage.png'))
 print('saved',name)
