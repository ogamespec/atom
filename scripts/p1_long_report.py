import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
import matplotlib as mpl
OUT=Path('results/ca2d_gif'); size=420; font=ImageFont.load_default()
def frame(p,T,box=320):
 d=Path(p).read_bytes();cnt=int(np.frombuffer(d[:8],np.uint64)[0]);rec=np.frombuffer(d[8:],np.int32).reshape(-1,4)
 c=rec[:,:2].astype(float);p_=rec[:,2].astype(np.int64);n_=rec[:,3].astype(np.int64)
 front=float(np.sqrt((c**2).sum(1)).max());S=box;sc=max(2*front/S,1)
 ch=np.zeros((S,S));q=np.clip(((c+front)/sc).astype(int),0,S-1);ch[q[:,0],q[:,1]]=p_-n_
 sm=ndimage.uniform_filter(ch,5);vmax=max(np.abs(sm).max(),1);t=np.clip(0.5+0.5*sm/vmax,0,1)
 rgb=(mpl.colormaps['coolwarm'](t)[:,:,:3]*255).astype(np.uint8);rgb[ch==0]=0
 im=Image.fromarray(rgb).resize((size,size),Image.NEAREST);dr=ImageDraw.Draw(im);dr.text((6,6),f'T={T}',fill=(255,255,255),font=font)
 return im,front
f100,fr100=frame(OUT/'p1_1m.bin.f0100000.bin',100000)
f200,fr200=frame(OUT/'p1_1m.bin.f0200000.bin',200000)
m=Image.new('RGB',(size*2,size),(0,0,0));m.paste(f100,(0,0));m.paste(f200,(size,0));m.save(OUT/'p1_100k_200k.png')
f100.save(OUT/'p1_100k.png');f200.save(OUT/'p1_200k.png')
print('front',fr100,fr200,'saved')
