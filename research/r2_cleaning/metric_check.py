import sys; sys.path.insert(0,'.')
from common import *
L=json.load(open('labels.json')); regs={r['id']:r for r in L['pos']+L['neg']}
def masks(rgb,dpi,r):
    d,n=darkness(rgb,dpi); ds=cv2.GaussianBlur(d,(0,0),max(.6,.006*dpi)); x,y,w,h=r['box']
    m=np.zeros(d.shape,bool); m[y:y+h,x:x+w]=True
    old=m&(ds>=.05)&(ds<=r['dmax']); v=ds[m&(ds>=.03)]; thr=max(.04,.5*np.percentile(v,95)); core=m&(ds>=thr)&(ds<=r['dmax'])
    return old,core
for rid in ['pencil_pair03_a','grid_image04_b','dotted_epson_text']:
    r=regs[rid]; rgb=load_rgb(ROOT/r['file']); old,core=masks(rgb,r['dpi'],r); x,y,w,h=r['box']; mg=15
    sl=(slice(max(0,y-mg),y+h+mg),slice(max(0,x-mg),x+w+mg)); c=rgb[sl]
    def ov(mask,col):
        o=c.copy(); mm=mask[sl]; o[mm]=(0.4*o[mm]+0.6*np.array(col)).astype(np.uint8); return o
    tiles=[c,ov(old,(255,0,0)),ov(core,(0,200,0))]
    tiles=[cv2.copyMakeBorder(t,16,4,4,4,cv2.BORDER_CONSTANT,value=(128,128,128)) for t in tiles]
    for t,cap in zip(tiles,[rid+' native 1:1','OLD mask ds>=.05: %d px'%old.sum(),'CORE mask: %d px'%core.sum()]): cv2.putText(t,cap,(3,12),0,.4,(255,255,0),1)
    save_rgb(f'panels/metric_check_{rid}.png',np.concatenate(tiles,1))
