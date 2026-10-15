import json,cv2,numpy as np,random,sys
idx=json.load(open('index.json'));A=np.load('A7.npy')
dis=json.load(open('dis_big.json'));random.seed(2);sel=random.sample(dis,10)
def corr(im,a):
    h,w=im.shape[:2];M=cv2.getRotationMatrix2D((w/2,h/2),-a,1.0)
    return cv2.warpAffine(im,M,(w,h),flags=cv2.INTER_CUBIC,borderValue=(255,255,255))
rows=[]
for i in sel:
    im=cv2.imread(idx[i]['fn']);h,w=im.shape[:2]
    tiles=[]
    for nm,a in (('orig',0),('proj %.2f'%A[i,0],A[i,0]),('lsd %.2f'%A[i,5],A[i,5]),('hough %.2f'%A[i,1],A[i,1])):
        c=corr(im,a)[int(h*.25):int(h*.25)+330,50:w-50]
        c=c.copy()
        for y in range(0,c.shape[0],40): c[y,:]=(c[y,:]*0.6+np.array([0,0,255])*0.4).astype(np.uint8)
        cv2.putText(c,f'{i} {nm}',(5,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,200),1)
        tiles.append(c)
    rows.append(np.hstack([t[:,:380] for t in tiles]))
cv2.imwrite('dis_view.png',np.vstack(rows[:5])); cv2.imwrite('dis_view2.png',np.vstack(rows[5:]))
print(sel)
