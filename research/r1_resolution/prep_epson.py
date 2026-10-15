import cv2, sys
from lib import *
cv2.setNumThreads(4)
for d in [150,200,300,400,600,1200]:
    im=cv2.imread(f'{SCANS}/epson_{d}dpi.png',cv2.IMREAD_COLOR)
    g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    g=cv2.rotate(g,cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(f'{WORK}/gray_{d}.png',g); print(d,g.shape,flush=True)
