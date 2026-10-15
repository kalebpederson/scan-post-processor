import os, re, subprocess, tempfile, csv, io, time
import numpy as np, cv2
cv2.setNumThreads(1)
HERE=os.path.dirname(os.path.abspath(__file__))
WORK=os.path.join(HERE,'work')
SCANS=os.path.join(HERE,'..','scans')
CORPUS=os.path.join(HERE,'..','corpus')
RATES=[100,120,133,150,175,200,250,300]
TMP=os.environ.get('R1_TMP',os.path.join(WORK,'tmp'))
os.makedirs(TMP,exist_ok=True)

def downsample(g, src_dpi, r):
    h,w=g.shape[:2]
    return cv2.resize(g,(max(1,round(w*r/src_dpi)),max(1,round(h*r/src_dpi))),interpolation=cv2.INTER_AREA)
def upsample(g, size):  # size=(w,h)
    return cv2.resize(g,size,interpolation=cv2.INTER_LINEAR)

def norm(tok):
    t=tok.strip()
    if not re.search(r'[A-Za-z0-9]',t): return None   # punctuation/symbol-only tokens dropped
    return re.sub(r'^[^\w]+|[^\w]+$','',t)  # strip leading/trailing punct
def ocr(g, dpi, psm=11, tag=None):
    """returns list of dict(text,conf,x,y,w,h,line) in pixel coords of g. g: uint8 gray."""
    fd,p=tempfile.mkstemp(suffix='.png',dir=TMP); os.close(fd)
    cv2.imwrite(p,g)
    env=dict(os.environ,OMP_THREAD_LIMIT='1')
    out=subprocess.run(['tesseract',p,'stdout','-l','eng','--psm',str(psm),'--oem','1','--dpi',str(int(dpi)),'tsv'],capture_output=True,text=True,env=env).stdout
    os.remove(p)
    rows=[]
    for r in csv.DictReader(io.StringIO(out),delimiter='\t',quoting=csv.QUOTE_NONE):
        if r['level']!='5': continue
        t=norm(r['text'] or '')
        if not t: continue
        rows.append(dict(text=t,conf=float(r['conf']),x=int(r['left']),y=int(r['top']),w=int(r['width']),h=int(r['height']),
                         line=(int(r['block_num']),int(r['par_num']),int(r['line_num']))))
    return rows
