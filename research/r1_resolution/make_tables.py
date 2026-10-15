import json, numpy as np, time, cv2
from lib import *
rng=np.random.default_rng(0)
def ci(h,B=2000):
    h=np.asarray(h,float); m=[h[rng.integers(0,len(h),len(h))].mean() for _ in range(B)]; return h.mean(),np.percentile(m,2.5),np.percentile(m,97.5)
def pdiff(a,b,B=2000):
    a=np.asarray(a,float);b=np.asarray(b,float);d=a-b;m=[d[rng.integers(0,len(d),len(d))].mean() for _ in range(B)]
    return d.mean(),np.percentile(m,2.5),np.percentile(m,97.5)
out=[]
P=lambda s='':out.append(s)
R={}
for psm in (11,3):
    d=json.load(open(f'epson_results_psm{psm}.json')); R[psm]={(x['src'],x['r']):x for x in d['out']}
REFN=json.load(open('work/consensus.json'))
d11=json.load(open('epson_results_psm11.json'));P(f"Reference words: {d11['nref']}; classes (by mean bbox darkness at 600 dpi: bold>=50, mid 35-50, light<35): {d11['classes']}; bold alphabetic>=4 letters (headings/bold words): {d11['out'][0]['rt']['n_heading']}")
P()
P('### E1. Epson, simulation from 600 dpi scan (area down, bilinear up to 600 grid, OCR psm 11). Positional recall by class; ms = multiset recall. 95% bootstrap CI on pos_all.')
P('| r ppi | multiset recall | pos recall (all) [95% CI] | light gray halftone (n=270) | mid (n=64) | bold (n=54) | bold words/headings (n=16) | CER | direct-OCR ms recall (no upsample) |')
P('|---|---|---|---|---|---|---|---|---|')
for r in RATES+[400,600]:
    x=R[11][(600,r)]['rt']; m,lo,hi=ci(x['hits'])
    P(f"| {r} | {x['ms_recall']:.3f} | {m:.3f} [{lo:.3f},{hi:.3f}] | {x['pos_light']:.3f} | {x['pos_mid']:.3f} | {x['pos_bold']:.3f} | {x['pos_heading']:.2f} | {x['cer']:.3f} | {R[11][(600,r)]['direct']['ms_recall']:.3f} |")
P('(r=600 is the reference itself, so 1.000 by construction.)')
P()
P('### E2. Same, source = 1200 dpi scan (r<=600 compared on the 600 grid) and psm 3 repeat (source 600): multiset recall')
P('| r | src600 psm11 | src1200 psm11 | src600 psm3 | src1200 psm3 |'); P('|---|---|---|---|---|')
for r in RATES+[400]:
    P(f"| {r} | {R[11][(600,r)]['rt']['ms_recall']:.3f} | {R[11][(1200,r)]['rt']['ms_recall']:.3f} | {R[3][(600,r)]['rt']['ms_recall']:.3f} | {R[3][(1200,r)]['rt']['ms_recall']:.3f} |")
P(); P('### E3. Real hardware vs simulation (round-trip pos recall, psm 11; paired bootstrap diff HW minus sim600, 95% CI)')
P('| density | HW scan pos / ms | sim from 600 | sim from 1200 | HW-sim600 diff [CI] | HW light-class | sim600 light-class |'); P('|---|---|---|---|---|---|---|')
for dd in [150,200,300,400]:
    h=R[11][(dd,dd)]['rt']; s6=R[11][(600,dd)]['rt']; s12=R[11][(1200,dd)]['rt']; df=pdiff(h['hits'],s6['hits'])
    P(f"| {dd} | {h['pos_recall']:.3f} / {h['ms_recall']:.3f} | {s6['pos_recall']:.3f} / {s6['ms_recall']:.3f} | {s12['pos_recall']:.3f} / {s12['ms_recall']:.3f} | {df[0]:+.3f} [{df[1]:+.3f},{df[2]:+.3f}] | {h['pos_light']:.3f} | {s6['pos_light']:.3f} |")
P(); P('### E4. Scan at 300/400 dpi hardware, then downsample (pos recall round-trip) vs scanning directly at that low dpi')
P('| target r | HW scan at r | HW300 -> r | HW400 -> r | HW600 -> r | paired diff (HW300->r) - (HW r) [CI] |'); P('|---|---|---|---|---|---|')
for r in [150,200]:
    h=R[11][(r,r)]['rt']; a=R[11][(300,r)]['rt']; df=pdiff(a['hits'],h['hits'])
    P(f"| {r} | {h['pos_recall']:.3f} | {a['pos_recall']:.3f} | {R[11][(400,r)]['rt']['pos_recall']:.3f} | {R[11][(600,r)]['rt']['pos_recall']:.3f} | {df[0]:+.3f} [{df[1]:+.3f},{df[2]:+.3f}] |")
P(); P('### E5. Source dependence at each r (pos recall, round-trip): 300 / 400 / 600 / 1200 hardware sources')
P('| r | from 300 | from 400 | from 600 | from 1200 |'); P('|---|---|---|---|---|')
for r in RATES:
    P(f"| {r} | "+' | '.join(f"{R[11][(s,r)]['rt']['pos_recall']:.3f}" if (s,r) in R[11] else '-' for s in [300,400,600,1200])+' |')
# corpus
c=json.load(open('corpus_results.json')); ce=json.load(open('corpus_ext_results.json'))
P(); P('### C1. Corpus (300 dpi native pages; reference = native OCR, psm 3, conf>=90, multiset recall; round-trip). Pages with <20 ref words excluded from aggregates')
good=[o for o in c if o['nref']>=20]; P(f"Pages: {len(c)} at 300 dpi, {len(good)} used (ref words total {sum(o['nref'] for o in good)}). Median word-box height at 300 dpi: {np.median([o['median_word_h_px'] for o in good]):.0f} px.")
P('| r | median | p10 | mean | min | pages <0.90 | direct-OCR median | SSIM median (page, gray) |'); P('|---|---|---|---|---|---|---|---|')
def agg(o,r,k): return [x[k] for x in o['rows'] if x['r']==r][0]
for r in [50,60,75,90]+RATES:
    if r in (50,60,75,90): v=np.array([agg(o,r,'rt') for o in ce if o['nref']>=20]); P(f"| {r} | {np.median(v):.3f} | {np.percentile(v,10):.3f} | {v.mean():.3f} | {v.min():.2f} | {(v<.9).sum()} | - | - |")
    else:
        v=np.array([agg(o,r,'rt') for o in good]); dv=np.array([agg(o,r,'direct') for o in good]); sv=np.array([agg(o,r,'ssim') for o in good])
        P(f"| {r} | {np.median(v):.3f} | {np.percentile(v,10):.3f} | {v.mean():.3f} | {v.min():.2f} | {(v<.9).sum()} | {np.median(dv):.3f} | {np.median(sv):.3f} |")
P(); P('Per-page recall (round-trip): file, nref, r=100,120,133,150,175,200,250 ')
for o in c: P('- '+o['file']+f" n={o['nref']}: "+', '.join(f"{x['r']}:{x['rt']:.2f}" for x in o['rows'] if x['r']<300))
# jpeg
j=json.load(open('jpeg_results.json')); base={x['r']:x for x in j if x['cfg']=='lossless'}
P(); P('### J1. JPEG input quality (Epson 300 dpi lossless -> JPEG -> downsample -> bilinear up -> OCR psm 11). pos recall; paired diff vs lossless [95% CI]; gray PSNR vs lossless at same r')
P('| config | file bytes | '+' | '.join(f'r={r}: pos / diff [CI] / PSNR' for r in [133,150,200,300])+' |'); P('|---|---|---|---|---|---|')
for cfg in ['lossless','q75_420','q85_420','q92_420','q100_420','q75_444','q92_444']:
    row=[x for x in j if x['cfg']==cfg]; cells=[]
    for r in [133,150,200,300]:
        x=[y for y in row if y['r']==r][0]; df=pdiff(x['hits'],base[r]['hits']); ps='inf' if x['psnr_gray']==float('inf') else f"{x['psnr_gray']:.1f}"
        cells.append(f"{x['pos_recall']:.3f} / {df[0]:+.3f} [{df[1]:+.3f},{df[2]:+.3f}] / {ps}")
    P(f"| {cfg} | {row[0]['bytes'] or 'PNG'} | "+' | '.join(cells)+' |')
# timing
P(); P('### T1. Downsample timing (cv2.INTER_AREA, single thread, median of 7)')
cv2.setNumThreads(1)
def tm(img,sd,r):
    ts=[]
    for _ in range(7): t=time.perf_counter(); downsample(img,sd,r); ts.append((time.perf_counter()-t)*1e3)
    return np.median(ts)
g=cv2.imread('work/gray_600.png',0); P(f'| 600 dpi gray 7016x5096 -> 150 | {tm(g,600,150):.1f} ms |')
g3=cv2.imread(f'{CORPUS}/pair02_front.jpg'); P(f'| 300 dpi color 2544x3295 -> 150 | {tm(g3,300,150):.1f} ms |'); P(f'| 300 dpi color 2544x3295 -> 133 | {tm(g3,300,133):.1f} ms |')
g12=cv2.imread('work/gray_1200.png',0); P(f'| 1200 dpi gray 14031x10200 -> 150 | {tm(g12,1200,150):.1f} ms |')
open('work/tables.md','w').write('\n'.join(out)); print('\n'.join(out))
