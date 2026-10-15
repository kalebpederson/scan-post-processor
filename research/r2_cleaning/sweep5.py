import sys, pickle; sys.path.insert(0,'.')
from run_eval import *
cfgs=[]
base=dict(gate="hyst",t_seed=0.25,comp_min_in2=0.5,t_hi_unsup=0.12)
def H(name,**kw): cfgs.append((name,"clean",{**base,**kw}))
for tl in (0.015,0.02,0.03):
    for wd in (0.10,0.14,0.18):
        H(f"H5_tl{tl}_wd{wd}_fg2.5",t_lo=tl,t_hi=tl+wd,faint_gain=2.5)
for fg in (2.0,3.0):
    for wd in (0.12,0.16): H(f"H5_tl0.02_wd{wd}_fg{fg}",t_lo=0.02,t_hi=0.02+wd,faint_gain=fg)
for sw in (1.0,2.0): H(f"HS5_tl0.02_wd0.14_fg2.5_sw{sw}",t_lo=0.02,t_hi=0.16,faint_gain=2.5,sharp_w=sw)
for tu in (0.09,0.16): H(f"H5_tl0.02_wd0.14_fg2.5_tu{tu}",t_lo=0.02,t_hi=0.16,faint_gain=2.5,t_hi_unsup=tu)
H("H5_tl0.02_wd0.14_fg2.5_g1.15",t_lo=0.02,t_hi=0.16,faint_gain=2.5,gamma=1.15)
if __name__=="__main__":
    agg,res=evaluate(cfgs,quick=True)
    pickle.dump((cfgs,agg),open('work/sweep5.pkl','wb')); print('done',len(cfgs))
