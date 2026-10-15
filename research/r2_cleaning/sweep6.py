import sys, pickle; sys.path.insert(0,'.')
from run_eval import *
cfgs=[]
base=dict(gate="hyst",t_seed=0.25,comp_min_in2=0.5,t_hi_unsup=0.12,faint_gain=2.5)
def H(name,**kw): cfgs.append((name,"clean",{**base,**kw}))
for sw in (0.5,1.0):
    for wd in (0.10,0.12): H(f"HS6_tl0.02_wd{wd}_sw{sw}",t_lo=0.02,t_hi=0.02+wd,sharp_w=sw)
H("HS6_tl0.025_wd0.10_sw1.0",t_lo=0.025,t_hi=0.125,sharp_w=1.0)
H("HS6_tl0.02_wd0.10_sw1.0_fg3",t_lo=0.02,t_hi=0.12,sharp_w=1.0,faint_gain=3.0)
H("HS6_tl0.02_wd0.10_sw1.0_sig2",t_lo=0.02,t_hi=0.12,sharp_w=1.0,sharp_sigma_in=0.0066)
if __name__=="__main__":
    agg,res=evaluate(cfgs,quick=True)
    pickle.dump((cfgs,agg),open('work/sweep6.pkl','wb')); print('done',len(cfgs))
