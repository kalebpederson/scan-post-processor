import sys, pickle; sys.path.insert(0,'.')
from run_eval import *
cfgs=[("B0","b0",{})]
base=dict(gate="hyst",t_seed=0.25,comp_min_in2=0.5)
def H(name,**kw): cfgs.append((name,"clean",{**base,**kw}))
for tl in (0.01,0.02,0.03):
    for tu in (0.10,0.14):
        H(f"H_tl{tl}_tu{tu}",t_lo=tl,t_hi=tl+0.06,t_hi_unsup=tu)
for seed in (0.18,0.35): H(f"H_tl0.02_tu0.12_seed{seed}",t_lo=0.02,t_hi=0.08,t_hi_unsup=0.12,t_seed=seed)
H("H_tl0.02_tu0.12_nocomp",t_lo=0.02,t_hi=0.08,t_hi_unsup=0.12,comp_min_in2=0)
H("H_tl0.02_tu0.12_comp2",t_lo=0.02,t_hi=0.08,t_hi_unsup=0.12,comp_min_in2=2.0)
for sw in (2.0,4.0): H(f"HS_tl0.02_tu0.12_sw{sw}",t_lo=0.02,t_hi=0.08,t_hi_unsup=0.12,sharp_w=sw)
for fg in (1.5,2.0,2.5,3.0):
    for g in (1.0,1.2):
        H(f"H_tl0.02_tu0.12_fg{fg}_g{g}",t_lo=0.02,t_hi=0.08,t_hi_unsup=0.12,faint_gain=fg,gamma=g)
H("H_tl0.02_tu0.12_fg2_hi",t_lo=0.02,t_hi=0.06,t_hi_unsup=0.12,faint_gain=2.0)
H("H_tl0.02_tu0.12_fg2_wide",t_lo=0.02,t_hi=0.12,t_hi_unsup=0.12,faint_gain=2.0)
if __name__=="__main__":
    agg,res=evaluate(cfgs,quick=True)
    pickle.dump((cfgs,agg),open('work/sweep4.pkl','wb')); print('done',len(cfgs))
