import sys, pickle; sys.path.insert(0,'.')
from run_eval import *
cfgs=[("B0","b0",{}),("DEFAULT","clean",{})]
for fg in (1.5,2.0):
    for fh in (0.5,0.3): cfgs.append((f"fg{fg}_fh{fh}","clean",dict(faint_gain=fg,faint_hi=fh)))
cfgs.append(("fg2.5_fh0.3","clean",dict(faint_gain=2.5,faint_hi=0.3)))
cfgs.append(("fg2.5_fh0.3_lo0.08","clean",dict(faint_gain=2.5,faint_hi=0.3,faint_lo=0.08)))
cfgs.append(("fg3_fh0.25_lo0.05","clean",dict(faint_gain=3.0,faint_hi=0.25,faint_lo=0.05)))
cfgs.append(("fg1.0","clean",dict(faint_gain=1.0)))
if __name__=="__main__":
    agg,res=evaluate(cfgs,quick=True)
    pickle.dump((cfgs,agg),open('work/sweep9.pkl','wb')); print('done',len(cfgs))
