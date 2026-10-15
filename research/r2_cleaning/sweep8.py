import sys, pickle; sys.path.insert(0,'.')
from run_eval import *
cfgs=[("DEFAULT","clean",{})]
for bp in (0.16,0.20,0.25): cfgs.append((f"bp{bp}","clean",dict(bp=bp)))
cfgs.append(("bp0.2_g1.1","clean",dict(bp=0.20,gamma=1.1)))
cfgs.append(("keep","clean",{},"keep"))
cfgs.append(("wd0.10","clean",dict(t_hi=0.12)))
cfgs.append(("wd0.10_sw1","clean",dict(t_hi=0.12,sharp_w=1.0)))
if __name__=="__main__":
    agg,res=evaluate(cfgs,quick=True)
    pickle.dump((cfgs,agg),open('work/sweep8.pkl','wb')); print('done',len(cfgs))
