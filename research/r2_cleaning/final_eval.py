import sys, pickle, json; sys.path.insert(0,'.')
from run_eval import *
cfgs=[("B0","b0",{}),("FINAL_flatten","clean",{}),("FINAL_keep","clean",{},"keep"),
      ("M1_flatfield_levels","clean",dict(gate="m1",bg_win_in=0.35,wp=0.90,bp=0.12)),
      ("M2_threshold_matched","clean",dict(gate="none",t_lo=0.08,t_hi=0.14,faint_gain=1.0,bp=0.12)),
      ("M2_threshold_retain","clean",dict(gate="none",t_lo=0.03,t_hi=0.09,faint_gain=1.0,bp=0.12))]
if __name__=="__main__":
    agg,res=evaluate(cfgs,quick=False)
    pickle.dump((cfgs,agg,res),open('work/final.pkl','wb')); print('done')
