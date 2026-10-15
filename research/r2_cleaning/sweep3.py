import sys, pickle; sys.path.insert(0,'.')
from run_eval import *
cfgs=[("B0","b0",{})]
for win in (0.15,0.35,0.7):
    cfgs.append((f"M1_win{win}_w0.86","clean",dict(gate="m1",bg_win_in=win,wp=0.86,bp=0.12)))
for wp in (0.80,0.90,0.95): cfgs.append((f"M1_win0.35_w{wp}","clean",dict(gate="m1",wp=wp,bp=0.12)))
for g in (1.0,1.5,2.0):
    for tl in (0.02,0.03,0.04,0.06,0.08,0.11,0.14):
        cfgs.append((f"M2_tl{tl}_g{g}","clean",dict(gate="none",t_lo=tl,t_hi=tl+0.06,gamma=g)))
for w in (2.0,4.0,8.0):
    for tl in (0.04,0.06,0.08,0.10,0.12):
        cfgs.append((f"M3s_tl{tl}_w{w}_g1.5","clean",dict(gate="sharp",t_lo=tl,t_hi=tl+0.06,sharp_w=w,gamma=1.5)))
for tl in (0.02,0.04):
    for tu in (0.08,0.12,0.16):
        cfgs.append((f"M2h_tl{tl}_tu{tu}_g1.5","clean",dict(gate="hyst",t_lo=tl,t_hi=tl+0.06,t_seed=0.25,t_hi_unsup=tu,comp_min_in2=0.5,gamma=1.5)))
if __name__=="__main__":
    agg,res=evaluate(cfgs,quick=True)
    pickle.dump((cfgs,agg),open('work/sweep3.pkl','wb')); print('done',len(cfgs))
