import sys, pickle; sys.path.insert(0,'.')
from run_eval import *
cfgs=[]
base=dict(gate="hyst",t_seed=0.25,comp_min_in2=0.5,t_hi_unsup=0.12,faint_gain=2.5,line_support=True)
def H(name,**kw): cfgs.append((name,"clean",{**base,**kw}))
for tl in (0.02,0.03):
    for wd in (0.04,0.07,0.10):
        H(f"L7_tl{tl}_wd{wd}",t_lo=tl,t_hi=tl+wd)
H("L7_tl0.02_wd0.07_tline0.02",t_lo=0.02,t_hi=0.09,t_line=0.02)
H("L7_tl0.02_wd0.07_tline0.04",t_lo=0.02,t_hi=0.09,t_line=0.04)
H("L7_tl0.02_wd0.07_len0.25",t_lo=0.02,t_hi=0.09,line_len_in=0.25)
H("L7_tl0.02_wd0.07_len0.7",t_lo=0.02,t_hi=0.09,line_len_in=0.7)
H("L7S_tl0.02_wd0.07_sw1",t_lo=0.02,t_hi=0.09,sharp_w=1.0)
H("L7_tl0.02_wd0.07_nochromafade",t_lo=0.02,t_hi=0.09,faint_chroma_lo=5,faint_chroma_hi=6)
if __name__=="__main__":
    agg,res=evaluate(cfgs,quick=True)
    pickle.dump((cfgs,agg),open('work/sweep7.pkl','wb')); print('done',len(cfgs))
