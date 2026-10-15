import pickle,sys
for f in sys.argv[1:]:
    cfgs,agg=pickle.load(open(f,'rb'))
    print('name | pos_ret grid pencil light | outdE/natdE | neg | pairpool | white bk | ink_med | hue chr')
    for c in cfgs:
        a=agg[c[0]]; p=a['pos_by_class']
        print(c[0],'|',round(a['pos_ret'],3),round(p['grid'],2),round(p['pencil'],2),round(p['light_print'],2),'|',round(a['pos_out_dE'],1),round(a['pos_nat_dE'],1),'|',round(a['neg_surv_regions'],3),'|',round(a['bleed_surv_pooled'],3),'|',round(a['white_whitepaper'],4),round(a['bk_clean'],3),'|',a['ink_med'],'|',round(a['hue_shift_mean'],1),round(a['chroma_ratio_med'],2))
