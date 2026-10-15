import json,sys
d=json.load(open(sys.argv[1]));print(d['nref'],d['classes'])
rows=sorted(d['out'],key=lambda x:(x['src'],x['r']))
for k in ['rt','direct']:
  print(k,'src r ms pos bold mid light head cer')
  for x in rows:
    s=x[k]; print(x['src'],x['r'],*[round(s[m],2) for m in ['ms_recall','pos_recall','pos_bold','pos_mid','pos_light','pos_heading','cer']])
