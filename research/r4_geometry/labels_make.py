import json,numpy as np
idx=json.load(open('index.json'));ink=json.load(open('ink_raw.json'));S=json.load(open('label_set.json'))
fc=np.array([x['f0.1'] for x in ink])
notblank={867}|{i for i in S if fc[i]*1e4>=20}
lab={}
reasons={867:'purple marker "1A" at top-left (separator/tracking label)',
 894:'large handwritten marker label "A1" only (separator sheet)',1044:'large handwritten marker label "A3" only',
 1356:'large handwritten label "B7" only',1581:'large handwritten label "B8" only',1220:'large handwritten label "B5" only',
 891:'blank: faint smudge, 1 dot, thin vertical dust streak, dark edge strips',
 1710:'blank: bleed-through scribble only (blurred, low contrast)',
 905:'blank: row of dust specks',925:'blank: row of dust specks + dark strip at right edge',
 875:'blank: dark strip along top edge + dust',879:'blank: dark strip along top edge + dust',883:'blank: top edge strip + dust',871:'blank: top edge strip + dust',
 99:'content (green paper, boxed answer grid with numbers)',1925:'content (handwritten math)',}
for i in S:
    lab[str(i)]=dict(label='NOT_BLANK' if i in notblank else 'BLANK',
        page=idx[i]['pdf'].split('samples/')[-1]+f" p{idx[i]['page']+1}",
        reason=reasons.get(i,'content' if i in notblank else 'no mark / bleed-through / dust only'))
json.dump(dict(note='keys = index into research/r4_geometry/index.json (global page id). Pages not in this dict (1643 pages with core ink fraction >1.5%) are all content = NOT_BLANK.',labels=lab),open('labels_blank.json','w'),indent=0)
print(sum(v['label']=='BLANK' for v in lab.values()),sum(v['label']=='NOT_BLANK' for v in lab.values()))
