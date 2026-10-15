"""Aggregate scratch/synth_raw.json -> research/r6_fold/synth_results.json (+ printed tables)."""
import json, numpy as np, collections
from common import SCR
R = [r for r in json.load(open(SCR + '/synth_raw.json'))]
errs = [r for r in R if 'err' in r]
R = [r for r in R if 'err' not in r and 'skip' not in r]
VS = ['ref', 'v0', 'vstroke', 'vdefault', 'vtd55', 'vnoprotchroma', 'vbw12', 'vc'] + (['vocr60', 'vocr30'] if any('ret_vocr60' in r for r in R) else [])
def agg(rows, tag):
    if not rows:
        return None
    o = dict(n=len(rows))
    for v in VS:
        o['ret_' + v] = float(np.mean([r['ret_' + v] for r in rows]))
    for v in VS[1:]:
        loss = np.array([r['ret_ref'] - r['ret_' + v] for r in rows])
        o['loss_mean_' + v] = float(loss.mean()); o['loss_p95_' + v] = float(np.percentile(loss, 95)); o['loss_max_' + v] = float(loss.max())
    for thr in ('10', '40'):
        for v in VS:
            o[f'surv{thr}_{v}'] = float(np.sum([r[f'surv{thr}_{v}'] for r in rows]) / max(1, np.sum([r['foot' + thr] for r in rows])))
    for v in VS[1:]:
        o['extra_px_' + v] = float(np.mean([r['extra_' + v] for r in rows]))
    return o
out = {'n_errors': len(errs), 'groups': {}}
for mode in ('ink', 'prior', 'e2e'):
    for cls in ('all', 'text', 'colour', 'faint'):
        rows = [r for r in R if r['mode'] == mode and (cls == 'all' or r['cls'] == cls)]
        a = agg(rows, mode + '/' + cls)
        if a:
            out['groups'][mode + '/' + cls] = a
e2e = [r for r in R if r['mode'] == 'e2e']
out['e2e_detect'] = dict(n=len(e2e), detected=sum(1 for r in e2e if r['det']), by_strip={s: [sum(1 for r in e2e if r['strip'] == s and r['det']), sum(1 for r in e2e if r['strip'] == s)] for s in ('p4', 'p6', 'p14', 'p13')},
                         by_class={c: [sum(1 for r in e2e if r['cls'] == c and r['det']), sum(1 for r in e2e if r['cls'] == c)] for c in ('text', 'colour', 'faint')})
out['timing_s'] = dict(detect_mean=float(np.mean([r['t_det'] for r in e2e])), remove_mean=float(np.mean([r['t_rm'] for r in R])))
json.dump(out, open('synth_results.json', 'w'), indent=1)
json.dump([{k: v for k, v in r.items() if k not in ('pdf',)} for r in R], open('synth_instances.json', 'w'))
for k, v in out['groups'].items():
    print(k, 'n=%d' % v['n'])
    for name in VS:
        print('   %-12s ret %.4f' % (name, v['ret_' + name]) + ('' if name in ('ref',) else '  loss mean %.4f p95 %.4f max %.3f' % (v['loss_mean_' + name], v['loss_p95_' + name], v['loss_max_' + name])) + '   surv10 %.3f surv40 %.3f' % (v['surv10_' + name], v['surv40_' + name]) + ('' if name == 'ref' else '  extra px %.0f' % v['extra_px_' + name]))
print(out['e2e_detect']); print(out['timing_s'])
