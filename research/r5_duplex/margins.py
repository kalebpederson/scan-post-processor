import json, numpy as np
R = json.load(open('work/fine.json'))
G = dict(g_two=0.05, agree=0.12, ncc_two=0.20, g_one=0.06, ncc_one=0.50, cmin=400)
def srcc(r): return r['dB']['chroma_px'] if r['best'] == 'a' else r['dA']['chroma_px']
def crit(r):
    two_parts = dict(g=min(r['ga'], r['gb']) >= G['g_two'], agree=r.get('agree_in') is not None and r['agree_in'] <= G['agree'], ncc=r['ncc'] >= G['ncc_two'])
    one_parts = dict(g=max(r['ga'], r['gb']) >= G['g_one'], ncc=r['ncc'] >= G['ncc_one'])
    return two_parts, one_parts, srcc(r) >= G['cmin']
neg = [r for r in R if r['kind'] in ('far', 'cross') and r.get('ncc') is not None]
TRUE_NEG_EXCLUDE = {(1937, 1939), (521, 404), (404, 521)}   # verified by eye to be real sheets (non-adjacent pages / the same sheet in two PDFs)
neg = [r for r in neg if (r['i'], r['j']) not in TRUE_NEG_EXCLUDE]
print('negatives reaching fine stage (coarse gmax>=.05):', len(neg))
def passes(r, drop=None):
    t, o, c = crit(r)
    t2 = {k: v for k, v in t.items() if k != drop}; o2 = {k: v for k, v in o.items() if k != drop}
    return (all(t2.values()) or all(o2.values())) and (c or drop == 'chroma')
print('accepted:', sum(passes(r) for r in neg))
for drop in ('g', 'agree', 'ncc', 'chroma'):
    print('drop criterion', drop, '->', sum(passes(r, drop) for r in neg), 'negatives would pass')
# nearest misses: max over negatives of each statistic given all OTHER criteria pass
for name, f in (('min gap (two-way)', lambda r: min(r['ga'], r['gb'])), ('ncc', lambda r: r['ncc']), ('max gap', lambda r: max(r['ga'], r['gb'])), ('src chroma px', srcc)):
    pass
print('negatives with chroma ok: ', sum(crit(r)[2] for r in neg), ' of which pass two-way gap&agree:', sum(crit(r)[2] and crit(r)[0]['g'] and crit(r)[0]['agree'] for r in neg), ' pass ncc>=.2 too:', sum(crit(r)[2] and all(crit(r)[0].values()) for r in neg))
best = sorted([r for r in neg if crit(r)[2]], key=lambda r: -(min(r['ga'], r['gb'])))[:5]
for r in best: print('  top neg by min gap with chroma ok: min gap %.3f max gap %.3f agree %s ncc %.2f srcchroma %d' % (min(r['ga'], r['gb']), max(r['ga'], r['gb']), r.get('agree_in'), r['ncc'], srcc(r)))
best = sorted([r for r in neg if crit(r)[2]], key=lambda r: -r['ncc'])[:5]
for r in best: print('  top neg by ncc with chroma ok: ncc %.2f min gap %.3f max gap %.3f agree %s' % (r['ncc'], min(r['ga'], r['gb']), max(r['ga'], r['gb']), r.get('agree_in')))
# positives (adjacent): same stats
adj = [r for r in R if r['kind'].startswith('adj') and r.get('ncc') is not None]
acc = [r for r in adj if passes(r)]
print('adjacent pairs reaching fine stage', len(adj), 'accepted', len(acc))
for nm, f in (('min gap', lambda r: min(r['ga'], r['gb'])), ('ncc', lambda r: r['ncc']), ('src chroma px', srcc)):
    v = np.array([f(r) for r in acc]); print('  accepted adjacent', nm, 'pct 0/10/50/90', np.percentile(v, [0, 10, 50, 90]).round(3))
