"""Corpus sweep (stage 1): fold.line_candidates on every page of the R4 100-ppi cache, edge-cleaned like the pipeline.
Writes scratch/sweep3.json (per page: page size in inches, candidate dicts)."""
import json, sys, time
from multiprocessing import Pool
from common import *
import fold
from scanpp import geometry
P_OVERRIDE = None


def work(i):
    try:
        rgb, r = corpus_page(i)
        img = geometry._clean_edges_g(rgb.copy(), 100.0, inplace=True)[0]
        cs, (wi, hi) = fold.line_candidates(img, 100.0, P_OVERRIDE or fold.FoldParams())
        for c in cs:
            c.pop('rid', None)
        return i, dict(wi=wi, hi=hi, c=cs)
    except Exception as e:
        return i, dict(err=repr(e))
def work_kw(args):
    """(page index, FoldParams overrides) -> same as work(); the parameters travel with the job (worker processes are spawned, not forked)."""
    global P_OVERRIDE
    i, kw = args
    P_OVERRIDE = fold.FoldParams(**kw)
    return work(i)


if __name__ == '__main__':
    with Pool(14) as p:
        res = dict(p.map(work, range(len(index())), chunksize=8))
    json.dump(res, open(SCR + '/sweep3.json', 'w'))
    print(len(res), sum('err' in v for v in res.values()))
