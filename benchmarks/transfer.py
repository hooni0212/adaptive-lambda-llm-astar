"""Transfer and Python-allocation checks using frozen selected configuration.

All guidance is synthetic; these are supplementary tests, not paper reproduction.
"""
import csv
from dataclasses import replace
import json
from pathlib import Path
import random
import statistics
import time
import tracemalloc

from llmastar.adaptive import AdaptiveLLMAStar,GridGraph,LambdaConfig,validate_path
from .run import DATA,ROOT,oracle,split_ids,synthetic


class WeightedGraph(GridGraph):
    def neighbors(self,n):
        # Costly middle band. Euclidean distance remains an admissible lower bound.
        return tuple((nxt,c*(5 if 15<=nxt[0]<=30 else 1)) for nxt,c in super().neighbors(n))


def transform(m,kind):
    if kind=='scale2':
        return {**m,'range_x':[0,(m['range_x'][1]-1)*2+1],
                'range_y':[0,(m['range_y'][1]-1)*2+1],
                'horizontal_barriers':[[v*2 for v in b] for b in m['horizontal_barriers']],
                'vertical_barriers':[[v*2 for v in b] for b in m['vertical_barriers']],
                'start_goal':[[[v*2 for v in p] for p in sg[:2]] for sg in m['start_goal']]}
    if kind=='transpose':
        return {**m,'range_x':m['range_y'][:],'range_y':m['range_x'][:],
                'horizontal_barriers':[b[:] for b in m['vertical_barriers']],
                'vertical_barriers':[b[:] for b in m['horizontal_barriers']],
                'start_goal':[[[p[1],p[0]] for p in sg[:2]] for sg in m['start_goal']]}
    return m


def main():
    selected=json.loads((ROOT/'benchmarks/selected_config.json').read_text())
    methods={n:LambdaConfig(**selected[n]) for n in ['adaptive','fixed_1_controlled','fixed_2_controlled']}
    ids=split_ids()['validation'];data=[m for m in json.loads(DATA.read_text()) if m['id'] in ids]
    rows=[]
    for kind in ['transpose','scale2','weighted']:
        for raw in data:
            m=transform(raw,kind);graph=(WeightedGraph if kind=='weighted' else GridGraph)(m)
            for n in graph.states():graph.neighbors(n)
            for sample_id,sg in enumerate(m['start_goal'][:3]):
                q={**m,'start':sg[0],'goal':sg[1]}
                optimal,path=oracle(graph,tuple(sg[0]),tuple(sg[1]))
                if optimal is None:raise AssertionError('unreachable transformed problem')
                for scenario in ['clean','corrupt50','corrupt100']:
                    wp=synthetic(q,path,graph,m['id'],sample_id,scenario)
                    for name,cfg in methods.items():
                        r=AdaptiveLLMAStar(cfg).searching(q,wp,graph=graph)
                        if not r.success or not validate_path(graph,r.path,tuple(sg[0]),tuple(sg[1])):
                            raise AssertionError('invalid transfer result')
                        if r.length+1e-9<optimal:raise AssertionError('cost below optimal reference')
                        rows.append({'map_id':m['id'],'sample_id':sample_id,'domain':kind,'scenario':scenario,
                                     'method':name,'operation':r.operation,'storage':r.storage,'cost':r.length,
                                     'optimal':optimal,'runtime_seconds':r.runtime_seconds})
        print(f'{kind}: complete',flush=True)
    out=ROOT/'results/transfer';out.mkdir(parents=True,exist_ok=True)
    with (out/'raw.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # Planner-only allocations, warmed graph. Does not include shared graph memory,
    # NumPy/native allocations or model memory; tracemalloc slows execution.
    allocations=[]
    for m in data[:10]:
        graph=GridGraph(m)
        for n in graph.states():graph.neighbors(n)
        for sample_id,sg in enumerate(m['start_goal'][:2]):
            q={**m,'start':sg[0],'goal':sg[1]};_,path=oracle(graph,tuple(sg[0]),tuple(sg[1]))
            for scenario in ['clean','corrupt100']:
                wp=synthetic(q,path,graph,m['id'],sample_id,scenario)
                for name,cfg in methods.items():
                    tracemalloc.start()
                    r=AdaptiveLLMAStar(cfg).searching(q,wp,graph=graph)
                    _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
                    allocations.append({'map_id':m['id'],'sample_id':sample_id,'scenario':scenario,
                                        'method':name,'peak_python_bytes':peak})
    (out/'allocations.json').write_text(json.dumps(allocations,indent=2))
    (out/'metadata.json').write_text(json.dumps({'configuration':'benchmarks/selected_config.json',
                  'split':'validation','guidance':'synthetic oracle/corruption; NOT LLM',
                  'transfer':'transpose, scale2, destination-dependent edge costs',
                  'samples_per_domain':180,'allocation_samples_per_method':40,
                  'allocations':'tracemalloc planner-only Python bytes, warmed shared graph excluded'},indent=2))


if __name__=='__main__':main()
