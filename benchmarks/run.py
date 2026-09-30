"""python -m benchmarks.run --split validation --output results/validation

Offline stress tests use oracle-derived SYNTHETIC waypoints. They are not LLM
outputs and cannot reproduce the paper's numerical model results. Cached real
guidance can be evaluated using --waypoints, an array of records containing
map_id, sample_id, waypoints, and optional provenance. No API calls are made.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import heapq
import json
import math
from pathlib import Path
import random
import statistics
import time

from llmastar.adaptive import AdaptiveLLMAStar,GridGraph,LambdaConfig,fixed_config,validate_path
from .legacy import replay,sanitize

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'dataset/environment_50_30.json'


def split_ids():
    ids = list(range(100))
    random.Random(20261001).shuffle(ids)
    return {'train':set(ids[:20]),'validation':set(ids[20:40]),'test':set(ids[40:]),'all':set(ids)}


def configs():
    # Fix all settings before opening the test split.
    return {
        'astar':fixed_config(0,reopen=True),
        'fixed_1':fixed_config(1),
        'fixed_025':fixed_config(.25,reopen=True),
        'fixed_05':fixed_config(.5,reopen=True),
        'fixed_2':fixed_config(2,reopen=True),
        'adaptive':LambdaConfig(),
    }


def oracle(graph,start,goal):
    """Dijkstra provides independent optimal costs and synthetic clean guidance."""
    g={start:0.0};parent={};q=[(0.0,start)]
    while q:
        cost,n=heapq.heappop(q)
        if cost != g[n]: continue
        if n==goal:
            path=[goal]
            while path[-1]!=start: path.append(parent[path[-1]])
            return cost,list(reversed(path))
        for nxt,edge in graph.neighbors(n):
            new=cost+edge
            if new+1e-12 < g.get(nxt,math.inf):
                g[nxt]=new;parent[nxt]=n;heapq.heappush(q,(new,nxt))
    return None,[]


def synthetic(query, path, graph, map_id, sample_id, scenario):
    if scenario=='none': return []
    rng=random.Random(f'20261001:{map_id}:{sample_id}:{scenario}')
    if scenario=='straight':
        start,goal=query['start'],query['goal']
        points=[tuple(round(start[d]+(goal[d]-start[d])*t/5) for d in (0,1)) for t in (1,2,3,4)]
    else:
        points=[path[min(len(path)-2,max(1,round((len(path)-1)*t/5)))] for t in (1,2,3,4)] if len(path)>2 else []
        fraction={'clean':0,'corrupt25':.25,'corrupt50':.5,'corrupt100':1,'reverse':0}[scenario]
        free=[n for n in graph.states() if n not in (tuple(query['start']),tuple(query['goal']))]
        for i in rng.sample(range(len(points)),round(fraction*len(points))):
            points[i]=rng.choice(free)
        if scenario=='reverse': points.reverse()
    return sanitize(query,points)


def geomean(values):
    return math.exp(statistics.mean(math.log(v) for v in values)) if values else None


def summarize(rows):
    base={(r['map_id'],r['sample_id'],r['scenario']):r for r in rows if r['method']=='legacy'}
    summary=[]
    for scenario in sorted(set(r['scenario'] for r in rows)):
        for method in sorted(set(r['method'] for r in rows)):
            subset=[r for r in rows if r['scenario']==scenario and r['method']==method]
            pairs=[(r,base[(r['map_id'],r['sample_id'],r['scenario'])]) for r in subset if r['success'] and base[(r['map_id'],r['sample_id'],r['scenario'])]['success']]
            ratios={k:[r[k]/b[k] for r,b in pairs if b[k]>0] for k in ('operation','storage','length')}
            triple=sum(r['operation']<b['operation'] and r['storage']<b['storage'] and r['length']<b['length']-1e-9 for r,b in pairs)
            weak=sum(all(r[k]<=b[k]+1e-9 for k in ('operation','storage','length')) and any(r[k]<b[k]-1e-9 for k in ('operation','storage','length')) for r,b in pairs)
            costs=[r['length']/r['optimal'] for r in subset if r['success'] and r['optimal']>0]
            op_ratio=sorted(ratios['operation'])
            summary.append({'scenario':scenario,'method':method,'samples':len(subset),
                            'successes':sum(r['success'] for r in subset),'paired_successes':len(pairs),
                            **{k+'_ratio_vs_legacy':geomean(v) for k,v in ratios.items()},
                            'path_ratio_vs_optimal':geomean(costs),
                            'strict_three_improvements':triple,'pareto_improvements':weak,
                            'operation_p95_ratio':op_ratio[min(len(op_ratio)-1,math.ceil(.95*len(op_ratio))-1)] if op_ratio else None,
                            'mean_runtime_seconds':statistics.mean(r['runtime_seconds'] for r in subset) if subset else None})
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',choices=split_ids(),default='validation')
    parser.add_argument('--output',type=Path,default=ROOT/'results/validation')
    parser.add_argument('--scenarios',nargs='+',default=['clean','corrupt25','corrupt50','corrupt100','reverse','straight','none'])
    parser.add_argument('--max-maps',type=int)
    parser.add_argument('--waypoints',type=Path)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--methods',nargs='+')
    parser.add_argument('--save-paths',action='store_true')
    args=parser.parse_args()
    methods=configs()
    if args.config:
        extra=json.loads(args.config.read_text())
        methods.update({name:LambdaConfig(**values) for name,values in extra.items()})
    if args.methods: methods={k:v for k,v in methods.items() if k in args.methods}
    cache=None
    if args.waypoints:
        cache={(r['map_id'],r['sample_id']):r for r in json.loads(args.waypoints.read_text())}
    data=json.loads(DATA.read_text());ids=split_ids()[args.split]
    selected=[m for m in data if m['id'] in ids]
    if cache is not None:
        selected=[m for m in selected if any(k[0]==m['id'] for k in cache)]
    if args.max_maps is not None:selected=selected[:args.max_maps]
    args.output.mkdir(parents=True,exist_ok=True)
    metadata={'split':args.split,'map_ids':[m['id'] for m in selected],'seed':20261001,
              'guidance':'cached_external' if cache is not None else 'synthetic_oracle_and_corruption_NOT_LLM',
              'dataset_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),
              'waypoint_sha256':hashlib.sha256(args.waypoints.read_bytes()).hexdigest() if cache is not None else None,
              'configs':{k:asdict(v) for k,v in methods.items()},
              'geometry':'closed barrier segments, 8-neighbor Euclidean edge cost',
              'runtime':'planner-only; shared graph cache and oracle preparation excluded; legacy collision checks not cached',
              'storage':'legacy=len(g), including infinity entries; new=finite discovered states; see fixed_1 for controlled comparison'}
    metadata['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [ROOT/'llmastar/adaptive.py',ROOT/'benchmarks/run.py',ROOT/'benchmarks/legacy.py',ROOT/'llmastar/pather/llm_a_star/llm_a_star.py']}
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2))
    rows=[];traces=[];waypoint_cache=[];start_time=time.perf_counter()
    for m in selected:
        graph=GridGraph(m)
        for n in graph.states(): graph.neighbors(n)
        for sample_id,sg in enumerate(m['start_goal']):
            if cache is not None and (m['id'],sample_id) not in cache:
                continue
            query={**m,'start':sg[0],'goal':sg[1]}
            optimal,path=oracle(graph,tuple(sg[0]),tuple(sg[1]))
            if optimal is None: raise RuntimeError(f'Unreachable dataset sample {m["id"]}:{sample_id}')
            scenarios=['cached'] if cache is not None else args.scenarios
            for scenario in scenarios:
                if cache is not None:
                    if (m['id'],sample_id) not in cache:continue
                    wp=sanitize(query,cache[(m['id'],sample_id)]['waypoints'])
                else:wp=synthetic(query,path,graph,m['id'],sample_id,scenario)
                waypoint_cache.append({'map_id':m['id'],'sample_id':sample_id,'scenario':scenario,'waypoints':wp})
                results={'legacy':replay(query,wp)}
                for name,cfg in methods.items():
                    results[name]=AdaptiveLLMAStar(cfg).searching(query,wp,graph=graph).to_dict()
                for name,result in results.items():
                    if result['success']:
                        if not validate_path(graph,result['path'],tuple(sg[0]),tuple(sg[1])):
                            raise AssertionError(f'Invalid path {m["id"]}:{sample_id}:{scenario}:{name}')
                        if result['length']+1e-9<optimal:raise AssertionError('path shorter than Dijkstra reference')
                        if name=='astar' and abs(result['length']-optimal)>1e-8:raise AssertionError('A* disagrees with Dijkstra')
                    row={'map_id':m['id'],'sample_id':sample_id,'scenario':scenario,'method':name,'optimal':optimal,
                         **{k:result.get(k) for k in ['success','operation','storage','length','runtime_seconds','peak_open','peak_queue_entries','reopened','queue_rebuilds','score_evaluations','fallback_count','skipped_waypoints']}}
                    rows.append(row)
                    if args.save_paths:traces.append({'map_id':m['id'],'sample_id':sample_id,'scenario':scenario,'method':name,'path':result['path'],'lambda_history':result.get('lambda_history',[])})
        print(f'map {m["id"]} completed ({len(rows)} runs; {time.perf_counter()-start_time:.1f}s)',flush=True)
    with (args.output/'raw.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else [])
        writer.writeheader();writer.writerows(rows)
    summary=summarize(rows)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    (args.output/'waypoints.json').write_text(json.dumps(waypoint_cache))
    if args.save_paths:(args.output/'traces.json').write_text(json.dumps(traces))
    print(json.dumps([r for r in summary if r['method']=='adaptive'],indent=2))


if __name__=='__main__':main()
