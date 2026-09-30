"""Evaluate saved real local-model advice, never call a model or retune lambda."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import asdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import tracemalloc

from llmastar.adaptive import AdaptiveLLMAStar, GridGraph, LambdaConfig, fixed_config, validate_path
from .legacy import replay, sanitize
from .run import ROOT, oracle

METRICS = ('operation', 'storage', 'length')
CONDITIONS = ('actual', 'reverse', 'drop_half', 'corrupt50', 'none')


def stress(points, query, graph, case, condition):
    if condition == 'actual':
        return points
    if condition == 'reverse':
        return points[::-1]
    if condition == 'drop_half':
        return points[::2]
    if condition == 'none':
        return []
    rng = random.Random(f"local-20261001:{case['domain']}:{case['map_id']}:{case['sample_id']}")
    copy = list(points)
    free = [n for n in graph.states() if n not in (tuple(query['start']), tuple(query['goal']))]
    for i in rng.sample(range(len(copy)), math.ceil(len(copy)/2)):
        copy[i] = rng.choice(free)
    return sanitize(query, copy)


def paired(rows, baseline, scenario, domain):
    subset = [r for r in rows if r['scenario'] == scenario and r['domain'] == domain]
    base = {(r['map_id'], r['sample_id']): r for r in subset if r['method'] == baseline}
    return [(r, base[(r['map_id'], r['sample_id'])]) for r in subset
            if r['method'] == 'adaptive' and r['success'] and base[(r['map_id'], r['sample_id'])]['success']]


def summarize(rows):
    out = []
    for domain, scenario in sorted({(r['domain'], r['scenario']) for r in rows}):
        for baseline in ('legacy', 'fixed_1_controlled', 'fixed_2_controlled'):
            pairs = paired(rows, baseline, scenario, domain)
            ratios = {k: [a[k]/b[k] for a,b in pairs] for k in METRICS}
            out.append({'domain': domain, 'scenario': scenario, 'baseline': baseline,
                        'pairs': len(pairs),
                        'ratios': {k: math.exp(statistics.mean(math.log(v) for v in vs)) if vs else None
                                   for k,vs in ratios.items()},
                        'operation_p95_ratio': sorted(ratios['operation'])[math.ceil(.95*len(pairs))-1] if pairs else None,
                        'strict_three_improvements': sum(all(a[k] < b[k]-1e-9 for k in METRICS) for a,b in pairs),
                        'weak_pareto_improvements': sum(all(a[k] <= b[k]+1e-9 for k in METRICS)
                            and any(a[k] < b[k]-1e-9 for k in METRICS) for a,b in pairs)})
    return out


def cluster_intervals(rows, baseline, scenario='actual', domain='original'):
    pairs = paired(rows, baseline, scenario, domain)
    groups = defaultdict(lambda: defaultdict(list))
    for a,b in pairs:
        for k in METRICS:
            groups[a['map_id']][k].append(math.log(a[k]/b[k]))
    means = {i: {k: statistics.mean(v) for k,v in g.items()} for i,g in groups.items()}
    if not means:
        return {}
    rng = random.Random(64001)
    ids = sorted(means)
    draws = {k: [] for k in METRICS}
    for _ in range(2000):
        sampled = rng.choices(ids, k=len(ids))
        for k in METRICS:
            draws[k].append(math.exp(statistics.mean(means[i][k] for i in sampled)))
    return {k: [sorted(v)[49], sorted(v)[1949]] for k,v in draws.items()}


def evaluate(records, output):
    selected = json.loads((ROOT/'benchmarks/selected_config.json').read_text())
    methods = {'astar': fixed_config(0, reopen=True),
               **{k: LambdaConfig(**selected[k]) for k in ['fixed_1_controlled', 'fixed_2_controlled', 'adaptive']}}
    rows, traces, memory, quality = [], [], [], []
    for case in records:
        query = case['query']
        graph = GridGraph(query)
        for n in graph.states():
            graph.neighbors(n)
        cost, _ = oracle(graph, tuple(query['start']), tuple(query['goal']))
        if cost is None:
            raise ValueError('An evaluation query is unreachable')
        points = sanitize(query, case['waypoints'])
        quality.append({'map_id': case['map_id'], 'sample_id': case['sample_id'], 'domain': case['domain'],
                        'parse_status': case['parse_status'], 'raw_points': len(case['waypoints']),
                        'retained_points': len(points), 'empty_after_filter': not points,
                        'raw_endpoints_match': bool(case['waypoints']) and
                            case['waypoints'][0] == query['start'] and case['waypoints'][-1] == query['goal'],
                        'no_intermediate_advice': not any(tuple(p) not in
                            (tuple(query['start']), tuple(query['goal'])) for p in points),
                        'truncated': case['generation']['done_reason'] == 'length'})
        conditions = CONDITIONS if case['domain'] == 'original' else ('actual',)
        for condition in conditions:
            advice = stress(points, query, graph, case, condition)
            results = {'legacy': replay(query, advice)}
            results.update({name: AdaptiveLLMAStar(cfg).searching(query, advice, graph=graph).to_dict()
                            for name,cfg in methods.items()})
            for name,result in results.items():
                if result['success']:
                    assert validate_path(graph, result['path'], tuple(query['start']), tuple(query['goal']))
                    assert result['length'] + 1e-8 >= cost
                    if name == 'astar':
                        assert abs(result['length'] - cost) < 1e-8
                key = {k: case[k] for k in ['map_id', 'sample_id', 'domain']}
                rows.append({**key, 'scenario': condition, 'method': name, 'optimal': cost,
                             **{k: result.get(k) for k in ['success', 'operation', 'storage', 'length',
                                   'runtime_seconds', 'peak_open', 'peak_queue_entries', 'reopened',
                                   'queue_rebuilds', 'score_evaluations']}})
                traces.append({**key, 'scenario': condition, 'method': name,
                               'path': result['path'], 'lambda_history': result.get('lambda_history', [])})
        # Python planner allocations, excluding the model and shared graph cache.
        for name in ['fixed_1_controlled', 'adaptive']:
            tracemalloc.start()
            result = AdaptiveLLMAStar(methods[name]).searching(query, points, graph=graph)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            memory.append({**{k: case[k] for k in ['map_id','sample_id','domain']},
                           'method': name, 'peak_python_bytes': peak, 'success': result.success})
    output.mkdir(parents=True, exist_ok=True)
    with (output/'raw.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (output/'summary.json').write_text(json.dumps(summarize(rows), indent=2))
    (output/'quality.json').write_text(json.dumps(quality, indent=2))
    (output/'allocations.json').write_text(json.dumps(memory, indent=2))
    ci = {b: cluster_intervals(rows, b) for b in ['legacy','fixed_1_controlled']}
    (output/'bootstrap.json').write_text(json.dumps(ci, indent=2))
    unpacked = json.dumps(traces).encode()
    packed = gzip.compress(unpacked, mtime=0)
    (output/'traces.json.gz').write_bytes(packed)
    meta = {'cases': len(records), 'rows': len(rows), 'configs': {k: asdict(v) for k,v in methods.items()},
            'advice_source': 'Actual local model output; errors retained as empty advice; stress variants are synthetic modifications',
            'case_identifiers': [{k:r[k] for k in ['map_id','sample_id','domain']} for r in records],
            'model': records[0]['provenance'],
            'memory': 'tracemalloc Python planner only; shared graph/native/model memory excluded',
            'runtime': 'Planner-only, graph construction, model generation, memory tracing and oracle excluded',
            'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [ROOT/'benchmarks/local_evaluate.py',ROOT/'llmastar/adaptive.py',ROOT/'benchmarks/legacy.py']},
            'traces_uncompressed_sha256': hashlib.sha256(unpacked).hexdigest(),
            'traces_compressed_sha256': hashlib.sha256(packed).hexdigest(),
            'api_cost_usd': 0}
    (output/'metadata.json').write_text(json.dumps(meta, indent=2))
    print(f'Evaluated {len(records)} actual responses, {len(rows)} searches; results: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = [r for p in args.inputs for r in json.loads(p.read_text())]
    keys = [(r['map_id'], r['sample_id'], r['domain']) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate cases are not allowed')
    if len({r['provenance']['digest'] for r in records}) != 1:
        raise ValueError('Evaluate models separately')
    evaluate(records, args.output)


if __name__ == '__main__':
    main()
