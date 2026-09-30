"""Generate real waypoint advice with downloaded LOCAL Ollama models only.

No API key, remote provider, paid endpoint, model pull, or oracle path is used.
Run one model at a time. Immutable plans and append-only responses allow resume.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .run import ROOT, DATA, split_ids

ENDPOINT = 'http://127.0.0.1:11434'
OPTIONS = {'temperature': 0, 'seed': 20261001, 'num_ctx': 4096,
           'num_predict': 256, 'num_thread': 4}
SYSTEM_PROMPT = ('You are a path-planning assistant. Reply with only '
                 'Generated Path: [[x1, y1], [x2, y2], ...]. '
                 'Use integer coordinates, include the start and goal, and use at most 8 points. '
                 'Do not write code, explanations, or reasoning.')


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('Redirects are forbidden for local-only experiments')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def api(path, payload=None):
    if path not in ('/api/tags', '/api/show', '/api/version', '/api/generate'):
        raise ValueError('Only the documented local Ollama endpoints are allowed')
    body = json.dumps(payload).encode() if payload is not None else None
    request = Request(ENDPOINT + path, data=body,
                      headers={'Content-Type': 'application/json'})
    # Redirects to remote hosts must never turn a local run into a cloud call.
    with build_opener(NoRedirects()).open(request, timeout=180) as response:
        if not response.geturl().startswith(ENDPOINT + '/'):
            raise ValueError('Non-local response rejected')
        return json.load(response)


def local_model(name):
    if name not in ('qwen2.5:1.5b', 'llama3.2:3b'):
        raise ValueError('This experiment only permits the two named local models')
    tag = next((m for m in api('/api/tags')['models'] if m['name'] == name), None)
    if tag is None:
        raise ValueError(f'Download {name} locally before running the experiment')
    show = api('/api/show', {'model': name})
    if show.get('remote_host') or show.get('remote_model') or tag.get('remote_host'):
        raise ValueError('Cloud models are forbidden')
    return {'name': name, 'digest': tag['digest'], 'size': tag['size'],
            'details': show.get('details', tag.get('details')),
            'template_sha256': digest(show.get('template', '').encode()),
            'system_sha256': digest(show.get('system', '').encode()),
            'ollama_version': api('/api/version')['version']}


def prompt_template():
    source = (ROOT/'llmastar/pather/llm_a_star/prompt.py').read_text()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == 'standard_gpt' for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Upstream standard prompt is missing')


def make_prompt(query):
    text = prompt_template().format(**query)
    bounds = (f"Grid bounds: 0 <= x < {query['range_x'][1]}, "
              f"0 <= y < {query['range_y'][1]}.\n")
    before, after = text.rsplit('Generated Path:', 1)
    return before + bounds + 'Generated Path:' + after


def parse_advice(text):
    """Parse one complete nested list, without eval, repairs or oracle fallback."""
    marker = text.rfind('Generated Path:')
    candidate = text[marker + len('Generated Path:'):] if marker >= 0 else text
    start = candidate.find('[[')
    if start < 0:
        return [], 'missing_list'
    depth = 0
    end = None
    for i in range(start, len(candidate)):
        if candidate[i] == '[':
            depth += 1
        elif candidate[i] == ']':
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        return [], 'incomplete_list'
    try:
        values = ast.literal_eval(candidate[start:end])
    except (ValueError, SyntaxError, RecursionError):
        return [], 'invalid_list'
    if not isinstance(values, list) or not values or len(values) > 128:
        return [], 'invalid_shape'
    points = []
    for point in values:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return [], 'invalid_shape'
        if any(type(x) not in (int, float) or not math.isfinite(x) or int(x) != x for x in point):
            return [], 'non_integer_coordinate'
        points.append([int(x) for x in point])
    return points, 'ok'


def cases(split, max_maps=None, sample_ids=(0, 5), domain='original'):
    maps = [m for m in json.loads(DATA.read_text()) if m['id'] in split_ids()[split]]
    if max_maps is not None:
        maps = maps[:max_maps]
    result = []
    for m in maps:
        for sample_id in sample_ids:
            start, goal = m['start_goal'][sample_id][:2]
            query = {k: m[k] for k in ('range_x', 'range_y', 'horizontal_barriers', 'vertical_barriers')}
            query.update(start=start, goal=goal)
            if domain == 'transpose':
                query = {'start': start[::-1], 'goal': goal[::-1],
                         'range_x': m['range_y'], 'range_y': m['range_x'],
                         'horizontal_barriers': m['vertical_barriers'],
                         'vertical_barriers': m['horizontal_barriers']}
            elif domain == 'scale2':
                query = {'start': [2*x for x in start], 'goal': [2*x for x in goal],
                         'range_x': [0, 2*(m['range_x'][1]-1)+1],
                         'range_y': [0, 2*(m['range_y'][1]-1)+1],
                         'horizontal_barriers': [[2*x for x in b] for b in m['horizontal_barriers']],
                         'vertical_barriers': [[2*x for x in b] for b in m['vertical_barriers']]}
            result.append({'map_id': m['id'], 'sample_id': sample_id,
                           'domain': domain, 'query': query})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--split', choices=['validation', 'test'], default='test')
    parser.add_argument('--max-maps', type=int)
    parser.add_argument('--sample-ids', type=int, nargs='+', default=[0, 5])
    parser.add_argument('--domain', choices=['original', 'transpose', 'scale2'], default='original')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if any(i < 0 or i >= 10 for i in args.sample_ids) or len(set(args.sample_ids)) != len(args.sample_ids):
        parser.error('Sample IDs must be unique integers between 0 and 9')
    info = local_model(args.model)
    selected = cases(args.split, args.max_maps, args.sample_ids, args.domain)
    plan = {'model': info, 'split': args.split, 'cases': selected,
            'system_prompt': SYSTEM_PROMPT,
            'options': OPTIONS, 'endpoint': ENDPOINT, 'api_cost_usd': 0,
            'dataset_sha256': digest(DATA.read_bytes()),
            'prompt_template_sha256': digest(prompt_template().encode()),
            'selected_config_sha256': digest((ROOT/'benchmarks/selected_config.json').read_bytes()),
            'prompt_change': 'Upstream standard_gpt text + grid bounds + shared output-only system instruction chosen on validation, Ollama model template applied; not exact paper reproduction',
            'source_sha256': digest(Path(__file__).read_bytes()),
            'parse_policy': 'No oracle, output repair, retries, or successful-output selection; parse errors use empty advice'}
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Saved plan differs; use a new output folder')
    if not plan_path.exists():
        plan_path.write_text(json.dumps(plan, indent=2))
    raw_path = args.output/'responses.jsonl'
    existing = [json.loads(line) for line in raw_path.read_text().splitlines()] if raw_path.exists() else []
    seen = {(r['map_id'], r['sample_id'], r['domain']) for r in existing}
    started = time.perf_counter()
    for case in selected:
        key = (case['map_id'], case['sample_id'], case['domain'])
        if key in seen:
            continue
        prompt = make_prompt(case['query'])
        response = api('/api/generate', {'model': args.model, 'prompt': prompt,
                       'system': SYSTEM_PROMPT,
                       'stream': False, 'options': OPTIONS, 'keep_alive': '5m'})
        text = response.get('response', '')
        points, status = parse_advice(text)
        record = {**case, 'waypoints': points, 'parse_status': status,
                  'raw_response': text, 'prompt': prompt, 'prompt_sha256': digest(prompt.encode()),
                  'system_prompt': SYSTEM_PROMPT,
                  'generated_at': datetime.now(timezone.utc).isoformat(),
                  'provenance': {**info, 'options': OPTIONS, 'plan_sha256': digest(plan_path.read_bytes())},
                  'generation': {k: response.get(k) for k in ['done', 'done_reason', 'total_duration',
                      'load_duration', 'prompt_eval_count', 'prompt_eval_duration', 'eval_count', 'eval_duration']}}
        with raw_path.open('a') as f:
            f.write(json.dumps(record) + '\n')
        existing.append(record)
        seen.add(key)
        if len(seen) % 5 == 0 or len(seen) == len(selected):
            print(f'{args.model} {args.domain}: {len(seen)}/{len(selected)} responses; {time.perf_counter()-started:.1f}s', flush=True)
    (args.output/'waypoints.json').write_text(json.dumps(existing))
    summary = {'responses': len(existing), 'parse_errors': sum(r['parse_status'] != 'ok' for r in existing),
               'truncated': sum(r['generation']['done_reason'] == 'length' for r in existing),
               'eval_tokens': sum(r['generation'].get('eval_count') or 0 for r in existing),
               'total_generation_seconds': sum((r['generation'].get('total_duration') or 0)/1e9 for r in existing),
               'api_cost_usd': 0}
    (args.output/'generation_summary.json').write_text(json.dumps(summary, indent=2))
    # Unload our model before loading the next one on an 8 GB machine.
    api('/api/generate', {'model': args.model, 'keep_alive': 0, 'stream': False})
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
