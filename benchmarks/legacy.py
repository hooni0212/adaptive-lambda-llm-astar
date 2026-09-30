"""Replay the untouched upstream LLMAStar class without making API calls.

Only the class AST is loaded, avoiding heavyweight model imports. Its search,
queue, target filtering, metrics and collision methods remain unmodified.
Plotting/model generation are replaced by cached waypoints and no-op plotting.
Geometry is the same closed-segment intersection, verified against Shapely.
"""
import ast
import copy
import contextlib
import heapq
import io
import json
import math
from pathlib import Path
import runpy
from types import SimpleNamespace
from time import perf_counter

from llmastar.adaptive import intersects


class NoPlot:
    def __init__(self, *args): pass
    def animation(self, *args): pass


def load_class():
    root = Path(__file__).resolve().parents[1]
    source = ast.parse((root/'llmastar/pather/llm_a_star/llm_a_star.py').read_text())
    classes = [n for n in source.body if isinstance(n,ast.ClassDef)]
    namespace = {'json':json,'math':math,'heapq':heapq,
                 'env':SimpleNamespace(Env=runpy.run_path(str(root/'llmastar/env/search/env.py'))['Env']),
                 'plotting':SimpleNamespace(Plotting=NoPlot),
                 'is_lines_collision':lambda a,b: intersects(*a,*b)}
    exec(compile(ast.Module(body=classes,type_ignores=[]),'<upstream-class>','exec'),namespace)
    return namespace['LLMAStar']


UPSTREAM = load_class()


def sanitize(query, waypoints):
    """Use upstream waypoint filtering for every method in paper comparisons."""
    planner = object.__new__(UPSTREAM)
    planner._initialize_parameters(copy.deepcopy(query))
    return planner._filter_valid_nodes(waypoints)


def replay(query, waypoints):
    planner = object.__new__(UPSTREAM)
    def initialize():
        nodes = planner._filter_valid_nodes(waypoints)
        if not nodes or nodes[0] != planner.s_start:
            nodes.insert(0,planner.s_start)
        if not nodes or nodes[-1] != planner.s_goal:
            nodes.append(planner.s_goal)
        planner.target_list = nodes
        planner.i = 1
        planner.s_target = nodes[1]
    planner._initialize_llm_paths = initialize
    begin = perf_counter()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result = planner.searching(copy.deepcopy(query))
        return {**result,'success':True,'runtime_seconds':perf_counter()-begin,
                'path':planner.extract_path(planner.PARENT)}
    except KeyError:
        return {'operation':len(planner.CLOSED),'storage':len(planner.g),
                'success':False,'length':None,'runtime_seconds':perf_counter()-begin,'path':[]}
