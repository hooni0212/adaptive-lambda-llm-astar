"""Offline, confidence-weighted LLM-A*; no model or plotting dependencies.

Waypoints are advice, never mandatory path constraints. A graph supplies legal
transitions and an admissible goal heuristic. Lambda decay is not an optimality
guarantee: the first solution can still be suboptimal.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import heapq
import math
from time import perf_counter
from typing import Hashable, Protocol

State = Hashable
MOTIONS = ((-1, 0), (-1, 1), (0, 1), (1, 1),
           (1, 0), (1, -1), (0, -1), (-1, -1))


class SearchGraph(Protocol):
    def contains(self, state: State) -> bool: ...
    def neighbors(self, state: State): ...
    def distance(self, a: State, b: State) -> float: ...


def intersects(a, b, c, d):
    """Closed line segments intersect, including endpoints and collinearity."""
    def cross(p, q, r):
        return (q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0])
    def on(p, q, r):
        return min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and min(p[1], q[1]) <= r[1] <= max(p[1], q[1])
    x, y, z, t = cross(a,b,c), cross(a,b,d), cross(c,d,a), cross(c,d,b)
    return ((x*y < 0 and z*t < 0) or
            (x == 0 and on(a,b,c)) or (y == 0 and on(a,b,d)) or
            (z == 0 and on(c,d,a)) or (t == 0 and on(c,d,b)))


class GridGraph:
    """Original repository's 8-connected grid and closed barrier geometry.

range_x/y upper values are sizes (51/31 for the original dataset); outer
boundaries are at 0 and size-1. Input lists are copied, never modified.
The adjacency cache is shared by all compared methods; preprocessing time and
memory should be reported separately from planner-only measurements.
"""
    def __init__(self, query, *, diagonal=True):
        self.bounds = (tuple(query['range_x']), tuple(query['range_y']))
        self.barriers = tuple(
            [((x0,y), (x1,y)) for y,x0,x1 in query['horizontal_barriers']] +
            [((x,y0), (x,y1)) for x,y0,y1 in query['vertical_barriers']])
        self.motions = MOTIONS if diagonal else tuple(u for u in MOTIONS if 0 in u)
        self._adj = {}
        self._valid = {}

    def contains(self, state):
        if not isinstance(state, tuple) or len(state) != 2:
            return False
        if state in self._valid:
            return self._valid[state]
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and int(v) == v for v in state):
            return False
        x, y = state
        rx, ry = self.bounds
        valid = rx[0] < x < rx[1]-1 and ry[0] < y < ry[1]-1
        if valid:
            valid = not any(intersects(state,state,a,b) for a,b in self.barriers)
        self._valid[state] = valid
        return valid

    def neighbors(self, state):
        if state not in self._adj:
            self._adj[state] = tuple(
                (n, math.hypot(dx,dy)) for dx,dy in self.motions
                if self.contains(n := (state[0]+dx,state[1]+dy))
                and not any(intersects(state,n,a,b) for a,b in self.barriers))
        return self._adj[state]

    @staticmethod
    def distance(a, b):
        return math.dist(a,b)

    def states(self):
        rx, ry = self.bounds
        return [(x,y) for x in range(rx[0]+1,rx[1]-1)
                for y in range(ry[0]+1,ry[1]-1) if self.contains((x,y))]


@dataclass(frozen=True)
class LambdaConfig:
    # Frozen after train-only selection; see benchmarks/selection.json.
    initial_weight: float = 2.0
    interval: int = 32
    good_progress: float = 0.0  # keep positive progress; decay on stagnation
    slow_decay: float = 0.8
    stalled_decay: float = 0.5
    min_weight: float = 0.1
    use_progress: bool = True
    use_budget: bool = True
    budget_factor: float = 8.0  # waypoint distance in minimum-step units
    use_detour: bool = True
    detour_alpha: float = 0.75
    # Upstream retains the waypoint term at the final goal (h_goal appears twice).
    # Preserve that convention by default; set to zero for an A*-ordered finish.
    final_goal_weight: float = 1.0
    reopen: bool = True
    skip: bool = False  # evaluate separately from lambda adaptation

    def __post_init__(self):
        for name in ('initial_weight','good_progress','min_weight','detour_alpha','final_goal_weight'):
            if not math.isfinite(getattr(self,name)) or getattr(self,name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.interval < 1 or not isinstance(self.interval,int):
            raise ValueError('interval must be a positive integer')
        if self.budget_factor <= 0 or not math.isfinite(self.budget_factor):
            raise ValueError('budget_factor must be finite and positive')
        for name in ('slow_decay','stalled_decay'):
            if not 0 <= getattr(self,name) <= 1:
                raise ValueError(f'{name} must be in [0, 1]')


@dataclass
class SearchResult:
    success: bool
    path: list
    operation: int
    storage: int  # distinct finite-cost discovered states, not bytes
    length: float | None
    runtime_seconds: float
    peak_open: int
    peak_queue_entries: int
    reopened: int
    queue_rebuilds: int
    score_evaluations: int
    lambda_history: list
    waypoint_count: int
    invalid_waypoints: int
    fallback_count: int
    skipped_waypoints: int

    def to_dict(self):
        return asdict(self)


class AdaptiveLLMAStar:
    """Search using g + h_goal + lambda * h_waypoint on a supplied graph.

    States are mutually orderable tuples, also used for deterministic heap ties.
    A non-grid graph must supply a goal lower bound via distance for A* claims;
    geometric waypoint distances need not be admissible. The selected default
    retains upstream's weighted final-goal phase, so empty advice is NOT vanilla
    A*. Use LambdaConfig(final_goal_weight=0) for that behavior.
    """
    def __init__(self, config: LambdaConfig | None = None):
        self.config = config or LambdaConfig()

    def searching(self, query, waypoints=(), *, graph=None, max_expansions=None):
        start, goal = tuple(query['start']), tuple(query['goal'])
        graph = graph or GridGraph(query)
        cfg = self.config
        begin = perf_counter()
        invalid = 0
        targets = []
        for raw in waypoints:
            try:
                wp = tuple(raw)
                legal = graph.contains(wp)
            except (TypeError, ValueError):
                legal = False
            if not legal:
                invalid += 1
            elif wp not in (start,goal) and (not targets or wp != targets[-1]):
                targets.append(wp)
        count_wp = len(targets)
        targets.append(goal)
        idx = 0
        target = targets[idx]
        g = {start:0.0}
        parent = {start:start}
        opened = {start}
        closed = set()
        queue = []
        operation = reopened = rebuilds = evaluations = fallbacks = skips = 0
        peak_open = peak_entries = 1
        history = []
        since = 0
        prev_distance = graph.distance(start,target)
        scale = max(graph.distance(start,goal),1.0)
        budget = max(cfg.interval, cfg.budget_factor*prev_distance)

        def initial_weight(node, wp):
            if wp == goal:
                return cfg.final_goal_weight
            weight = cfg.initial_weight
            if cfg.use_detour:
                direct = graph.distance(node,goal)
                via = graph.distance(node,wp)+graph.distance(wp,goal)
                ratio = via/max(direct,1e-12)
                weight *= math.exp(-cfg.detour_alpha*max(0.0,ratio-1))
            return weight if weight >= cfg.min_weight else 0.0

        weight = initial_weight(start,target)

        def score(node):
            nonlocal evaluations
            evaluations += 1
            return g[node]+graph.distance(node,goal)+weight*graph.distance(node,target)

        def record(reason):
            history.append({'expansion':operation,'target_index':idx,
                            'target':target,'weight':weight,'reason':reason})

        def rebuild():
            nonlocal queue, rebuilds
            queue = [(score(n),n,g[n]) for n in opened]
            heapq.heapify(queue)
            rebuilds += 1

        def advance(node, reason):
            nonlocal idx,target,weight,since,prev_distance,budget
            idx += 1
            target = targets[idx]
            weight = initial_weight(node,target)
            since = 0
            prev_distance = graph.distance(node,target)
            budget = max(cfg.interval,cfg.budget_factor*prev_distance)
            record(reason)
            rebuild()

        record('initial')
        heapq.heappush(queue,(score(start),start,0.0))
        success = False
        if not graph.contains(start) or not graph.contains(goal):
            queue.clear()
            opened.clear()
        while queue and (max_expansions is None or operation < max_expansions):
            _, node, old_g = heapq.heappop(queue)
            if node not in opened or old_g != g[node]:
                continue
            opened.remove(node)
            closed.add(node)
            operation += 1
            if node == goal:
                success = True
                break
            since += 1
            # Match paper's neighbor-discovery target switching, but only after
            # checking the transition is legal. No forced waypoint traversal.
            neighbors = graph.neighbors(node)
            for nxt,cost in neighbors:
                if not math.isfinite(cost) or cost < 0:
                    raise ValueError('edge costs must be finite and nonnegative')
                if nxt == target and target != goal:
                    advance(nxt,'waypoint_discovered')
                candidate = g[node]+cost
                if candidate+1e-12 >= g.get(nxt,math.inf):
                    continue
                if nxt in closed:
                    if not cfg.reopen:
                        continue
                    closed.remove(nxt)
                    reopened += 1
                g[nxt] = candidate
                parent[nxt] = node
                opened.add(nxt)
                heapq.heappush(queue,(score(nxt),nxt,candidate))
                peak_open = max(peak_open,len(opened))
                peak_entries = max(peak_entries,len(queue))
            if target != goal and since > 0 and since % cfg.interval == 0:
                current_distance = graph.distance(node,target)
                progress = (prev_distance-current_distance)/cfg.interval/scale
                new_weight = weight
                if cfg.use_progress:
                    if progress <= 0:
                        new_weight *= cfg.stalled_decay
                    elif progress <= cfg.good_progress:
                        new_weight *= cfg.slow_decay
                if cfg.use_budget and since > budget:
                    new_weight *= cfg.stalled_decay
                if new_weight < cfg.min_weight:
                    new_weight = 0.0
                if new_weight != weight:
                    if new_weight == 0 and weight > 0:
                        fallbacks += 1
                    weight = new_weight
                    record('decay')
                    rebuild()
                prev_distance = current_distance
                if cfg.skip and weight == 0 and since > 2*budget:
                    skips += 1
                    advance(node,'skip')
            peak_open = max(peak_open,len(opened))
            peak_entries = max(peak_entries,len(queue))
        path = []
        if success:
            n = goal
            while n != start:
                path.append(n)
                n = parent[n]
            path.append(start)
            path.reverse()
        # Reopened ancestors may improve parent chains before all downstream
        # g-values propagate; measure the returned path, as upstream does.
        length = sum(next(c for nxt,c in graph.neighbors(a) if nxt == b)
                     for a,b in zip(path,path[1:])) if success else None
        return SearchResult(success,path,operation,len(g),length,
                            perf_counter()-begin,peak_open,peak_entries,reopened,rebuilds,
                            evaluations,history,count_wp,invalid,fallbacks,skips)


def fixed_config(weight=1.0, *, paper_goal=True, reopen=False):
    """Controlled fixed-lambda baseline on the same search engine."""
    return LambdaConfig(initial_weight=weight,use_progress=False,use_budget=False,
                        use_detour=False,final_goal_weight=weight if paper_goal else 0.0,
                        min_weight=0.0,reopen=reopen)


def validate_path(graph, path, start, goal):
    if not path or path[0] != start or path[-1] != goal:
        return False
    return all(graph.contains(n) for n in path) and all(
        any(n == b and math.isfinite(c) for n,c in graph.neighbors(a))
        for a,b in zip(path,path[1:]))
