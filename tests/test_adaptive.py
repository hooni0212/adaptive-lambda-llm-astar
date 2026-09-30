import copy
from dataclasses import replace
import math
import random
import unittest

from llmastar.adaptive import (AdaptiveLLMAStar, GridGraph, LambdaConfig,
                              fixed_config, intersects, validate_path)
from benchmarks.run import oracle, split_ids
from benchmarks.legacy import replay


def query(start=(2,2),goal=(8,8),h=(),v=(),size=12):
    return {'start':list(start),'goal':list(goal),'range_x':[0,size],
            'range_y':[0,size],'horizontal_barriers':list(h),'vertical_barriers':list(v)}


class PlannerTests(unittest.TestCase):
    def test_zero_lambda_matches_dijkstra_and_does_not_mutate_query(self):
        q=query(h=[[5,0,7]])
        before=copy.deepcopy(q);g=GridGraph(q)
        expected,_=oracle(g,tuple(q['start']),tuple(q['goal']))
        r=AdaptiveLLMAStar(fixed_config(0,reopen=True)).searching(q,[(7,4),(8,6)],graph=g)
        self.assertTrue(r.success)
        self.assertAlmostEqual(r.length,expected)
        self.assertEqual(q,before)
        self.assertTrue(validate_path(g,r.path,tuple(q['start']),tuple(q['goal'])))

    def test_start_equals_goal(self):
        q=query(goal=(2,2));r=AdaptiveLLMAStar().searching(q,[(3,3)])
        self.assertEqual(r.path,[(2,2)]);self.assertEqual(r.length,0)
        self.assertEqual(r.operation,1)

    def test_no_path_returns_failure(self):
        q=query(h=[[5,0,11]])
        r=AdaptiveLLMAStar().searching(q)
        self.assertFalse(r.success);self.assertIsNone(r.length);self.assertEqual(r.path,[])

    def test_invalid_endpoints_and_waypoints(self):
        q=query(start=(0,0));r=AdaptiveLLMAStar().searching(q)
        self.assertFalse(r.success)
        q=query(h=[[5,0,7]])
        r=AdaptiveLLMAStar().searching(q,[(5,5),(99,99),(2.5,3),(float('nan'),2),None,(3,3),(3,3)])
        self.assertEqual(r.invalid_waypoints,5)
        self.assertEqual(r.waypoint_count,1)

    def test_no_waypoints_is_astar_when_goal_weight_is_zero(self):
        q=query(h=[[5,0,7]])
        a=AdaptiveLLMAStar(LambdaConfig(final_goal_weight=0)).searching(q)
        b=AdaptiveLLMAStar(fixed_config(0,reopen=True)).searching(q)
        self.assertEqual(a.operation,b.operation);self.assertEqual(a.path,b.path)

    def test_lambda_updates_are_monotone_per_target_and_queue_rebuilt(self):
        q=query(h=[[5,0,7]])
        c=LambdaConfig(interval=1,use_detour=False,good_progress=100,
                       budget_factor=1,final_goal_weight=0)
        r=AdaptiveLLMAStar(c).searching(q,[(2,9)])
        weights=[h['weight'] for h in r.lambda_history if h['target_index']==0]
        self.assertGreater(len(weights),1)
        self.assertTrue(all(a>=b for a,b in zip(weights,weights[1:])))
        self.assertGreater(r.queue_rebuilds,0)
        self.assertTrue(validate_path(GridGraph(q),r.path,tuple(q['start']),tuple(q['goal'])))

    def test_blocked_neighbor_cannot_advance_waypoint(self):
        q=query(start=(4,4),goal=(8,8),h=[[4.5,0,6]])
        c=LambdaConfig(use_progress=False,use_budget=False,use_detour=False)
        r=AdaptiveLLMAStar(c).searching(q,[(4,5)])
        transition=[h for h in r.lambda_history if h['reason']=='waypoint_discovered']
        self.assertTrue(r.success)
        self.assertGreater(transition[0]['expansion'],1)

    def test_memory_and_operation_are_not_upstream_duplicate_counts(self):
        q=query(h=[[5,0,7]])
        r=AdaptiveLLMAStar(fixed_config(0)).searching(q)
        self.assertLessEqual(r.operation,r.storage)
        self.assertGreaterEqual(r.storage,r.peak_open)
        self.assertGreaterEqual(r.peak_queue_entries,r.peak_open)

    def test_random_grids_astar_matches_dijkstra(self):
        rng=random.Random(310)
        for _ in range(30):
            q=query(h=[[rng.randint(3,8),0,rng.randint(3,8)]],
                    v=[[rng.randint(3,8),rng.randint(3,8),10]])
            g=GridGraph(q);cost,_=oracle(g,tuple(q['start']),tuple(q['goal']))
            r=AdaptiveLLMAStar(fixed_config(0,reopen=True)).searching(q,graph=g)
            self.assertEqual(r.success,cost is not None)
            if cost is not None:self.assertAlmostEqual(r.length,cost)

    def test_split_separates_maps(self):
        splits=split_ids()
        self.assertEqual(len(splits['test']),60)
        self.assertFalse(splits['train']&splits['test'])
        self.assertFalse(splits['validation']&splits['test'])
        self.assertEqual(splits['train']|splits['validation']|splits['test'],set(range(100)))

    def test_config_validation(self):
        for kw in [{'interval':0},{'initial_weight':-1},{'slow_decay':2},
                   {'detour_alpha':float('nan')},{'budget_factor':0}]:
            with self.assertRaises(ValueError):LambdaConfig(**kw)

    def test_upstream_replay_frozen_example(self):
        q=query(start=(3,3),goal=(8,8))
        old=copy.deepcopy(q);r=replay(q,[(5,5),(6,6)])
        self.assertTrue(r['success']);self.assertEqual(q,old)
        self.assertAlmostEqual(r['length'],5*math.sqrt(2))

    def test_segment_geometry_matches_shapely(self):
        from shapely.geometry import LineString
        rng=random.Random(777)
        for _ in range(500):
            pts=[(rng.randint(-5,5),rng.randint(-5,5)) for _ in range(4)]
            if pts[0]==pts[1] or pts[2]==pts[3]:continue
            self.assertEqual(intersects(*pts),LineString(pts[:2]).intersects(LineString(pts[2:])))
        self.assertTrue(intersects((0,0),(2,2),(2,2),(4,0)))
        self.assertTrue(intersects((0,0),(2,0),(1,0),(3,0)))

    def test_non_grid_weighted_graph(self):
        class Graph:
            edges={(0,):[((1,),2),((2,),1)],(2,):[((1,),.25)],(1,):[((3,),1)],(3,):[]}
            def contains(self,n):return n in self.edges
            def neighbors(self,n):return self.edges[n]
            def distance(self,a,b):return 0
        g=Graph();q={'start':[0],'goal':[3]}
        r=AdaptiveLLMAStar().searching(q,[(1,)],graph=g)
        self.assertTrue(r.success);self.assertEqual(r.length,2.25)
        self.assertEqual(r.path,[(0,),(2,),(1,),(3,)])

    def test_better_path_reopens_closed_nodes(self):
        class Graph:
            edges={(0,):[((1,),4),((2,),1)],(1,):[((3,),1)],
                   (2,):[((1,),1)],(3,):[((4,),10)],(4,):[]}
            def contains(self,n):return n in self.edges
            def neighbors(self,n):return self.edges[n]
            def distance(self,a,b):return 5 if a==(2,) and b==(4,) else 0
        q={'start':[0],'goal':[4]};g=Graph()
        r=AdaptiveLLMAStar(fixed_config(0,reopen=True)).searching(q,graph=g)
        old=AdaptiveLLMAStar(fixed_config(0,reopen=False)).searching(q,graph=g)
        self.assertEqual(r.reopened,2);self.assertEqual(r.length,13)
        self.assertEqual(old.length,15)


if __name__=='__main__':unittest.main()
