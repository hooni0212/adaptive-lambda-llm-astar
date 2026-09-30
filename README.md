# Adaptive Lambda LLM-A*

An independent research extension of [LLM-A*](https://github.com/SilinMeng0510/llm-astar)
that adjusts the influence of waypoint advice during search.

The planner uses **waypoint progress**, **expansion budget**, and **geometric detour**
to adjust a nonnegative lambda weight. Cached waypoints make experiments offline
and repeatable without model calls.

> **Research status:** the large benchmark uses synthetic oracle-derived and
> corrupted waypoints. These results do **not** reproduce the paper's GPT-3.5 or
> LLAMA3 outputs and do **not** establish superiority over its published results.

## Held-out results

Training, validation, and test maps are disjoint: **20 / 20 / 60 maps**. The
selected configuration was fixed before validation and test evaluation.
The final test contains 600 start/goal queries in seven advice conditions,
for **4,200 adaptive paths**, all checked for validity against the graph.

The table shows reductions in paired geometric means, pooling the seven
synthetic advice conditions with equal weight. Storage counts states, not bytes.

| Comparator | Operation reduction | Storage reduction | Path-length reduction |
|---|---:|---:|---:|
| Upstream code replay | 15.68% | 11.71% | 0.594% |
| Fixed lambda=1 on the same corrected engine | 6.04% | 2.98% | 0.296% |

![Held-out synthetic benchmark](results/test/comparison.png)

Not every individual case improves. The same-engine fixed lambda=2 baseline
uses fewer operations overall but produces longer paths than the adaptive method.
Exact advice comes from a Dijkstra oracle, and random corruption does not model
the full distribution of real LLM errors.

See the [full Korean research report](docs/adaptive_lambda_research.md) for
per-condition results, ablations, confidence intervals, transfer checks,
actual-memory caveats, and the 12-case assistant-authored pilot.

## Quick start

Python 3.11+ is recommended. The offline planner and benchmark harness need only
the standard library. Geometry verification and report plotting use the small
optional dependency list below; the upstream model stack is not needed.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-research.txt
python -m unittest discover -s tests -v
```

```python
from llmastar.adaptive import AdaptiveLLMAStar

query = {
    "start": [5, 5], "goal": [27, 15],
    "horizontal_barriers": [[10, 0, 25], [15, 30, 50]],
    "vertical_barriers": [[25, 10, 22]],
    "range_x": [0, 51], "range_y": [0, 31],
}
result = AdaptiveLLMAStar().searching(query, waypoints=[[26, 9], [26, 14]])
print(result.success, result.operation, result.storage, result.length)
```

The score is $f(n)=g(n)+h_{goal}(n)+\lambda_t h_{waypoint}(n)$.
Lambda changes rebuild OPEN priorities; better paths can reopen CLOSED states.
Waypoints provide advice and are not mandatory path constraints.

The selected default preserves upstream's additional goal term in the final
phase (`g + 2*h_goal`). Use `LambdaConfig(final_goal_weight=0)` for an
A*-ordered final phase. Changing this option changes the reported performance.
Lambda decay does not guarantee an optimal first solution or a worst-case
expansion bound.

## Reproduce the benchmark

From the repository root, with the environment activated:

```bash
python -m benchmarks.run --split test \
  --config benchmarks/selected_config.json \
  --methods astar fixed_1 fixed_025 fixed_05 fixed_2 adaptive \
    no_progress no_budget no_detour no_reopen fixed_2_controlled fixed_1_controlled \
  --output results/test --save-paths
python -m benchmarks.transfer
python -m benchmarks.run --split test \
  --waypoints benchmarks/assistant_waypoints_pilot.json \
  --config benchmarks/selected_config.json \
  --methods astar fixed_1_controlled adaptive \
  --output results/assistant_pilot --save-paths
python -m benchmarks.report
```

The runner accepts real cached model outputs through `--waypoints FILE`.
Each record contains `map_id`, `sample_id`, and `waypoints`; retain the model,
prompt, and generation settings in `provenance`. The runner makes no API calls.

## Repository guide

| Path | Contents |
|---|---|
| `llmastar/adaptive.py` | Adaptive planner and grid/graph interface |
| `benchmarks/` | Replay harness, frozen configurations, and report tools |
| `tests/test_adaptive.py` | 15 correctness checks |
| `results/` | Raw measurements, cached advice, summaries, and compressed traces |
| `docs/adaptive_lambda_research.md` | Full research report |
| `docs/upstream_readme.md` | Original project README |
| `llmastar/pather/` | Unmodified upstream planners |
| `dataset/` | Original maps and visualization assets |

See [results/README.md](results/README.md) for artifact formats and trace
extraction. GitHub Actions checks correctness and a small offline benchmark on
Python 3.11 and 3.14. Model generation, larger external benchmarks, and a direct
comparison with the paper's actual LLM outputs remain future work.

## Attribution

Based on Silin Meng and collaborators' [LLM-A* paper](https://arxiv.org/abs/2407.02511)
and [official code](https://github.com/SilinMeng0510/llm-astar), starting at commit
`9414ed8`. Original history and the [MIT license](LICENSE) are retained.
See [NOTICE.md](NOTICE.md) for scope and attribution. This repository does not
publish the original `llm-astar` package to PyPI.
