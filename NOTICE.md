# Attribution and scope

This research repository builds on [SilinMeng0510/llm-astar](https://github.com/SilinMeng0510/llm-astar),
initially cloned at commit `9414ed8`. The original MIT license and author
attribution are retained in [LICENSE](LICENSE), and the upstream commit history
is preserved. The original README is retained in [docs/upstream_readme.md](docs/upstream_readme.md).

Original paper: Silin Meng, Yiwei Wang, Cheng-Fu Yang, Nanyun Peng, and Kai-Wei
Chang, *LLM-A*: Large Language Model Enhanced Incremental Heuristic Search on
Path Planning*, Findings of EMNLP 2024, [arXiv:2407.02511](https://arxiv.org/abs/2407.02511).

The adaptive-lambda planner, offline replay harness, tests, configurations, and
research reports are additions maintained in this repository. This is an
independent research extension, not an official release by the paper's authors.

The benchmark results use synthetic oracle-derived/corrupted waypoints and a
separately labeled 12-case assistant-authored pilot. They do not reproduce the
paper's GPT-3.5 or LLAMA3 outputs, and do not establish superiority over its
published results.
