# Adaptive Lambda LLM-A*

탐색 중 경유점 안내의 영향력을 조절하는 [LLM-A*](https://github.com/SilinMeng0510/llm-astar) 기반의 독립 연구 프로젝트입니다.

**경유점으로의 진척**, **탐색량 예산**, **기하학적 우회 비율**을 이용해 0 이상의 람다(λ) 가중치를 조절합니다. 저장된 경유점 안내를 사용하므로 모델을 호출하지 않고 오프라인에서 실험을 재현할 수 있습니다.

> **현재 검증 범위:** 대규모 실험에는 최적 경로에서 추출한 합성 경유점과 오류를 넣은 경유점을 사용했습니다. 논문의 GPT-3.5·LLAMA3 출력을 재현한 결과가 아니며, 논문에 보고된 성능을 넘어섰다고 판단할 근거는 아직 없습니다.

## 평가용 지도에서의 결과

학습·검증·평가용 지도를 각각 **20 / 20 / 60개**로 나누었으며, 지도는 서로 겹치지 않습니다. 검증·평가에 앞서 사용할 설정을 고정했습니다. 최종 평가에서는 600개 시작·목표 문제에 일곱 가지 안내 조건을 적용해 **4,200개 경로**를 생성했고, 모두 그래프상에서 유효한지 확인했습니다.

아래 표는 같은 입력에서 얻은 지표 비율의 기하평균을 감소율로 나타낸 것입니다. 일곱 합성 안내 조건에 동일한 비중을 적용했습니다. Operation은 탐색 시 노드 확장 횟수, Storage는 저장한 상태 수를 뜻합니다. Storage는 실제 메모리 사용량(바이트)과 다릅니다.

| 비교 대상 | Operation 감소 | Storage 감소 | 경로 길이 감소 |
|---|---:|---:|---:|
| 원본 코드 재실행 | 15.68% | 11.71% | 0.594% |
| 같은 수정 엔진의 고정 λ=1 | 6.04% | 2.98% | 0.296% |

![평가용 지도에서의 합성 실험 결과](results/test/comparison.png)

모든 개별 문제에서 성능이 개선된 것은 아닙니다. 같은 엔진의 고정 λ=2 방식은 전체적으로 더 적은 확장 횟수를 사용하지만, 적응형 방식보다 경로가 깁니다. 정확한 합성 안내는 Dijkstra 최적 경로에서 추출한 것이며, 무작위 오류는 실제 LLM 오류 분포 전체를 대표하지 않습니다.

[전체 연구 보고서](docs/adaptive_lambda_research.md)에서 조건별 결과, 구성 요소를 제거한 비교 실험, 신뢰구간, 다른 환경에서의 검증, 실제 메모리 측정의 한계, AI 도우미가 경유점을 작성한 12개 예비 실험을 확인할 수 있습니다.

## 빠르게 시작하기

Python 3.11 이상을 권장합니다. 오프라인 경로 탐색기와 실험 실행 도구는 표준 라이브러리만으로 동작합니다. 아래 추가 패키지는 장애물 충돌 판정 검증과 보고서 그래프 작성에 사용하며, 원본 프로젝트의 모델 관련 패키지는 필요하지 않습니다.

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

탐색 점수는 $f(n)=g(n)+h_{goal}(n)+\lambda_t h_{waypoint}(n)$입니다. 람다가 바뀌면 탐색 대기 목록(OPEN)의 우선순위를 다시 계산합니다. 더 짧은 경로를 찾으면 이미 확장한 상태(CLOSED)도 다시 탐색할 수 있습니다. 경유점은 탐색을 돕는 안내이며, 반드시 통과해야 하는 경로 제약은 아닙니다.

선택한 기본 설정은 마지막 단계에서 원본의 추가 목표 항을 유지합니다(`g + 2*h_goal`). 마지막 단계를 A* 순서로 탐색하려면 `LambdaConfig(final_goal_weight=0)`을 사용하세요. 이 옵션을 바꾸면 위 성능 수치가 그대로 적용되지 않습니다. 람다를 줄이는 것만으로 첫 번째 경로의 최적성이나 최악의 경우의 확장 횟수 상한을 보장하지는 않습니다.

## 실험 재현하기

저장소의 최상위 폴더에서 가상환경을 활성화한 뒤 실행합니다.

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

`--waypoints FILE`을 통해 실제 모델이 생성한 경유점 캐시도 입력할 수 있습니다. 각 기록에는 `map_id`, `sample_id`, `waypoints`가 필요하며, `provenance`에는 모델·프롬프트·생성 설정을 남겨야 합니다. 실험 실행 도구 자체는 API를 호출하지 않습니다.

## 저장소 구성

| 경로 | 내용 |
|---|---|
| `llmastar/adaptive.py` | 적응형 경로 탐색기와 격자·그래프 인터페이스 |
| `benchmarks/` | 비교 실험 도구, 고정한 설정, 보고서 생성 도구 |
| `tests/test_adaptive.py` | 정답성을 확인하는 15개 테스트 |
| `results/` | 원시 측정값, 안내 캐시, 결과 요약, 압축한 탐색 기록 |
| `docs/adaptive_lambda_research.md` | 전체 연구 보고서 |
| `docs/upstream_readme.md` | 보존한 원본 프로젝트 README(영문) |
| `llmastar/pather/` | 수정하지 않은 원본 경로 탐색기 |
| `dataset/` | 원본 지도와 시각화 자료 |

결과 파일의 형식과 탐색 기록을 푸는 방법은 [실험 결과 안내](results/README.md)에 정리했습니다. GitHub Actions는 Python 3.11·3.14에서 정답성 테스트와 소규모 오프라인 실험을 실행합니다. 실제 모델을 이용한 안내 생성, 더 큰 외부 평가 데이터, 논문의 실제 LLM 출력과의 직접 비교는 후속 과제로 남아 있습니다.

## 원본 출처와 라이선스

Silin Meng 등 저자들의 [LLM-A* 논문](https://arxiv.org/abs/2407.02511)과 [공식 코드](https://github.com/SilinMeng0510/llm-astar)를 기반으로 하며, 커밋 `9414ed8`에서 시작했습니다. 원본의 커밋 이력과 [MIT 라이선스](LICENSE)를 유지합니다. 저자 표시와 확장 범위는 [NOTICE.md](NOTICE.md)를 참고하세요. 이 저장소에서는 원본 `llm-astar` 패키지를 PyPI에 배포하지 않습니다.
