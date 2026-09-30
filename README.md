# Adaptive Lambda LLM-A*

탐색 중 경유점 안내의 영향력을 조절하는 [LLM-A*](https://github.com/SilinMeng0510/llm-astar) 기반의 독립 연구 프로젝트입니다.

**경유점으로의 진척**, **탐색량 예산**, **기하학적 우회 비율**을 이용해 0 이상의 람다(λ) 가중치를 조절합니다. 저장된 경유점 안내를 사용하므로 모델을 호출하지 않고 오프라인에서 실험을 재현할 수 있습니다.

> **현재 검증 범위:** 합성 안내 실험에 더해 Qwen2.5 1.5B·Llama 3.2 3B의 실제 로컬 출력도 평가했습니다. 원 논문의 GPT-3.5·Llama 3 8B 출력을 재현한 결과는 아니며, 논문에 보고된 성능을 넘어섰다고 판단할 근거는 아직 없습니다.

## 무료 로컬 LLM 실험

유료 API 없이 이 Mac에서 두 공개 모델을 실행했습니다. 최종 프로토콜은 모델당 검증 20개·평가 120개·전이 환경 20개, 총 **실제 응답 320개**입니다. 평가용 지도 60개 모두를 포함하고, 기본 안내·순서 반전·누락·무작위 오류·안내 없음 조건과 좌표 전치·확대 환경에서 **6,200회 경로 탐색**을 비교했습니다. 람다 설정을 바꾸지 않고 형식 오류와 잘못된 좌표도 보존했습니다.

같은 수정 엔진의 고정 λ=1 대비, 변형하지 않은 실제 안내의 결과:

| 모델 | Operation 감소 | Storage 감소 | 경로 길이 변화 |
|---|---:|---:|---:|
| Qwen2.5 1.5B | 12.96% | 8.56% | 0.268% 증가 |
| Llama 3.2 3B | 14.45% | 8.89% | 0.114% 증가 |

확장 횟수와 상태 수는 줄었지만 **세 지표의 동시 개선은 확인하지 못했습니다.** 경로 길이 비율의 95% 신뢰구간은 두 모델 모두 1을 포함합니다. 일부 문제에서는 탐색량이 늘었으며 최악 성능 제한을 보장하지 않습니다.

[무료 로컬 LLM 실험 보고서](docs/local_llm_research.md)에서 모델별 성능, 신뢰구간, 안내 오류와 메모리 측정을 확인할 수 있습니다. 세 지표의 동시 개선 여부는 같은 수정 엔진의 고정 람다 비교군으로 판단해야 합니다. 아래 표는 이전 **합성 안내 실험**의 결과입니다.

## 합성 안내 실험 결과

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
| `benchmarks/local_llm.py` | 무료 로컬 모델의 실제 안내 생성·캐시 저장 |
| `benchmarks/local_evaluate.py` | 실제 안내와 오류 조건의 공통 입력 비교 |
| `tests/` | 경로 탐색·로컬 생성·실패 처리 등을 확인하는 22개 테스트 |
| `results/` | 원시 측정값, 안내 캐시, 결과 요약, 압축한 탐색 기록 |
| `docs/adaptive_lambda_research.md` | 기존 합성 안내 연구 보고서 |
| `docs/local_llm_research.md` | 실제 로컬 모델 실험 보고서 |
| `docs/upstream_readme.md` | 보존한 원본 프로젝트 README(영문) |
| `llmastar/pather/` | 수정하지 않은 원본 경로 탐색기 |
| `dataset/` | 원본 지도와 시각화 자료 |

결과 파일의 형식과 탐색 기록을 푸는 방법은 [실험 결과 안내](results/README.md)에 정리했습니다. GitHub Actions는 Python 3.11·3.14에서 정답성 테스트와 소규모 오프라인 실험을 실행합니다. 더 큰 외부 평가 데이터, 여러 생성 시드, 논문과 같은 모델·프롬프트를 사용한 직접 비교는 후속 과제로 남아 있습니다. 무료 로컬 실험은 [결과 파일 안내](results/local_llm/README.md)를 참고하세요.

## 원본 출처와 라이선스

Silin Meng 등 저자들의 [LLM-A* 논문](https://arxiv.org/abs/2407.02511)과 [공식 코드](https://github.com/SilinMeng0510/llm-astar)를 기반으로 하며, 커밋 `9414ed8`에서 시작했습니다. 원본의 커밋 이력과 [MIT 라이선스](LICENSE)를 유지합니다. 저자 표시와 확장 범위는 [NOTICE.md](NOTICE.md)를 참고하세요. 이 저장소에서는 원본 `llm-astar` 패키지를 PyPI에 배포하지 않습니다.
