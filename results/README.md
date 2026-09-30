# 실험 결과 파일 안내

모델 호출 없이 실행한 비교 실험 결과를 검토할 수 있도록 보존했습니다. 저장된 측정 결과이며, 저장소를 열기만 해서는 실험이 다시 실행되지 않습니다.

| 폴더 | 용도 |
|---|---|
| `train_tuning` | 학습용 지도만 사용한 첫 번째 설정 탐색; 초기 구현의 결과 |
| `train_refined` | 학습용 지도만 사용한 두 번째 설정 탐색; 기본 설정을 선택한 실험 |
| `validation` | 별도의 지도 20개를 이용한 설정 검증 |
| `test` | 최종 평가용 지도 60개·문제 600개·안내 조건 7가지의 결과 |
| `transfer` | 검증용 지도의 좌표 전치·2배 확대·가중 간선 환경에서의 검증 |
| `assistant_pilot` | AI 도우미가 경유점을 작성한 예비 실험 12개 |
| `smoke` | 초기 지도 2개에서의 동작 확인; 최종 결론에는 사용하지 않음 |

주요 실험에는 `metadata.json`, `raw.csv`, `summary.json`, 경유점 캐시인 `waypoints.json`이 포함됩니다. 최종 평가에는 전체 집계인 `aggregate.json`과 비교 그래프인 `comparison.png`도 있습니다. 메타데이터에는 설정, 지도 ID, 안내의 출처, 데이터셋 해시와 최종 실험의 소스 코드 해시를 기록했습니다. 과거 실험의 소스 코드 해시는 당시 코드를 나타내며, 이후 수정된 코드와 다를 수 있습니다.

용량이 큰 경로와 람다 변화 기록은 `traces.json.gz`로 손실 없이 압축했습니다. `archive_manifest.json`에는 압축 전후의 SHA256 해시가 있습니다. 로컬에서 기록을 풀려면 다음 명령을 실행하세요.

```bash
python -c 'import gzip,pathlib; p=pathlib.Path("results/test/traces.json.gz"); p.with_suffix("").write_bytes(gzip.decompress(p.read_bytes()))'
```

실험을 다시 실행하면 소요 시간은 달라질 수 있습니다. 탐색기 실행 시간에는 그래프 준비와 LLM 안내 생성 시간이 포함되지 않습니다. `--save-paths` 옵션은 `traces.json`을 생성하며, 이 파일은 압축 전까지 Git 추적 대상에서 제외됩니다. 보고서 생성기는 CSV·JSON 요약을 읽으므로 탐색 기록을 풀 필요가 없습니다.

논문과 결과를 비교하기 전에 [전체 보고서](../docs/adaptive_lambda_research.md)를 확인하세요. 특히 원본 Storage에는 비용이 무한대인 항목도 포함되지만, 새 엔진은 발견한 상태 중 비용이 유한한 상태를 셉니다. 같은 수정 엔진을 사용하는 고정 람다 비교군을 함께 제공해 공통 구현 변경과 적응형 가중치의 효과를 구분했습니다.
