"""Build a Korean report from completed local-model v2 measurements."""
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt

from .run import ROOT

MODELS = [('qwen', 'Qwen2.5 1.5B'), ('llama', 'Llama 3.2 3B')]
LABELS = {'actual':'실제 모델 안내', 'reverse':'순서 반전', 'drop_half':'절반 누락',
          'corrupt50':'절반 무작위 교체', 'none':'안내 없음'}


def read(path):
    return json.loads(path.read_text())


def main():
    root = ROOT/'results/local_llm'
    evaluations = {}
    generations = {}
    for prefix,label in MODELS:
        folder = root/(prefix+'_evaluation_v2')
        with (folder/'raw.csv').open() as f:
            rows = list(csv.DictReader(f))
        evaluations[prefix] = {'summary':read(folder/'summary.json'), 'rows':rows,
                               'quality':read(folder/'quality.json'),
                               'ci':read(folder/'bootstrap.json'),
                               'memory':read(folder/'allocations.json'),
                               'metadata':read(folder/'metadata.json')}
        generations[prefix] = {s:read(root/f'{prefix}_{s}_v2/generation_summary.json')
                                for s in ['validation','test','transpose','scale2']}
        assert generations[prefix]['test']['responses'] == 120
        assert len(rows) == 3100

    def result(prefix, baseline, scenario='actual', domain='original'):
        return next(r for r in evaluations[prefix]['summary'] if r['baseline']==baseline
                    and r['scenario']==scenario and r['domain']==domain)

    existing = {f.name for f in font_manager.fontManager.ttflist}
    for family in ['Apple SD Gothic Neo', 'NanumGothic', 'Noto Sans CJK KR']:
        if family in existing:
            plt.rcParams['font.family'] = family
            break
    plt.rcParams['axes.unicode_minus'] = False
    fig,axes = plt.subplots(1,3,figsize=(14,4.8),layout='constrained')
    labels = [f'{label}\n{base}' for _,label in MODELS for base in ['원본 재실행','고정 λ=1']]
    for axis,k,title in zip(axes,['operation','storage','length'],['Operation','Storage · 상태 수','경로 길이']):
        values, low, high = [], [], []
        for prefix,_ in MODELS:
            for base in ['legacy','fixed_1_controlled']:
                v = 100*result(prefix,base)['ratios'][k]
                ci = evaluations[prefix]['ci'][base][k]
                values.append(v); low.append(v-100*ci[0]); high.append(100*ci[1]-v)
        axis.scatter(values,range(4),color=['#267c78','#659e99','#267c78','#659e99'],s=65,zorder=3)
        axis.set_yticks(range(4),labels)
        axis.errorbar(values,range(4),xerr=[low,high],fmt='none',ecolor='#263b45',capsize=3)
        axis.axvline(100,color='#b74d43',linestyle='--')
        pad = 5 if k != 'length' else .2
        axis.set_xlim(min([100]+[v-l for v,l in zip(values,low)])-pad,
                      max([100]+[v+h for v,h in zip(values,high)])+pad)
        axis.set_ylim(3.5,-.5); axis.set_title(title)
        axis.set_xlabel('비교 대상 = 100 · 낮을수록 좋음')
        axis.grid(axis='x',alpha=.15);axis.set_axisbelow(True)
        for i,v in enumerate(values):
            axis.text(v,i-.2,f'{v:.2f}',ha='center',color='#263b45',fontsize=11)
    fig.suptitle('무료 로컬 LLM · 실제 안내 120개씩 · 동일한 람다 설정\n오차 막대: 지도 단위 95% 신뢰구간 · 원 논문 모델의 재현 실험이 아님',fontsize=14)
    fig.savefig(root/'comparison_v2.png',dpi=180);plt.close(fig)

    table = ['| 모델 | 비교 대상 | Operation 비율 | Storage 비율 | 경로 길이 비율 |',
             '|---|---|---:|---:|---:|']
    intervals = ['| 모델 | 비교 대상 | Operation 95% 구간 | Storage 95% 구간 | 경로 길이 95% 구간 |',
                 '|---|---|---:|---:|---:|']
    stress = ['| 모델 | 안내 조건 | Operation 비율 | Storage 비율 | 경로 길이 비율 | Operation P95 비율 |',
              '|---|---|---:|---:|---:|---:|']
    transfers = ['| 모델 | 환경 | 문제 수 | Operation 비율 | Storage 비율 | 경로 길이 비율 |',
                 '|---|---|---:|---:|---:|---:|']
    quality = ['| 모델 | 파싱 실패 | 출력 한도 도달 | 시작·목표 좌표 불일치 | 사용 가능한 중간 안내 없음 |',
               '|---|---:|---:|---:|---:|']
    memory = ['| 모델 | 적응형 최대 할당 평균 | 고정 λ=1 최대 할당 평균 | 비율 |',
              '|---|---:|---:|---:|']
    counts = []
    for prefix,label in MODELS:
        for base,name in [('legacy','원본 재실행'),('fixed_1_controlled','같은 엔진의 고정 λ=1'),('fixed_2_controlled','같은 엔진의 고정 λ=2')]:
            r = result(prefix,base);v=r['ratios']
            table.append(f"| {label} | {name} | {v['operation']:.4f} | {v['storage']:.4f} | {v['length']:.4f} |")
            if base in evaluations[prefix]['ci']:
                c=evaluations[prefix]['ci'][base]
                intervals.append('| '+label+' | '+name+' | '+' | '.join(f"[{c[k][0]:.4f}, {c[k][1]:.4f}]" for k in ['operation','storage','length'])+' |')
        for condition in LABELS:
            r=result(prefix,'fixed_1_controlled',condition);v=r['ratios']
            stress.append(f"| {label} | {LABELS[condition]} | {v['operation']:.4f} | {v['storage']:.4f} | {v['length']:.4f} | {r['operation_p95_ratio']:.3f} |")
        for domain,name in [('transpose','좌표 전치'),('scale2','2배 확대')]:
            r=result(prefix,'fixed_1_controlled',domain=domain);v=r['ratios']
            transfers.append(f"| {label} | {name} | {r['pairs']} | {v['operation']:.4f} | {v['storage']:.4f} | {v['length']:.4f} |")
        q=[r for r in evaluations[prefix]['quality'] if r['domain']=='original']
        quality.append(f"| {label} | {sum(r['parse_status']!='ok' for r in q)}/120 | {sum(r['truncated'] for r in q)}/120 | {sum(not r['raw_endpoints_match'] for r in q)}/120 | {sum(r['no_intermediate_advice'] for r in q)}/120 |")
        traced=[r for r in evaluations[prefix]['memory'] if r['domain']=='original']
        a=statistics.mean(r['peak_python_bytes'] for r in traced if r['method']=='adaptive')
        b=statistics.mean(r['peak_python_bytes'] for r in traced if r['method']=='fixed_1_controlled')
        memory.append(f'| {label} | {a:,.0f} bytes | {b:,.0f} bytes | {a/b:.4f} |')
        r=result(prefix,'fixed_1_controlled')
        rows=[x for x in evaluations[prefix]['rows'] if x['method']=='adaptive']
        counts.append(f"- {label}: 적응형 경로 {sum(x['success']=='True' for x in rows)}/{len(rows)}개 유효. 실제 안내에서 고정 λ=1 대비 세 지표가 모두 엄격하게 개선된 문제는 {r['strict_three_improvements']}/120개, 약한 Pareto 개선은 {r['weak_pareto_improvements']}/120개.")

    total_responses=sum(s['responses'] for g in generations.values() for s in g.values())
    total_seconds=sum(s['total_generation_seconds'] for g in generations.values() for s in g.values())
    models=[f"- `{evaluations[p]['metadata']['model']['name']}`: `{evaluations[p]['metadata']['model']['digest']}` · {evaluations[p]['metadata']['model']['details']['quantization_level']}" for p,_ in MODELS]
    findings=[]
    for prefix,label in MODELS:
        values=result(prefix,'fixed_1_controlled')['ratios']
        changes=[f"{name} {abs(100*(1-values[k])):.3f}% {'감소' if values[k]<1 else '증가' if values[k]>1 else '동일'}"
                 for k,name in [('operation','Operation'),('storage','Storage'),('length','경로 길이')]]
        findings.append('- '+label+': 같은 엔진의 고정 λ=1 대비 '+', '.join(changes)+'.')
    text=f'''# 무료 로컬 LLM 실험 보고서

작성일: 2026-10-01 (Asia/Seoul) · 프로토콜: local-free-v2

## 핵심 결과

{chr(10).join(findings)}

세 지표가 동시에 개선됐는지는 각 모델의 방향과 아래 신뢰구간을 함께 확인해야 한다. 원본 재실행 대비 감소만으로 적응형 람다의 효과를 판단하지 않는다.

## 실행 범위

**유료 API 없이 이 Mac에서 실제 모델 출력을 생성해 비교했다. API 사용료는 0달러다.** Apple M3·메모리 8GB 환경에서 Ollama로 Qwen2.5 1.5B와 Llama 3.2 3B를 순차 실행했다. 모델 가중치는 로컬에 저장했으며 GitHub에는 올리지 않는다. 로컬 실행에도 전력·시간·저장 공간은 사용된다.

최종 프로토콜의 실제 응답은 {total_responses}개다. 모델당 검증 20개, 평가 120개, 좌표 전치 10개, 2배 확대 10개를 생성했다. 평가용 지도 60개 모두에서 미리 정한 문제 ID 0·5를 선택했다. 전체 600문제 중 120문제를 사용한 부분 평가다. 두 모델의 평가·전이 비교는 총 6,200회 경로 탐색이다. 최종 생성 기록의 소요 시간 합계는 {total_seconds:.1f}초이며 모델 로딩을 포함한다.

**원 논문의 GPT-3.5·Llama 3 8B 모델을 재현한 실험은 아니다.** 다른 크기의 양자화 모델, Ollama 템플릿, 추가 출력 지시를 사용했다. 따라서 논문 수치보다 우수하다고 주장할 수 없다.

모델 식별 정보:

{chr(10).join(models)}

## 동일 안내에서의 비교

각 문제에서 LLM 출력을 한 번 생성하고 원본 재실행·A*·고정 λ=1·고정 λ=2·적응형 람다에 같은 좌표를 제공했다. 람다는 합성 학습 실험에서 선택한 설정을 그대로 사용했으며 실제 LLM 결과로 재조정하지 않았다. 아래는 **변형하지 않은 실제 모델 안내**의 지표 비율 기하평균이다. 1보다 작으면 적응형 방식이 비교 대상보다 낮은 값을 보인다.

{chr(10).join(table)}

![실제 로컬 모델 비교](../results/local_llm/comparison_v2.png)

원본과의 비교에는 큐 처리·노드 재개방·목표 전환 등의 공통 구현 변경도 포함된다. **람다 적응 효과는 같은 엔진의 고정 가중치 비교군으로 판단해야 한다.** 모델의 성능 차이와 탐색기의 성능 차이를 구분한다.

{chr(10).join(counts)}

모든 문제에서 세 지표가 동시에 개선되는지는 위 개별 문제 수로 확인해야 한다. 생성 실패를 제외한 성공 사례만 골라내지 않았으며, 안내가 없는 상태에서도 탐색을 진행했다. 경로 탐색 성공률과 LLM의 안내 생성 성공률은 다른 지표다.

## 신뢰구간

평가 지도 60개를 묶음 단위로 2,000회 재표집했다. 같은 지도에 속한 두 문제의 상관을 유지한 비율의 95% 구간이다. 구간이 1을 포함하면 해당 지표의 개선을 확정적으로 주장하지 않는다. 모델을 여러 생성 시드로 반복한 구간은 아니다.

{chr(10).join(intervals)}

## 강건성 검사

실제 출력에 순서 반전·절반 누락·절반 무작위 교체를 적용하거나 안내를 모두 제거했다. 추가 모델 호출 없이 동일 출력의 변형을 비교한다. 이 변형은 실제 LLM 오류 분포를 대표하는 데이터가 아니라 통제된 스트레스 조건이다. 아래 비교 기준은 **같은 엔진의 고정 λ=1**이다. P95는 문제별 Operation 비율의 95번째 백분위수로, 평균과 별도로 일부 문제의 악화를 보여준다.

{chr(10).join(stress)}

평균 개선은 최악의 경우의 확장 횟수 제한이나 모든 문제의 우위를 보장하지 않는다. 기본 마지막 단계는 원본 관례인 `g + 2*h_goal`이며, 안내 실패 시에도 일반 A*의 최적성 보장은 없다. A* 비교군은 별도로 `g + h_goal` 순서를 사용한다.

## 일반성 검사

동일한 람다 설정을 두 모델에 적용했다. 평가용 지도 중 처음 10개의 문제 ID 0을 좌표 전치·2배 확대하고, 변형한 입력마다 **새로운 LLM 안내**를 생성했다. 아래 비교 기준도 고정 λ=1이다.

{chr(10).join(transfers)}

두 모델과 작은 전이 검사에서의 결과다. 전이 입력은 기존 지도의 변형이므로 완전히 새로운 지도 분포에 대한 검증은 아니다. 대형 모델·외부 지도·3차원·동적 환경으로 일반화됐다고 판단할 수 없다.

## 안내 자체의 오류와 메모리

아래는 원본 좌표 환경의 120개 모델 응답이다. 좌표가 정수 목록으로 읽혔다고 해서 올바른 경유점이라는 뜻은 아니다. 시작·목표 불일치에는 파싱 실패도 포함되므로 열을 합산하지 않는다.

{chr(10).join(quality)}

파싱 실패는 빈 안내로 변환했다. 정상적으로 읽은 좌표에는 원본의 장애물·범위 필터를 동일하게 적용했다. 모델 답변을 최적 경로로 보정하거나 성공한 응답만 재선택하지 않았다. 실제 목표 좌표는 입력에서 유지하며, LLM 경유점은 강제 제약이 아니다. Dijkstra는 모델 응답을 생성한 **뒤** 평가 정답과 반환 경로의 검증에만 사용했다.

Storage는 실제 바이트가 아니라 상태 수다. 별도의 Python 할당 추적 결과는 다음과 같다.

{chr(10).join(memory)}

공유 그래프 캐시·LLM·네이티브 메모리는 위 측정에 포함되지 않는다. 상태 수 감소를 전체 시스템 메모리 감소로 해석하지 않는다. 경로 탐색 시간에도 LLM 생성과 그래프 준비 시간은 포함되지 않는다.

## 프롬프트 선택과 실패 기록 보존

초기 v1은 원본 `standard_gpt` 프롬프트에 격자 범위만 추가해 사용했다. 검증 20문제에서 Qwen은 파싱 실패 4개, Llama는 경로 대신 코드를 생성해 20개 모두 출력 한도에 도달하고 파싱에 실패했다. v1의 응답은 삭제하지 않았다. 자동으로 시작됐던 Llama v1 평가 생성은 첫 4개 응답 후 중단했으며, 이 불완전 기록도 보존했다. v1 Qwen 평가도 최종 비교 표에는 합치지 않는다.

검증 결과를 바탕으로 두 모델에 동일하게 “정수 좌표 최대 8개, 시작·목표 포함, 코드·설명 없이 좌표만 반환”이라는 시스템 지시를 추가했다. v2 검증에서는 Qwen 20개 중 20개, Llama 20개 중 19개가 파싱됐다. 이 지시와 평가 규칙을 고정한 커밋 `3fae8ca` 이후 최종 v2 평가를 실행했다. 각 생성 단계는 첫 응답 전에 `plan.json`을 기록한다.

최종 실험에서는 형식 오류·출력 잘림을 이유로 재시도하지 않았다. 원문 응답, 입력, 프롬프트, 시스템 지시, 모델 해시, 템플릿 해시, 생성 설정, 토큰 수, 소요 시간을 저장했다. 온도 0·고정 시드도 서로 다른 환경에서 완전히 같은 출력을 보장하지 않으므로 저장된 캐시를 이용한 재실행이 기준이다.

## 재현 방법

Ollama와 모델을 설치한 뒤 저장소 최상위 폴더에서 실행한다. 아래 모델은 Ollama 로컬 라이브러리의 [Qwen2.5 1.5B](https://ollama.com/library/qwen2.5:1.5b)와 [Llama 3.2 3B](https://ollama.com/library/llama3.2)다. 생성기는 로컬 주소만 사용하고 클라우드 모델·리디렉션을 거부한다.

```bash
ollama pull qwen2.5:1.5b
ollama pull llama3.2:3b
.venv/bin/python -m benchmarks.local_llm --model qwen2.5:1.5b --split test --output results/local_llm/qwen_test_v2
.venv/bin/python -m benchmarks.local_llm --model llama3.2:3b --split test --output results/local_llm/llama_test_v2
.venv/bin/python -m benchmarks.local_evaluate --inputs results/local_llm/qwen_test_v2/waypoints.json results/local_llm/qwen_transpose_v2/waypoints.json results/local_llm/qwen_scale2_v2/waypoints.json --output results/local_llm/qwen_evaluation_v2
.venv/bin/python -m benchmarks.local_evaluate --inputs results/local_llm/llama_test_v2/waypoints.json results/local_llm/llama_transpose_v2/waypoints.json results/local_llm/llama_scale2_v2/waypoints.json --output results/local_llm/llama_evaluation_v2
.venv/bin/python -m benchmarks.local_report
```

위 생성 명령은 평가용 원본 좌표 환경을 재생성한다. 좌표 전치 생성에는 `--domain transpose --max-maps 10 --sample-ids 0`, 2배 확대 생성에는 `--domain scale2 --max-maps 10 --sample-ids 0`을 추가하고 출력 폴더도 각 환경에 맞게 지정한다. 검증에는 `--split validation --max-maps 2 --sample-ids 0 1 2 3 4 5 6 7 8 9`를 사용한다. 저장한 계획과 현재 모델·소스 해시가 다르면 새 폴더를 지정해야 한다. 다운로드·로컬 생성 없이 저장된 JSON만으로 비교 평가를 다시 실행할 수도 있다.

원시 데이터는 `results/local_llm/`, 고정 프로토콜은 `benchmarks/local_protocol.json`에 있다. 이전 합성 실험과 수치는 [기존 연구 보고서](adaptive_lambda_research.md)에 별도로 유지한다.
'''
    (ROOT/'docs/local_llm_research.md').write_text(text)
    aggregate = {'api_cost_usd':0, 'final_responses':total_responses,
                 'final_evaluation_searches':6200, 'generation_seconds_including_load':total_seconds,
                 'actual_results':{p:{b:result(p,b) for b in ['legacy','fixed_1_controlled','fixed_2_controlled']} for p,_ in MODELS},
                 'bootstrap_95':{p:evaluations[p]['ci'] for p,_ in MODELS}}
    (root/'aggregate_v2.json').write_text(json.dumps(aggregate,indent=2))
    print(ROOT/'docs/local_llm_research.md')


if __name__ == '__main__':
    main()
