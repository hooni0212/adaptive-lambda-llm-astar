# 원본 출처와 연구 범위

이 연구 저장소는 [SilinMeng0510/llm-astar](https://github.com/SilinMeng0510/llm-astar)를 기반으로 하며, 처음 복제한 버전은 커밋 `9414ed8`입니다. 원본의 MIT 라이선스와 저자 표시는 [LICENSE](LICENSE)에 유지하고, 원본 커밋 이력도 보존했습니다. 원본 README는 [docs/upstream_readme.md](docs/upstream_readme.md)에 영문으로 보존했습니다.

원본 논문: Silin Meng, Yiwei Wang, Cheng-Fu Yang, Nanyun Peng, Kai-Wei Chang, **LLM-A*: Large Language Model Enhanced Incremental Heuristic Search on Path Planning**, Findings of EMNLP 2024, [arXiv:2407.02511](https://arxiv.org/abs/2407.02511).

적응형 람다 경로 탐색기, 오프라인 비교 실험 도구, 테스트, 설정 파일, 연구 보고서는 이 저장소에서 추가한 작업입니다. 논문 저자들이 배포한 공식 버전이 아니라 독립적으로 확장한 연구 프로젝트입니다.

기존 실험에는 최적 경로에서 추출한 합성 경유점과 오류를 넣은 경유점을 사용했습니다. AI 도우미가 경유점을 작성한 12개 예비 실험은 별도로 표시했습니다. 후속 무료 실험에서는 로컬 Qwen2.5 1.5B·Llama 3.2 3B의 실제 출력과 그 오류 변형을 별도로 평가했습니다. 이 결과는 논문의 GPT-3.5·LLAMA3 출력을 재현하지 않으며, 논문에 보고된 성능보다 우수하다는 결론을 뒷받침하지 않습니다.
