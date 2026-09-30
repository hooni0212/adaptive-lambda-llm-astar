# 문서 안내

이 폴더에는 연구 보고서와 원본 프로젝트에서 가져온 웹페이지 자료가 있습니다.

| 문서 | 내용 |
|---|---|
| [무료 로컬 LLM 보고서](local_llm_research.md) | 실제 모델 안내와 강건성·전이·메모리 검증 |
| [기존 연구 보고서](adaptive_lambda_research.md) | 적응형 람다 구현, 실험 설계, 성능 결과, 강건성·일반성 검증과 한계 |
| [원본 README](upstream_readme.md) | 출처 확인을 위해 보존한 원본 프로젝트의 영문 안내 |
| [프로젝트 시작 안내](../README.md) | 설치, 실행 예제, 실험 재현 방법 |
| [실험 결과 안내](../results/README.md) | 결과 파일 구성과 압축한 탐색 기록을 푸는 방법 |
| [출처와 연구 범위](../NOTICE.md) | 원본 저자, 논문, 이 저장소에서 추가한 작업의 범위 |

`index.html`과 `static/`은 원본에서 가져온 논문 소개 웹페이지 자료입니다. 현재 적응형 람다 연구 결과는 위 연구 보고서와 저장소의 메인 README에서 확인할 수 있습니다.

## 원본 학술 프로젝트 웹페이지 템플릿

아래는 원본에 포함된 템플릿 안내를 한국어로 옮긴 내용입니다.

이 템플릿으로 제작한 프로젝트 페이지 예시:

- https://vision.huji.ac.il/spectral_detuning/
- https://vision.huji.ac.il/podd/
- https://dreamix-video-editing.github.io
- https://vision.huji.ac.il/conffusion/
- https://vision.huji.ac.il/3d_ads/
- https://vision.huji.ac.il/ssrl_ad/
- https://vision.huji.ac.il/deepsim/

### 템플릿 사용 방법

새 프로젝트에서 템플릿을 사용하려면 GitHub의 `Use this Template`을 선택합니다. HTML은 내용, CSS는 스타일을 담당합니다. `index.html`에서 필요한 구성 요소를 편집하고 사용하지 않는 요소는 주석 처리합니다.

`static/images/`의 `favicon.ico`를 프로젝트에 맞는 이미지로 교체합니다. 원본 안내는 기본 이미지가 템플릿 제작자의 이미지 또는 소속 기관 아이콘일 수 있음을 설명합니다.

### 제공하는 구성 요소

- 소개 영상
- 이미지 슬라이드
- YouTube 영상 삽입
- 영상 슬라이드
- PDF 포스터
- BibTeX 인용 정보

### 편집 참고 사항

- `index.html`의 주석에는 교체할 부분에 대한 설명이 있습니다.
- `meta` 태그는 검색엔진과 공유 미리보기에 사용할 논문 정보를 담습니다.
- 원본 안내는 이미지·영상 해상도로 대체로 1920~2048 수준을 제안합니다. 더 높은 해상도는 불러오는 시간을 늘릴 수 있습니다.
- 이미지와 영상은 압축해 페이지가 빠르게 열리도록 합니다. 원본 안내는 이미지 도구로 [TinyPNG](https://tinypng.com)를 소개하며, 영상은 용량과 화질을 함께 고려하도록 설명합니다.
- 원본 안내는 10MB보다 큰 영상의 경우 YouTube에 올려 삽입하는 방법을 제안합니다.
- 원본 안내는 방문 경로 분석 도구로 [Statcounter](https://statcounter.com)를 소개합니다.
- 프로젝트 페이지는 GitHub Pages로 호스팅할 수 있습니다.
- 템플릿에 대한 제안은 원본 제작자의 이슈나 [연락처 페이지](https://pages.cs.huji.ac.il/eliahu-horwitz/)를 이용할 수 있습니다.

### 원본 템플릿의 출처

프로젝트 페이지의 일부는 [Nerfies](https://nerfies.github.io/) 페이지를 참고해 제작했습니다.

### 웹페이지 템플릿 라이선스

원본 안내에 명시된 웹페이지 템플릿의 라이선스는 [Creative Commons 저작자표시-동일조건변경허락 4.0 국제 라이선스(CC BY-SA 4.0)](http://creativecommons.org/licenses/by-sa/4.0/)입니다. 저장소 코드의 MIT 라이선스 안내는 [LICENSE](../LICENSE)를 확인하세요.

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons 라이선스" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a>
