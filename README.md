# LLM 기반 표 데이터 증강을 통한 신용카드 연체 예측

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.53+-orange.svg)](https://huggingface.co/transformers/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 프로젝트 개요

이 프로젝트는 대형 언어 모델(LLM)을 활용하여 신용카드 연체 예측을 위한 표 형태 데이터 증강을 구현한 연구입니다. Meta의 Llama-3-8B 모델을 파인튜닝하여 현실적인 통계적 특성을 유지하면서 개인정보를 보호하는 합성 신용카드 고객 데이터를 생성합니다.

### 🎯 주요 성과
- Llama-3-8B 모델을 신용카드 연체 데이터셋으로 파인튜닝
- 4,000개 이상의 합성 고객 레코드 생성 (연체 2,000개 + 정상 2,000개)
- 원본 데이터 분포와 일치하는 현실적인 데이터 분포 달성
- 구조화된 표 데이터 생성에서 LLM의 가능성 입증

## 🏗️ 시스템 아키텍처

```
원본 데이터 → 데이터 전처리 → 파인튜닝 → 추론 → 합성 데이터
     ↓            ↓            ↓        ↓         ↓
   신용카드     텍스트 형식    Llama-3-8B  생성      증강된
   데이터셋     변환          학습       파이프라인  데이터셋
```

## 📊 데이터셋

**신용카드 연체 데이터셋** (대만, 2005년)
- **피처**: 신용한도, 인구통계, 결제이력을 포함한 14개 속성
- **타겟**: 이진 분류 (연체 vs 정상 결제)
- **크기**: 학습용 1,000개 샘플

### 피처 카테고리:
- **인구통계**: 성별, 교육수준, 결혼상태, 나이
- **금융정보**: 신용한도, 청구서 금액, 결제 금액
- **행동패턴**: 상환 상태 (2005년 4월-6월)

## 🔧 구현 파이프라인

### 1단계: 데이터 전처리 (`Step_1. tab_to_list.ipynb`)
- 표 형태 데이터를 자연어 형식으로 변환
- LLM 학습을 위한 구조화된 텍스트 템플릿 생성
- 대화형 프롬프트 생성

```python
# 변환 예시
"Amount of given credit in NT dollars: 100000
Gender: 2
Education: 1
..."
```

### 2단계: 학습 데이터 준비 (`Step_2. finetuning_data.ipynb`)
- Llama-3 채팅 템플릿 형식으로 데이터 포맷팅
- system/user/assistant 대화 구조 생성
- 계층화된 훈련/테스트 분할 구현

### 3단계: 모델 파인튜닝 (`Step_3. fine tuning_code.ipynb`)
- **기본 모델**: Meta-Llama-3-8B-Instruct
- **기법**: 4비트 양자화를 통한 LoRA (Low-Rank Adaptation)
- **학습**: 코사인 학습률 스케줄링으로 50 에포크
- **하드웨어**: 혼합 정밀도 학습으로 GPU 최적화

#### 학습 설정:
```python
peft_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    bias="none", task_type="CAUSAL_LM"
)
```

### 4단계: 데이터 생성 (`Step_4. inference.ipynb`)
- 합성 고객 프로필 생성
- 연체 및 비연체 시나리오 모두 생성
- max_new_tokens=200-250으로 제어된 생성 구현

### 5단계: 후처리 (`Step_5. inference_post_processing.ipynb`)
- 정규표현식을 사용하여 생성된 텍스트에서 구조화된 데이터 추출
- 데이터 형식 및 일관성 검증
- 분석을 위한 Excel 내보내기

## 🚀 빠른 시작

### 필수 요구사항
```bash
pip install torch transformers datasets accelerate
pip install peft trl bitsandbytes
pip install pandas numpy scikit-learn
```

### 사용법

1. **데이터 준비**:
```python
# 표 형태 데이터 로드 및 전처리
fine_data = pd.read_pickle('./fine_data_last.pkl')
```

2. **모델 학습**:
```python
# Llama-3-8B 파인튜닝
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    peft_config=peft_config,
    # ... 기타 매개변수
)
trainer.train()
```

3. **합성 데이터 생성**:
```python
# 새로운 샘플 생성
generated_outputs = model.generate(**input, max_new_tokens=200)
```

## 📈 결과

### 생성된 데이터 품질
- **규모**: 4,000개 이상의 합성 레코드
- **다양성**: 연체/비연체 케이스의 균형잡힌 표현
- **현실성**: 원본 데이터의 통계적 특성 유지
- **형식 일관성**: 95% 이상의 성공적인 구조화된 데이터 추출

### 생성 결과 예시
```
Amount of given credit in NT dollars: 150000
Gender: 2
Education: 1
Marital status: 2
Age: 32
Repayment status in June, 2005: 0
...
```

## 🛠️ 기술 스택

- **언어**: Python 3.10+
- **ML 프레임워크**: PyTorch 2.7+
- **LLM 라이브러리**: Hugging Face Transformers
- **파인튜닝**: PEFT (Parameter Efficient Fine-Tuning)
- **최적화**: BitsAndBytesConfig를 통한 4비트 양자화
- **데이터 처리**: Pandas, NumPy

## 📁 프로젝트 구조

```
llm-tab-aug/
├── Step_1. tab_to_list.ipynb           # 데이터 전처리
├── Step_2. finetuning_data.ipynb       # 학습 데이터 준비
├── Step_3. fine tuning_code.ipynb      # 모델 파인튜닝
├── Step_4. inference.ipynb             # 데이터 생성
├── Step_5. inference_post_processing.ipynb  # 출력 후처리
├── data/                               # 데이터셋 파일
├── results/                            # 생성된 결과
└── README.md                          # 프로젝트 문서
```

## 🔬 연구 활용

### 잠재적 활용 분야:
- **데이터 프라이버시**: 민감한 금융 데이터를 위한 합성 데이터셋 생성
- **데이터 부족**: ML 모델 학습을 위한 소규모 데이터셋 증강
- **시나리오 테스트**: 스트레스 테스트를 위한 다양한 고객 프로필 생성
- **연구**: 구조화된 데이터 생성에서 LLM 역량 연구

### 향후 개선사항:
- 다중 모달 데이터 생성 (수치형 + 범주형)
- 특정 제약 조건 기반 조건부 생성
- 차별적 프라이버시 기법과의 통합
- 실시간 데이터 증강 파이프라인

## 📊 평가 지표

- **통계적 충실도**: 원본 데이터와의 분포 유사성
- **다양성**: 생성된 샘플의 고유성
- **유용성**: 하위 ML 모델의 성능
- **프라이버시**: k-익명성 및 l-다양성 측정

## 🤝 기여

기여를 환영합니다! Pull Request를 자유롭게 제출해 주세요. 주요 변경사항의 경우, 먼저 이슈를 열어 변경하고자 하는 내용에 대해 논의해 주세요.

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 라이선스가 부여됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 🙏 감사의 말

- Llama-3 모델을 제공한 Meta AI
- transformers 라이브러리를 제공한 Hugging Face
- 대만 신용카드 연체 데이터셋 기여자들

## 📞 연락처

이 프로젝트에 대한 질문이 있으시면 이슈를 열거나 관리자에게 연락해 주세요.

---
**키워드**: LLM, 표 데이터 증강, 신용 위험, Llama-3, 파인튜닝, 합성 데이터 생성, 금융 ML