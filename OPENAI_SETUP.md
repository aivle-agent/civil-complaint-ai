# OpenAI GPT-4o Mini Integration Guide

## 개요

이 프로젝트는 OpenAI GPT-4o mini를 사용하여 민원 처리 시스템을 구현했습니다. 각 노드는 LLM을 활용하여 다음을 생성합니다:
- `strategy`: 답변 전략
- `draft_answer`: 답변 초안  
- `verification_feedback`: 검증 피드백
- `final_answer`: 최종 답변

## 설정 방법

### 1. 환경 변수 설정

`.env` 파일을 생성하고 OpenAI API 키를 설정하세요:

```bash
# .env.example 파일을 복사하여 .env 파일 생성
cp .env.example .env
```

`.env` 파일에 API 키를 입력:

```
OPENAI_API_KEY=your-actual-openai-api-key-here
```

> ⚠️ **중요**: `.env` 파일은 `.gitignore`에 포함되어 있어 Git에 커밋되지 않습니다.

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

## 사용 방법

### 기본 실행

```bash
python main.py
```

### 테스트 실행

모든 테스트를 실행하고 state 검증 확인:

```bash
pytest -v
```

특정 노드 테스트만 실행:

```bash
# 전략 생성 노드 테스트
pytest tests/test_generate_strategy_node.py -v

# 답변 초안 노드 테스트
pytest tests/test_draft_reply_node.py -v

# 검증 노드 테스트
pytest tests/test_verify_reply_node.py -v

# 최종 답변 노드 테스트
pytest tests/test_final_answer_node.py -v
```

## 노드별 LLM 사용

### 1. Generate Strategy Node (`generate_strategy_node.py`)
- **기능**: 민원 질문에 대한 답변 전략 수립
- **LLM 모델**: GPT-4o mini
- **Temperature**: 0.7
- **출력**: `strategy` (답변 전략)

### 2. Draft Reply Node (`draft_reply_node.py`)
- **기능**: 전략을 바탕으로 답변 초안 작성
- **LLM 모델**: GPT-4o mini
- **Temperature**: 0.7
- **출력**: `draft_answer` (답변 초안)
- **특징**: 이전 검증 피드백을 반영하여 재작성 가능

### 3. Verify Reply Node (`verify_reply_node.py`)
- **기능**: 답변 초안의 품질 검증
- **LLM 모델**: GPT-4o mini
- **Temperature**: 0.3 (일관성 있는 검증을 위해 낮은 temperature)
- **출력**: `verification_feedback`, `is_verified`, `retry_count`
- **판정 기준**:
  - 질문에 대한 직접적이고 명확한 답변
  - 정중하고 친절한 어조
  - 구체적인 절차나 방법 안내
  - 오류나 부적절한 내용 유무

### 4. Final Answer Node (`final_answer_node.py`)
- **기능**: 검증된 답변을 최종 정리 및 다듬기
- **LLM 모델**: GPT-4o mini
- **Temperature**: 0.5
- **출력**: `final_answer` (최종 답변)
- **특징**: 인사말 추가, 문장 다듬기, 구조 정리

## State 검증

모든 테스트는 다음을 검증합니다:

1. **필수 키 존재 확인**: 각 노드가 올바른 state 키를 반환하는지 확인
2. **데이터 타입 검증**: 반환된 값이 올바른 타입인지 확인
3. **값의 유효성**: 반환된 값이 비어있지 않고 의미있는 내용인지 확인
4. **예상치 못한 키 검증**: 오직 예상된 키만 반환되는지 확인

### 테스트 예시 출력

```
✓ State validation passed: strategy properly generated and stored
✓ State validation passed: draft_answer properly generated and stored
✓ State validation passed: verification PASSED with proper state updates
✓ State validation passed: final_answer properly generated and stored
```

## Fallback 메커니즘

모든 노드는 LLM 호출 실패 시 fallback 로직을 포함합니다:

- **API 키 오류**: 명확한 에러 메시지 제공
- **네트워크 오류**: mock 데이터로 대체
- **기타 오류**: 기본 답변으로 처리 계속

## 프로젝트 구조

```
civil-complaint-ai/
├── .env.example          # 환경 변수 템플릿
├── src/
│   ├── config.py         # API 키 설정 관리
│   ├── models/
│   │   └── state.py      # State 타입 정의
│   └── nodes/
│       ├── generate_strategy_node.py
│       ├── draft_reply_node.py
│       ├── verify_reply_node.py
│       └── final_answer_node.py
└── tests/
    ├── test_generate_strategy_node.py
    ├── test_draft_reply_node.py
    ├── test_verify_reply_node.py
    └── test_final_answer_node.py
```

## 문제 해결

### API 키 관련 오류

```
ValueError: OPENAI_API_KEY not found in environment variables
```

**해결 방법**: `.env` 파일에 올바른 API 키가 설정되어 있는지 확인

### LLM 호출 실패

**증상**: "Falling back to mock..." 메시지 출력

**해결 방법**:
1. 인터넷 연결 확인
2. OpenAI API 할당량 확인
3. API 키 유효성 확인

## 추가 정보

- **OpenAI API 문서**: https://platform.openai.com/docs/
- **GPT-4o mini 모델 정보**: https://platform.openai.com/docs/models/gpt-4o-mini
- **LangChain 문서**: https://python.langchain.com/
