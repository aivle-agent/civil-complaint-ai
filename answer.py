# !pip install chromadb

import zipfile

zip_path = "/content/drive/MyDrive/chroma_lawdb.zip"
extract_path = "/content/complaint_system_AI/chroma_lawdb"   
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print(" Chroma ZIP extracted:", extract_path)


import os
import json
import math
import logging
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import chromadb


# 0. 경로 및 기본 설정


CHROMA_DIR = "/content/complaint_system_AI/chroma_lawdb/chroma_lawdb"
# MODEL_DIR = "/content/complaint_system_AI/finetuned-tinyllama-law"
MODEL_DIR = "/content/drive/MyDrive/finetuned_models/tinyllama-law"

NOVEL_PATH = "/content/complaint_system_AI/export/novel_test.jsonl"
MEMORY_PATH = "/content/complaint_system_AI/bandit_memory.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




logger.info("[INIT] Loading ChromaDB...")
client = chromadb.PersistentClient(path=CHROMA_DIR)

collections = client.list_collections()
if not collections:
    raise RuntimeError(f"[RAG] No collections found in {CHROMA_DIR}")

# 가장 첫 번째 컬렉션 사용(이미 구축된 DB라고 가정)
collection = client.get_collection(collections[0].name)
logger.info(f"[RAG] Using collection: {collections[0].name}")

logger.info("[INIT] Loading embedding model for RAG...")
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def rag_retrieve(question: str, top_k: int = 5) -> Tuple[str, Dict[str, Any]]:
    """
    질문 임베딩 → Chroma에서 상위 top_k 문서 검색 → 컨텍스트 문자열 조립
    """
    q_emb = embedder.encode(question, convert_to_numpy=True).tolist()

    res = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
    )

    docs = res.get("documents", [[]])[0] if res.get("documents") else []
    context_parts = []
    for i, d in enumerate(docs):
        if not isinstance(d, str):
            continue
        d = d.strip()
        if not d:
            continue

        context_parts.append(f"[자료 {i+1}]\n{d[:800]}")

    context_text = "\n\n".join(context_parts)
    return context_text, res


#  2. Finetuned TinyLlama 로드 (Generator)


logger.info(f"[INIT] Loading finetuned model from {MODEL_DIR} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)
model.eval()
logger.info("[INIT] Model ready.")


MAX_CONTEXT = getattr(model.config, "max_position_embeddings", 2048)
MAX_NEW_TOKENS = 256  
MAX_PROMPT_TOKENS = max(128, MAX_CONTEXT - MAX_NEW_TOKENS)  

logger.info(f"[LIMIT] MAX_CONTEXT={MAX_CONTEXT}, "
            f"MAX_PROMPT_TOKENS={MAX_PROMPT_TOKENS}, "
            f"MAX_NEW_TOKENS={MAX_NEW_TOKENS}")

# 3. 답변 스타일 Few-shot 템플릿


FEW_SHOT_TEMPLATE = """
아래는 실제 공공 민원에 대한 모범 답변 예시입니다.

[예시 1]

검토내용
1. 안녕하십니까. 귀하께서 국민제안을 통해 질의하신 민원에 대한 답변드립니다.

2. 귀하께서 올려주신 민원 내용은 도로 교차로 신호 운영을 변경해 달라는 요청으로 이해하였습니다.

3. 가. 먼저, 교통안전시설에 관심을 가져 주셔서 감사합니다. 귀하께서 제기하신 사항에 대해 검토한 결과는 다음과 같습니다.
나. 관계 기관 및 신호 연동업체와 협의한 결과, 야간 시간대 교통량, 보행자 통행량, 주변 사고 이력 등을 종합적으로 고려할 필요가 있어 단계적으로 신호체계를 조정하는 것이 타당한 것으로 판단됩니다.
다. 이에 따라 시에서는 24:00시부터 07:00시까지 점멸신호 운영으로 변경하는 방안을 우선 시행하고, 시행 후 교통 상황을 모니터링하여 추가 조정 여부를 검토할 예정임을 알려드립니다.

4. 귀하의 질문에 만족스러운 답변이 되었기를 바라며, 답변에 대한 추가 설명이 필요하신 경우 경찰서 교통과로 연락 주시면 성실히 안내해 드리겠습니다. 귀하와 가정에 항상 건강과 행복이 함께하시길 기원합니다. 감사합니다.


[예시 2]

검토내용
1. 안녕하십니까. 귀하께서 민원으로 제기하신 시설 설치 관련 사항에 대해 답변드립니다.

2. 귀하의 민원은 동 인근에 보행자 안전을 위한 안전시설을 추가로 설치해 달라는 요청으로 이해하였습니다.

3. 가. 먼저 지역 안전에 관심을 가져 주신 점에 감사드립니다.
나. 본 사항에 대해 현장 점검 및 관련 부서 협의를 실시한 결과, 해당 구간은 출퇴근 시간대 보행량이 많고 차량 통행 속도 역시 높은 편으로, 추가 안전시설이 필요한 것으로 검토되었습니다.
다. 이에 따라 금년 상반기 중 예산 범위 내에서 보행자 안내표지 및 과속방지시설 설치를 우선 추진하고, 이후 교통량 및 안전사고 발생 추이를 분석하여 추가 시설 설치 여부를 검토할 예정입니다.

4. 본 답변이 귀하의 궁금증 해소에 도움이 되었기를 바라며, 보다 자세한 안내가 필요하신 경우 구청 교통행정과로 문의하여 주시기 바랍니다. 감사합니다.

위 예시를 참고하여, 아래 [민원 내용]에 대해 유사한 형식과 톤으로 답변을 작성하십시오.
- "검토내용"으로 시작
- 1,2,3,4 번 항목 구조 유지
- 3번 항목에서 검토 과정과 결과를 구체적으로 설명
- 4번 항목에서 문의 안내 및 감사 인사 포함
"""


def build_prompt(question: str, rag_context: str) -> str:
    """
    Few-shot 예시 + 참고자료 + 민원 내용 → 하나의 프롬프트로 구성
    """
    prompt = (
        FEW_SHOT_TEMPLATE
        + "\n\n[참고자료]\n"
        + (rag_context if rag_context else "관련 법령 및 유사 민원 자료가 존재하지 않습니다.\n")
        + "\n\n[민원 내용]\n"
        + question.strip()
        + "\n\n[답변 작성]\n검토내용\n"
    )
    return prompt


# 4. Bandit 메모리 및 보상 정의


# 메모리 로딩
if os.path.exists(MEMORY_PATH):
    with open(MEMORY_PATH, "r", encoding="utf-8") as f:
        MEMORY: List[Dict[str, Any]] = json.load(f)
else:
    MEMORY: List[Dict[str, Any]] = []

# 3가지 전략 (temperature, top_p)
STRATEGIES = [
    {"temp": 0.6, "top_p": 0.9},    # 보수적 / 안정형
    {"temp": 0.8, "top_p": 0.9},    # 기본형
    {"temp": 1.0, "top_p": 0.95},   # 창의형
]


def compute_success(reward: float) -> int:
    """보상 점수가 일정 기준 이상이면 성공으로 간주"""
    return 1 if reward >= 0.7 else 0


def save_memory(question: str, answer: str, reward: float, strategy_index: int):
    MEMORY.append({
        "q": question,
        "a": answer,
        "reward": reward,
        "success": compute_success(reward),
        "strategy": strategy_index,
    })
    if len(MEMORY) > 2000:
        del MEMORY[:-2000]

    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(MEMORY, f, ensure_ascii=False, indent=2)


def choose_strategy() -> Tuple[int, Dict[str, float]]:
    """
    Thompson Sampling + UCB1 하이브리드 Bandit
    - 최근 메모리 기준으로 각 전략의 성공 확률을 Beta 분포로 추정
    - UCB1 탐색 보정을 더해 최종 점수 계산
    """
    if len(MEMORY) < len(STRATEGIES) * 2:
        return 1, STRATEGIES[1]

    # 최근 일부만 사용해 최근성 반영
    recent = MEMORY[-200:]

    successes = [0] * len(STRATEGIES)
    failures = [0] * len(STRATEGIES)
    counts = [0] * len(STRATEGIES)

    for m in recent:
        idx = m["strategy"]
        counts[idx] += 1
        if m["success"] == 1:
            successes[idx] += 1
        else:
            failures[idx] += 1

    # 1) Thompson sampling
    theta = []
    for s, f in zip(successes, failures):
        sample = np.random.beta(s + 1, f + 1)
        theta.append(sample)

    # 2) UCB1
    total = sum(counts) if sum(counts) > 0 else 1
    ucb = []
    for c in counts:
        if c == 0:
            ucb.append(float("inf"))
        else:
            ucb.append(math.sqrt(2 * math.log(total) / c))

    # 3) Hybrid score
    hybrid_score = np.array(theta) + 0.3 * np.array(ucb)

    best_idx = int(np.argmax(hybrid_score))
    return best_idx, STRATEGIES[best_idx]


# 5. 간단 CoT 스타일 검증기


def verify_answer(question: str, answer: str) -> Tuple[float, str]:
    """
    Heuristic 기반 CoT 스타일 검증:
    - 서두 인사, 민원 내용 재진술, 검토 과정, 조치 계획, 마무리 인사, 길이 등을 점수화
    - 0 ~ 1 사이 reward 반환 + 검증 사유 텍스트
    """
    reasoning = []
    score = 0.0

    # 1) 인사
    if "안녕하십니까" in answer:
        score += 0.15
        reasoning.append("- 인사 문구 포함: OK")
    else:
        reasoning.append("- 인사 문구 없음: 감점")

    # 2) 민원 내용 재진술
    if ("민원 내용" in answer) or ("귀하께서 올려주신 민원" in answer):
        score += 0.2
        reasoning.append("- 민원 내용 재진술: OK")
    else:
        reasoning.append("- 민원 내용 재진술 부족")

    # 3) 검토/조치 설명
    if any(k in answer for k in ["검토한 결과", "검토 결과", "검토한 바", "검토한 의견"]):
        score += 0.2
        reasoning.append("- 검토 과정/결과 설명: OK")
    else:
        reasoning.append("- 검토 과정 설명 부족")

    # 4) 조치 계획 / 향후 계획
    if any(k in answer for k in ["조치하겠습니다", "추진할 예정", "변경할 예정", "실시할 예정", "검토할 예정"]):
        score += 0.15
        reasoning.append("- 조치 계획/향후 계획 제시: OK")
    else:
        reasoning.append("- 조치 계획이 명확하지 않음")

    # 5) 법령/근거 언급
    if any(k in answer for k in ["법", "조례", "규정", "근거"]):
        score += 0.1
        reasoning.append("- 법령/근거 언급 있음: OK")
    else:
        reasoning.append("- 법령 또는 근거 언급 없음 (경미한 감점)")

    # 6) 마무리 인사
    if any(k in answer for k in ["질문에 만족스러운 답변", "도움이 되었기를 바라며", "기원합니다", "감사합니다."]):
        score += 0.1
        reasoning.append("- 마무리 인사 및 배려 문구 포함: OK")
    else:
        reasoning.append("- 마무리 인사가 다소 부족함")

    # 7) 길이 체크
    length = len(answer)
    if length < 200:
        score += 0.05
        reasoning.append(f"- 답변 길이 {length}자: 다소 짧음 (소폭 가점)")
    elif length < 1000:
        score += 0.1
        reasoning.append(f"- 답변 길이 {length}자: 적절한 분량")
    else:
        score += 0.08
        reasoning.append(f"- 답변 길이 {length}자: 다소 긴 편이지만 허용")

    reward = float(max(0.0, min(1.0, score)))
    reasoning_text = "검증 요약:\n" + "\n".join(reasoning) + f"\n\n최종 점수(추정): {reward:.2f}"
    return reward, reasoning_text



# 6. Generator: RAG + Bandit 전략 적용


def generate_answer_with_strategy(question: str, strategy_idx: int, strategy_cfg: Dict[str, float]) -> str:
    """
    - RAG로 컨텍스트 검색
    - Few-shot + RAG 문맥 포함 프롬프트 구성
    - 지정된 전략(temperature, top_p)으로 생성
    - 프롬프트 토큰 길이 기준으로 출력에서 '답변 부분'만 잘라냄
    """
    rag_context, _ = rag_retrieve(question)
    prompt = build_prompt(question, rag_context)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_PROMPT_TOKENS, 
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=strategy_cfg["temp"],
            top_p=strategy_cfg["top_p"],
            pad_token_id=tokenizer.eos_token_id,
        )


    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0][prompt_len:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return answer.strip()


def answer_question(question: str) -> Dict[str, Any]:
    """
    한 번의 질의 처리:
      - Bandit으로 전략 선택
      - Generator로 답변 생성
      - Verifier로 점수 및 CoT 스타일 설명
      - 메모리 저장
    """
    strategy_idx, strategy_cfg = choose_strategy()
    logger.info(f"[BANDIT] Selected strategy {strategy_idx}: {strategy_cfg}")

    answer = generate_answer_with_strategy(question, strategy_idx, strategy_cfg)
    reward, reasoning = verify_answer(question, answer)

    logger.info(f"[VERIFY] Reward={reward:.3f}")
    save_memory(question, answer, reward, strategy_idx)

    return {
        "question": question,
        "answer": answer,
        "reward": reward,
        "strategy_index": strategy_idx,
        "strategy_cfg": strategy_cfg,
        "verify_cot": reasoning,
    }


# 7. novel_test_data 


def load_novel_test(max_samples: int = 5) -> List[Dict[str, Any]]:
    if not os.path.exists(NOVEL_PATH):
        logger.warning(f"[NOVEL] {NOVEL_PATH} not found. Skipping novel test preview.")
        return []
    data = []
    with open(NOVEL_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            obj = json.loads(line)
            data.append(obj)
    logger.info(f"[NOVEL] Loaded {len(data)} novel test samples.")
    return data


def run_novel_preview(n_samples: int = 3):
    novel = load_novel_test(max_samples=n_samples)
    if not novel:
        return

    logger.info("[NOVEL] ===== Novel test preview =====")
    for i, item in enumerate(novel):
        q = item.get("input") or item.get("question") or ""
        if not q:
            continue
        logger.info(f"\n[NOVEL SAMPLE {i+1}] Q: {q[:200]}...")
        result = answer_question(q)
        print("=" * 80)
        print(f"[NOVEL {i+1}] 질문:\n{q}\n")
        print(f"[전략 index] {result['strategy_index']} / cfg={result['strategy_cfg']}")
        print(f"[추정 보상] {result['reward']:.3f}\n")
        print("▶ 생성 답변:")
        print(result["answer"][:1000])
        if len(result["answer"]) > 1000:
            print(f"\n[... 이하 생략: 총 길이 {len(result['answer'])}자 ...]")
        print("\n▶ 검증 CoT:")
        print(result["verify_cot"])
        print("=" * 80)



# 8. 메인: Novel Preview + 실시간 질의


if __name__ == "__main__":
    logger.info("==== 민원 RAG + Finetuned TinyLlama + Hybrid Bandit 시스템 시작 ====")

    # 1) novel_test_data 일부 샘플에 대해 자동 실행 (있으면)
    run_novel_preview(n_samples=3)

    # 2) 실시간 질의 응답 루프
    print("\n\n[실시간 민원 질의 모드]")
    print("엔터만 입력하면 종료됩니다.\n")

    while True:
        try:
            q = input("질문을 입력하세요: ").strip()
        except EOFError:
            break

        if not q:
            print("종료합니다.")
            break

        result = answer_question(q)
        print("\n" + "=" * 80)
        print("[질문]")
        print(q)
        print("\n[선택된 전략 index / cfg]")
        print(result["strategy_index"], result["strategy_cfg"])
        print(f"\n[추정 보상] {result['reward']:.3f}")
        print("\n[생성 답변]")
        print(result["answer"])
        print("\n[검증 CoT]")
        print(result["verify_cot"])
        print("=" * 80 + "\n")
