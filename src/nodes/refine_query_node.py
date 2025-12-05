# src/nodes/refine_query_node.py
# -*- coding: utf-8 -*-
"""
Local LangGraph node:
- Qwen/Qwen2.5-1.5B-Instruct 로 민원 질문 품질 평가
- RandomForestRegressor + SHAP 으로 품질 예측 및 중요도 분석
- refined_question, strategy, predicted_quality, quality_shap_plot_base64 를 state에 기록
"""

from __future__ import annotations

import base64
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..models.state import CivilComplaintState  # models/state.py 에서 타입 불러옴


# -------------------------------------------------------------------
# 1. 경로 및 기본 설정
# -------------------------------------------------------------------

# HF 모델 ID (환경변수 QWEN_MODEL_ID 로 덮어쓰기 가능)
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# 이 파일: src/nodes/refine_query_node.py
THIS_DIR = Path(__file__).resolve().parent        # .../src/nodes
MODEL_SAVE_DIR = THIS_DIR.parent / "models"       # .../src/models

# RF / SHAP 아티팩트 캐시
ARTIFACTS: Dict[str, Any] = {"model": None, "explainer": None, "features": None}

# LLM 전역 캐시 (lazy load)
_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModelForCausalLM] = None


def _get_device() -> torch.device:
    """사용할 디바이스 선택 (Apple Silicon 이면 mps 우선)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _get_device()


# -------------------------------------------------------------------
# 2. LLM 로딩 및 호출 유틸
# -------------------------------------------------------------------
def initialize_models() -> None:
    """
    Qwen2.5-1.5B-Instruct 모델 lazy-load.
    - QWEN_MODEL_ID 환경변수: 모델 ID 변경 가능
    - HUGGINGFACE_HUB_TOKEN / HF_TOKEN / HUGGINGFACE_TOKEN: HF 토큰 사용
    """
    global _tokenizer, _model

    if _model is not None and _tokenizer is not None:
        return

    model_id = os.getenv("QWEN_MODEL_ID", DEFAULT_MODEL_ID)

    hf_token = (
        os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
    )

    print(f"[INFO] Loading LLM: {model_id} on device={DEVICE} ...")

    torch_dtype = torch.float16 if DEVICE.type == "mps" else torch.float32

    _tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=hf_token if hf_token else None,
    )
    _model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=None,
        token=hf_token if hf_token else None,
    ).to(DEVICE)

    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    print("[INFO] LLM loaded successfully.")


def _generate_from_messages(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_new_tokens: int = 1024,
) -> str:
    """
    Qwen chat template 으로 messages 입력받아 텍스트 생성.
    messages 예:
      [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
    """
    initialize_models()
    assert _tokenizer is not None and _model is not None

    prompt_text = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = _tokenizer(prompt_text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        generated = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            eos_token_id=_tokenizer.eos_token_id,
        )

    gen_ids = generated[0, inputs["input_ids"].shape[1] :]
    out = _tokenizer.decode(gen_ids, skip_special_tokens=True)
    return out.strip()


def call_llm_json(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
) -> Optional[Dict[str, Any]]:
    """LLM 에 JSON 응답을 요청하고 파싱. 실패 시 None."""
    try:
        raw = _generate_from_messages(
            messages,
            temperature=temperature,
            max_new_tokens=512,
        )
        start_idx = raw.find("{")
        end_idx = raw.rfind("}")
        if start_idx == -1 or end_idx == -1:
            return None
        return json.loads(raw[start_idx : end_idx + 1])
    except Exception as e:
        print(f"[WARN] JSON Parsing Error: {e}")
        return None


def call_llm_text(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
) -> str:
    """LLM 에 일반 텍스트 응답을 요청."""
    try:
        return _generate_from_messages(
            messages,
            temperature=temperature,
            max_new_tokens=1024,
        )
    except Exception as e:
        return f"[ERROR] LLM call failed: {e}"


def score_question_quality(
    question_text: str,
    topic: Optional[str] = None,
) -> Dict[str, float]:
    """
    민원 질문 품질을 여러 차원으로 점수화.
    JSON 파싱 실패 시 모든 항목 0.5 로 fallback.
    """
    system_prompt = system_prompt = """You are an expert policy maker and law expert that rates the QUALITY of a citizen complaint QUESTION.
You must respond ONLY in strict JSON with floating-point scores between 0 and 1.

Definitions:
- clarity: how clearly who/when/where/what happened/what is requested are specified.
- specific: how concrete the description is (numbers, dates, durations, counts).
- fact_ratio: proportion of factual descriptions vs emotions/insults.
- noise: proportion of irrelevant or off-topic content (higher = worse).
- focus: whether the complaint is about a single main issue (1.0) or many mixed issues (0.0).
- law_situation_sim: how much this situation resembles typical legal/administrative cases,
  NOT whether legal articles are explicitly cited.
"""
    user_content = (
        f"[QUESTION]:\n{question_text}\n\n"
        "Return ONLY JSON like:\n"
        '{ "clarity": 0.5, "specific": 0.5, "fact_ratio": 0.5, '
        '"noise": 0.5, "focus": 0.5, "law_situation_sim": 0.5 }'
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    res = call_llm_json(messages)

    if res is None:
        return {
            "clarity": 0.5,
            "specific": 0.5,
            "fact_ratio": 0.5,
            "noise": 0.5,
            "focus": 0.5,
            "law_situation_sim": 0.5,
        }

    keys = {
        "clarity",
        "specific",
        "fact_ratio",
        "noise",
        "focus",
        "law_situation_sim",
    }
    return {k: float(v) for k, v in res.items() if k in keys}

def compute_question_quality(question_text: str, topic: Optional[str] = None) -> Dict[str, float]:
    """
    Backward-compatible wrapper used by tests.

    The original codebase exposed `compute_question_quality` for
    evaluating question quality. Internally we now use
    `score_question_quality`, so this function simply delegates
    to that implementation.

    Args:
        question_text: 원본 민원 또는 질문 텍스트.
        topic: (선택) 주제 정보. 현재는 사용하지 않지만
               기존 시그니처를 유지하기 위해 남겨둡니다.

    Returns:
        품질 점수 딕셔너리. 예:
        {
            "clarity": 0.0~1.0,
            "specific": 0.0~1.0,
            "fact_ratio": 0.0~1.0,
            "noise": 0.0~1.0,
            "focus": 0.0~1.0,
            "law_situation_sim": 0.0~1.0,
        }
    """
    return score_question_quality(question_text, topic=topic)

def generate_refinement_suggestions(
    question_text: str,
    shap_summary_text: Optional[str] = None,
    quality_scores: Optional[Dict[str, float]] = None,
    *args: Any,
    **kwargs: Any,
) -> str:
    """
    Backward-compatible helper used by tests.

    이전 버전 코드베이스에서는 이 함수가
    SHAP/품질 분석 결과를 바탕으로 민원 개선 가이드를 생성하는 역할을 했습니다.
    현재 리팩토링 버전에서는 내부적으로 LLM을 사용하는 동일한 목적의
    가이드를 생성하도록 구현합니다.

    Args:
        question_text:
            원본 민원 또는 질문 텍스트.
        shap_summary_text:
            (선택) 품질 지표와 SHAP 기여도를 요약한 문자열.
            예: "- 항목: q_clarity, 현재점수: 0.30, 개선필요도(SHAP): -0.0123" 형식.
        quality_scores:
            (선택) 품질 점수 딕셔너리.
            예: {"clarity": 0.3, "specific": 0.4, ...}

    Returns:
        한국어로 된 개선 가이드 문자열(여러 줄).
    """
    # LLM에 전달할 분석 텍스트 조합
    analysis_parts: list[str] = []

    if shap_summary_text:
        analysis_parts.append("SHAP 기반 분석 요약:\n" + shap_summary_text)

    if quality_scores:
        qs_lines = "\n".join(
            f"- {k}: {float(v):.3f}" for k, v in quality_scores.items()
        )
        analysis_parts.append("예측된 품질 점수:\n" + qs_lines)

    if analysis_parts:
        analysis_text = "\n\n".join(analysis_parts)
    else:
        analysis_text = "분석 정보는 별도로 제공되지 않았습니다."

    system_prompt = """
You are a writing assistant for Korean citizen complaints.
You receive an original complaint and some analysis info (quality scores or SHAP summaries).
Based on this, you must return 3–5 short, concrete suggestions in Korean
that help the citizen rewrite their complaint so that it is clearer, more specific,
focused on a single main issue, and suitable for submission to a public office.

Requirements:
- Output must be written ONLY in Korean.
- Do not mention SHAP, 점수, 모델, 차원, 특성, 피처, 분석 결과 or any internal analysis terms.
- Each suggestion should be one sentence starting with a dash (-) or a numbered item (예: "1.", "2.").
- Focus on what additional facts (누가, 언제, 어디서, 무엇을, 왜, 어떻게) should be 포함될지,
  또는 어떤 불필요한 감정적 표현이나 반복을 줄이면 좋을지에 대해 조언해 주세요.
"""

    user_prompt = f"""[원본 민원]
{question_text}

[분석 정보]
{analysis_text}

위 정보를 바탕으로, 민원인이 글을 다시 쓸 때 참고할 수 있는
구체적인 개선 제안을 3~5개 한국어로 작성해 주세요.
각 제안은 한 문장으로만 작성하고, 앞에 "-" 또는 "번호+점(예: 1.)"을 붙여 주세요.
"""

    suggestions = call_llm_text(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return suggestions.strip()


# -------------------------------------------------------------------
# 3. RF + SHAP 아티팩트 로딩
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# 3. RF + SHAP 아티팩트 로딩
# -------------------------------------------------------------------
def load_artifacts_if_needed() -> bool:
    """
    RandomForestRegressor, feature column 목록을 로드하고,
    가능하면 shap_explainer.joblib 을 사용하되
    실패 시에는 TreeExplainer 를 새로 만들어 사용한다.

    → 로컬 / GitHub Actions 모두에서 버전 차이에 최대한 견고하게 동작하도록 설계.
    """
    if (
        ARTIFACTS["model"] is not None
        and ARTIFACTS["explainer"] is not None
        and ARTIFACTS["features"] is not None
    ):
        return True

    try:
        rf_model = joblib.load(MODEL_SAVE_DIR / "rf_model.joblib")
        feature_cols = joblib.load(MODEL_SAVE_DIR / "feature_cols.joblib")

        explainer = None
        explainer_path = MODEL_SAVE_DIR / "shap_explainer.joblib"

        # 1순위: 기존에 저장해 둔 explainer 가 있다면 먼저 시도
        if explainer_path.exists():
            try:
                explainer = joblib.load(explainer_path)
                print("[INFO] SHAP explainer loaded from shap_explainer.joblib.")
            except Exception as e:
                print(
                    "[WARN] Failed to load shap_explainer.joblib, "
                    f"will rebuild TreeExplainer instead: {e}"
                )

        # 2순위: 파일이 없거나 로드 실패 → TreeExplainer 새로 구성
        if explainer is None:
            explainer = shap.TreeExplainer(rf_model)
            print("[INFO] SHAP TreeExplainer created from RF model (no pickle).")

        ARTIFACTS["model"] = rf_model
        ARTIFACTS["explainer"] = explainer
        ARTIFACTS["features"] = feature_cols

        return True

    except FileNotFoundError:
        print(
            "[ERROR] Model artifacts not found. "
            "Expected rf_model.joblib and feature_cols.joblib "
            f"under: {MODEL_SAVE_DIR}"
        )
        return False
    except Exception as e:
        print(f"[ERROR] Failed to initialize RF/SHAP artifacts: {e}")
        return False



# -------------------------------------------------------------------
# 4. LangGraph 노드 본체
# -------------------------------------------------------------------
def refine_query_node(state: CivilComplaintState) -> CivilComplaintState:
    """
    LangGraph 노드:
    - state["user_question"] 를 입력으로 받아
      refined_question, strategy, predicted_quality, quality_shap_plot_base64 를 채워서 반환.
    """
    user_question = state["user_question"]
    print(f"\n>> [Node] Refining Query: {user_question[:30]}...")

    if not load_artifacts_if_needed():
        state["strategy"] = "[ERROR] Model not loaded"
        state["quality_shap_plot_base64"] = None
        return state

    rf_model = ARTIFACTS["model"]
    explainer = ARTIFACTS["explainer"]
    feature_cols = ARTIFACTS["features"]

    # 1. LLM 기반 질문 품질 점수
    print("   ... (LLM) 질문 품질 점수 계산 중 ...")
    q_scores = score_question_quality(user_question)

    # 2. RF 입력 벡터 생성
    input_row: Dict[str, float] = {"qa_similarity": 0.5}
    for k, v in q_scores.items():
        input_row[f"q_{k}"] = v

    input_vector = np.array([[input_row.get(col, 0.5) for col in feature_cols]])

    # 3. RF 예측 + SHAP 값
    print("   ... (RF/SHAP) 품질 예측 및 중요도 분석 중 ...")
    predicted_quality = float(rf_model.predict(input_vector)[0])
    shap_vals = explainer.shap_values(input_vector)[0]
    local_shap = dict(zip(feature_cols, shap_vals))

    q_shap_items: List[Dict[str, float]] = []
    for k, v in local_shap.items():
        if k.startswith("q_"):
            score = input_row.get(k, 0.0)
            q_shap_items.append({"name": k, "score": score, "shap_value": float(v)})

    # SHAP 기여도 기준 정렬 (오름차순: 품질에 더 안 좋은 영향부터)
    sorted_items = sorted(q_shap_items, key=lambda x: x["shap_value"])

    # LLM 가이드라인 입력용 설명 텍스트
    info_lines = "\n".join(
        f"- 항목: {it['name']}, 현재점수: {it['score']:.2f}, 개선필요도(SHAP): {it['shap_value']:.4f}"
        for it in sorted_items
    )

    # 3-1. SHAP 막대그래프 이미지(Base64) 생성 (상위 |value| 6개)
    try:
        top_n = 6
        top_items = sorted(
            q_shap_items,
            key=lambda x: abs(x["shap_value"]),
            reverse=True,
        )[:top_n]

        feature_names = [it["name"].replace("q_", "") for it in top_items]
        shap_values_top = [it["shap_value"] for it in top_items]
        colors = ["tab:blue" if v >= 0 else "tab:red" for v in shap_values_top]

        plt.figure(figsize=(6, 4))
        plt.barh(feature_names, shap_values_top, color=colors)
        plt.axvline(0, color="black", linewidth=0.8)
        plt.xlabel("SHAP value")
        plt.title("Measurement of Civil Complaint")
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        plt.close()

        state["quality_shap_plot_base64"] = image_base64
    except Exception as e:
        print(f"[WARN] Failed to generate SHAP plot: {e}")
        state["quality_shap_plot_base64"] = None

    # 4. SHAP 기반 교정 가이드(strategy) 생성
    print("   ... (LLM) 교정 가이드라인 생성 중 ...")

    guideline_system_prompt = """You are a writing assistant for citizen complaints.
You must analyze the provided SHAP values to generate concrete, actionable advice in Korean.

[Quality Dimension Korean Mapping Table]
- clarity: 명확성 부족
- specific: 구체성 부족
- fact_ratio: 사실 기반 비율 부족 (감정/주관 개입 과다)
- noise: 불필요한 내용 (잡음)
- focus: 단일 주제 집중 부족 (주제 혼재)
- law_situation_sim: 법률/행정 상황 유사성 부족

[SHAP Interpretation and Guidance Rules]
1.  **Prioritization:** Address factors with LOW score (below 0.5) and strongly NEGATIVE SHAP value first.
2.  **No Redundancy:** Ensure each bullet point provides unique advice.
3.  **Diagnosis First (Korean):** For each bullet point, you MUST first state which quality dimension is low by using the **Korean term** from the mapping table (e.g., '명확성 부족' is 0.2, SHAP is -0.0032) and link it directly to the problematic part of the original question.
4.  **Actionable Advice:** The advice MUST suggest specific actions:
    * **Clarity/Specific:** Explicitly request missing factual data (e.g., '공장의 정확한 주소와 용도변경을 신청한 날짜를 추가로 적어 주세요.').
    * **Focus/Law:** Convert vague complaints into specific administrative inquiries.
5.  *
"""

    guideline_user_content = f"""[원본 민원]
{user_question}

[질문 품질 점수 및 SHAP 기여도]
{info_lines}

위 분석을 바탕으로, 민원인이 내용을 보완할 수 있도록 3가지 구체적인 가이드를 작성해 주세요.
"""

    guideline = call_llm_text(
        [
            {"role": "system", "content": guideline_system_prompt},
            {"role": "user", "content": guideline_user_content},
        ]
    )

    # 5. 최종 민원문(refined_question) 생성
    print("   ... (LLM) 최종 민원문 교정 중 ...")

    rewrite_system_prompt = """
You are a rewriting assistant for Korean citizen complaints.
Your goal is to transform the original complaint into a clear, concise, and natural Korean text
that a public official can easily read, understand the situation from, and use to consider appropriate administrative action.

CRITICAL LANGUAGE CONSTRAINTS
1. The final output MUST be written **only in Korean**.
2. Do NOT use any English words, Chinese characters, or any other foreign language in the final complaint text.
   - No English words such as "quality", "policy", "case", "SHAP", "score", "feature", "dimension", etc.
   - No Chinese characters or non-Hangul scripts. If a technical term or foreign word comes to mind,
     you MUST rewrite it into natural Korean explanation instead (e.g., "생활의 질", "사례", "정책", "층간소음").
3. Numbers (e.g., 1, 2, 3, 2025) are allowed, but you MUST NOT invent specific dates, counts, or article numbers
   that are not present in the original complaint or in the provided guideline.

TARGET ROLE AND PERSPECTIVE
- The text you generate will be stored as 'refined_question' and submitted directly as a citizen complaint.
- The reader is a public official in a Korean administrative agency.
- Always write from the citizen's first-person perspective:
  - Use expressions like "저는", "저희 가족은", "저희 아파트는".
- Use polite, formal Korean suitable for an official complaint.

ABSOLUTE PROHIBITIONS
You MUST NOT:
1. Use any analysis/model-related terminology in the final text:
   - No "SHAP", "점수", "모델", "차원", "특성", "피처", "분석 결과" or similar.
2. Produce placeholders or bracketed variables in the final text:
   - No "[아파트 이름]", "[날짜]", "[층]", "[SHAP 값]", "[질량 차원]" or any other text in square brackets [ ].
3. Give writing instructions or meta-comments inside the final complaint:
   - No "다음과 같이 작성해 주십시오", "추가로 ~를 적어 주시면 좋겠습니다", "이 글은 ~에 대한 민원입니다.".
4. Describe the document itself:
   - Do NOT say "이 민원은 ~에 대한 문의입니다.", "이 문서는 ~을 설명합니다." etc.
5. Invent new concrete facts that are not supported by the original complaint or guideline:
   - Do NOT invent exact dates like "2025년 6월 1일",
     exact counts like "최근 3개월간 신고 50건",
     or specific article numbers of laws or regulations.
   - If such specific information is needed in reality, you must instead
     phrase it as a general request for guidance, NOT by fabricating details.
6. **Say that the complaint is unclear or lacks information, or ask the citizen to add more information.**
   - Do NOT write sentences such as:
     - "민원 내용이 명확하지 않습니다."
     - "특정 위치나 시간 등에 대한 정보가 부족합니다."
     - "추가 정보를 제공해 주시면 감사하겠습니다."
     - "연락처나 신청 날짜를 추가로 적어 주십시오."
   - Even if important details are missing, you must NOT comment on the lack of information
     and must NOT request the citizen to add, supplement, or provide more details, contact information, or dates.
   - Instead, you must write the best possible complete complaint using general expressions
     (예: "최근", "출퇴근 시간대", "여러 차례", "인근 도로", "아파트 단지 앞 인도" 등).

FACT HANDLING RULES
1. Use only information that is:
   - Explicitly present in the original complaint, OR
   - Reasonably implied at a general level (e.g., "최근 몇 달간", "여러 차례", "출퇴근 시간대").
2. If a related regulation, law, or rule seems important:
   - Do NOT invent specific law or article names.
   - Instead, write a natural Korean request such as:
     - "해당 상황에 적용될 수 있는 관련 법령이나 조례, 기준을 안내해 주시기 바랍니다."
     - "관련 규정에 따라 어떤 조치가 가능한지 설명해 주시기 바랍니다."

CONTENT STRUCTURE AND LOGIC
You should organize the complaint into 2–3 short paragraphs in Korean:

[Paragraph 1: Background and overall situation]
- Briefly state who you are and what general problem you are experiencing.
- Include information such as:
  - Where (도로, 인도, 아파트 단지, 동네 등 — general description only),
  - Since when or how often (최근, 출퇴근 시간대, 여러 차례 등),
  - What kind of problem (눈·얼음으로 인한 미끄러움, 교통 체증, 소음 등).

[Paragraph 2: Concrete problem details and impact]
- Describe how the problem appears in daily life:
  - Repetition, severity, and concrete effects on safety, daily life, or traffic.
  - Any actions already taken (e.g., 관리사무소에 문의, 이전 민원 제기 등), if mentioned or reasonably implied.
- Use the six-question perspective (who, where, when, what, why, how) as much as the original content allows,
  but do NOT invent specific dates, counts, or article numbers.

[Paragraph 3: Clear request to the authority]
- Clearly state what you want the public office to do.
- Acceptable request types include:
  - Requesting specific administrative actions:
    - "제설 작업 등 안전 조치를 조속히 시행해 주시기 바랍니다."
    - "교통 체증을 완화할 수 있는 신호 체계 조정이나 우회 동선 마련을 검토해 주십시오."
  - Requesting guidance on regulations:
    - "해당 상황에 적용될 수 있는 관련 법령이나 조례, 기준이 있다면 함께 안내해 주시기 바랍니다."
    - "관련 규정에 따라 어떤 조치가 가능한지 설명해 주시기 바랍니다."

STYLE REQUIREMENTS
1. The tone must be polite, formal Korean appropriate for communication with a public official.
2. Sentences should not be excessively long. Split long explanations into two or more clear sentences.
3. Do NOT repeat the same complaint sentence in a redundant way.
4. Use natural Korean expressions instead of foreign terms.
   - If a foreign or technical term comes to mind, you MUST replace it with a natural Korean explanation.

GOOD EXAMPLE PATTERN (FOR STYLE ONLY, NOT FOR FACTS)
You may use the following pattern as a STYLE reference only (do NOT copy dates or numbers):
- 문제 상황 제시: 언제부터, 어느 장소에서, 어떤 문제가 반복되고 있는지 설명
- 구체적 맥락: 어느 시간대에 특히 심각한지, 어떤 불편이 있는지 설명
- 요청: "신속한 개선 조치와 ○○ 개선을 요청합니다."와 같이 명확한 조치 요청

Final Output Instructions
- Output ONLY the final complaint text in Korean, as continuous paragraphs.
- Do NOT include:
  - Any bullet points, headings, numbering, or lists.
  - Any explanations about what you are doing.
  - Any English or Chinese words, or other foreign language.
  - Any square-bracket placeholders like [내용].
  - Any sentences that analyse the complaint itself (e.g., "민원 내용이 부족합니다.", "정보가 명확하지 않습니다.")
  - Any sentences that ask the citizen to provide/add/supplement information, contact details, or dates.
- The output must be ready to submit directly as an official citizen complaint in Korean.

"""



    rewrite_user_content = f"""[원본 민원]
{user_question}

[교정 가이드]
{guideline}

위 가이드를 반영하여, 공무원이 읽기 좋게 다듬어진 '최종 민원 내용'만 작성해 주세요.
"""

    refined_question = call_llm_text(
        [
            {"role": "system", "content": rewrite_system_prompt},
            {"role": "user", "content": rewrite_user_content},
        ]
    )

    # 6. state 업데이트
    state["refined_question"] = refined_question.strip()
    state["strategy"] = guideline

    if state.get("quality_scores") is None:
        state["quality_scores"] = {}

    state["quality_scores"]["predicted_quality"] = round(predicted_quality, 4)

    return state


# if __name__ == "__main__":
#     # 단독 실행 테스트
#     test_query = (
#         "집 앞 인도에 눈이 쌓여 매우 미끄러운 상태입니다. "
#         "보행자 안전을 위해 제설 작업을 조속히 진행해 주시길 요청드립니다."
#     )

#     init_state: CivilComplaintState = {
#         "user_question": test_query,
#         "refined_question": None,
#         "quality_scores": None,
#         "strategy": None,
#         "draft_answer": None,
#         "verification_feedback": None,
#         "is_verified": False,
#         "final_answer": None,
#         "retry_count": 0,
#         "rag_context": None,
#         "quality_shap_plot_base64": None,
#     }

#     result_state = refine_query_node(init_state)

#     print("\n" + "=" * 40)
#     print("       [최종 결과 확인]       ")
#     print("=" * 40)
#     print("\n[교정 가이드]")
#     print(result_state.get("strategy"))
#     print("\n[교정된 민원 (최종)]")
#     print(result_state.get("refined_question"))
