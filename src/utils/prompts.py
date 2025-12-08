# -*- coding: utf-8 -*-
"""
프롬프트 상수 및 유틸리티 함수
rag_multi_test.py에서 추출한 Few-shot 템플릿 및 생성 전략
"""

# Few-shot 템플릿: 민원 답변 형식 예시
FEW_SHOT_TEMPLATE = """
아래는 공공 민원에 대한 모범 답변 예시입니다.
출력에서는 절대로 예시나 설명을 포함하지 말고,
오직 '검토내용'으로 시작하는 정식 답변만 작성하십시오.

※ 중요 규칙:
- 출력은 반드시 '검토내용'으로 시작해야 합니다.
- 검토내용 이후에는 번호 형태의 항목만 작성하십시오.
- '요약', '설명', '참고자료', '해설', '추가 안내', '요약문' 등 어떤 형태의 부가 설명도 절대로 작성하지 마십시오.
- 예시 문장, 프롬프트 내용, 분석 내용은 출력 금지.
- 최종 출력은 오직 정식 답변만 포함해야 하며, 그 외 텍스트는 절대로 포함하지 않습니다.

[예시 1]
민원 요지 요약: 교차로 신호 조정 요청
정식 답변:
검토내용
1. 안녕하십니까. 귀하께서 제기하신 민원에 대해 답변드립니다.
2. 민원 요지: 교차로 신호 운영 조정 요청입니다.
3. 가. 관계 기관 협의 결과 야간 시간대 단계적 조정이 타당한 것으로 검토되었습니다.
   나. 이에 따라 24시~07시 점멸 운영을 우선 시행하고 모니터링 후 추가 조정 여부를 검토하겠습니다.
4. 교통과로 문의 바랍니다. 감사합니다.

[예시 2]
민원 요지 요약: 보행자 안전시설 확충 요청
정식 답변:
검토내용
1. 안녕하십니까. 귀하께서 제기하신 민원에 대해 답변드립니다.
2. 민원 요지: 안전시설 설치 요청입니다.
3. 가. 현장 점검 결과 보행량·속도 등을 고려할 때 시설 확충이 필요한 것으로 확인되었습니다.
   나. 상반기 내 우선 조치 후 사고 추이를 보아 추가 설치를 검토할 예정입니다.
4. 자세한 사항은 교통행정과로 문의 바랍니다. 감사합니다.

위 형식을 참고하여 아래 민원에 대한 정식 답변만 작성하십시오.
예시 문장·설명·프롬프트는 출력하지 마십시오.
"""

# 답변 생성 전략: 3가지 view 기반
GEN_STRATEGIES = [
    {"name": "law_focus", "view": "law", "temp": 0.6, "top_p": 0.9},
    {"name": "case_focus", "view": "case", "temp": 0.7, "top_p": 0.9},
    {"name": "mixed_focus", "view": "mixed", "temp": 0.8, "top_p": 0.95},
]


def summarize_question(question: str, max_len: int = 180) -> str:
    """질문을 요약하여 프롬프트에 사용"""
    q = question.strip().replace("\n", " ")
    return q[:max_len] + ("..." if len(q) > max_len else "")


def strip_to_answer(full_text: str, prompt: str = "") -> str:
    """
    전체 생성 텍스트에서 프롬프트 제거하고 '검토내용'부터만 남겨서 리턴
    """
    text = full_text.replace(prompt, "").strip() if prompt else full_text.strip()
    idx = text.find("검토내용")
    if idx != -1:
        text = text[idx:]
    else:
        # 혹시 prefix가 없으면, '검토내용' 기준으로 split 시도
        parts = text.split("검토내용")
        if len(parts) > 1:
            text = "검토내용" + parts[-1]
    return text.strip()


def build_answer_prompt(question: str, view_summary: str) -> str:
    """답변 생성을 위한 프롬프트 구성"""
    return (
        FEW_SHOT_TEMPLATE
        + "\n민원 요지:\n"
        + summarize_question(question)
        + "\n\n[요약된 참고자료]\n"
        + (view_summary if view_summary else "별도의 참고자료 없음")
        + "\n\n위 내용을 바탕으로, '검토내용'으로 시작하는 공공 민원 답변을 작성하십시오.\n"
        + "예시 문장이나 설명은 출력하지 말고, 최종 답변만 작성하십시오.\n\n"
        + "검토내용\n"
    )
