from typing import TypedDict, Optional, Dict, List


class CivilComplaintState(TypedDict):
    user_question: str
    refined_question: Optional[str]
    quality_scores: Optional[Dict[str, float]]  # Quality metrics for the question
    strategy: Optional[str]
    draft_answer: Optional[str]
    verification_feedback: Optional[str]
    is_verified: bool  # Added for routing logic
    final_answer: Optional[str]
    retry_count: int

    rag_context: Optional[str]  # Context retrieved via RAG