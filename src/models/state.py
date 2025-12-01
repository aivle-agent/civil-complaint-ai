from typing import TypedDict, Optional, Dict


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
