from typing import TypedDict, Optional


class CivilComplaintState(TypedDict):
    user_question: str
    refined_question: Optional[str]
    strategy: Optional[str]
    draft_answer: Optional[str]
    verification_feedback: Optional[str]
    is_verified: bool  # Added for routing logic
    final_answer: Optional[str]
    retry_count: int
