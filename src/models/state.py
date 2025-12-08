from typing import TypedDict, Optional, Dict, List, Any


class CivilComplaintState(TypedDict):
    user_question: str
    
    refined_question: Optional[str]
    quality_scores: Optional[Dict[str, float]]  # Quality metrics for the question
    
    quality_shap_plot_base64: Optional[str]  # SHAP 막대그래프
    
    strategy: Optional[str] # retriever
    verification_feedback: Optional[str]
    
    retry_count: int
    rag_context: Optional[str]  # Context retrieved via RAG
    retrieved_documents: Optional[List[Dict[str, Any]]] # Retrieved documents from Kanana RAG
    
    draft_answer: Optional[str]
    final_answer: Optional[str]