import os
from src.graph.workflow import create_graph
from src.models.state import CivilComplaintState

def main():
    # 1. Initialize Graph
    print("[Main] Initializing LangGraph...")
    graph = create_graph()
    
    # 2. Define Initial State
    user_input = "아파트 층간소음 문제로 고통받고 있습니다. 해결 방법이 있을까요?"
    initial_state: CivilComplaintState = {
        "user_question": user_input,
        "refined_question": None,
        "quality_scores": None,
        "quality_shap_plot_base64": None,
        "strategy": None,
        "verification_feedback": None,
        "retry_count": 0,
        "rag_context": None,
        "retrieved_documents": None,
        "draft_answer": None,
        "final_answer": None
    }
    
    print(f"\n[Main] User Question: {user_input}")
    print("-" * 50)
    
    # 3. Run Graph
    try:
        result = graph.invoke(initial_state)
        
        print("\n" + "=" * 50)
        print("           [Workflow Execution Result]           ")
        print("=" * 50)
        
        # Refined Question
        print(f"\n1. Refined Question:\n{result.get('refined_question')}")
        
        # Retrieved Documents
        docs = result.get('retrieved_documents')
        print(f"\n2. Retrieved Documents: {len(docs) if docs else 0} items")
        if docs:
            for i, doc in enumerate(docs):
                print(f"   [{i+1}] {doc['doc'][:80]}... (Source: {doc.get('meta', {}).get('source', 'Unknown')})")
        
        # Draft Answer
        print(f"\n3. Draft Answer:\n{result.get('draft_answer')}")
        
    except Exception as e:
        print(f"\n[Error] Workflow execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
