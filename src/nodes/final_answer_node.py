from src.models.state import CivilComplaintState


def final_answer_node(state: CivilComplaintState) -> CivilComplaintState:
    """
    Produces the final answer.
    """
    print("---FINAL ANSWER NODE---")
    draft = state.get("draft_answer", "")
    final = f"Final Answer: {draft}"
    return {"final_answer": final}
