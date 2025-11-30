from src.models.state import CivilComplaintState

def draft_reply_node(state: CivilComplaintState) -> CivilComplaintState:
    """
    Drafts a reply based on the strategy.
    """
    print("---DRAFT REPLY NODE---")
    strategy = state.get("strategy", "")
    # Mock draft generation
    draft = f"Draft Answer based on {strategy}"
    return {"draft_answer": draft}
