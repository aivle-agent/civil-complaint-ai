from src.models.state import CivilComplaintState

def generate_strategy_node(state: CivilComplaintState) -> CivilComplaintState:
    """
    Generates a strategy to answer the complaint.
    """
    print("---GENERATE STRATEGY NODE---")
    refined_question = state.get("refined_question", "")
    # Mock strategy generation
    strategy = f"Strategy for: {refined_question}"
    return {"strategy": strategy}
