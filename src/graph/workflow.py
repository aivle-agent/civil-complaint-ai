from langgraph.graph import StateGraph, END
from src.models.state import CivilComplaintState

from src.nodes.civil_complaint_node import civil_complaint_node
from src.nodes.refine_query_node import refine_query_node
from src.nodes.generate_strategy_node import generate_strategy_node
from src.nodes.draft_reply_node import draft_reply_node
from src.nodes.verify_reply_node import verify_reply_node
from src.nodes.final_answer_node import final_answer_node


def should_continue(state: CivilComplaintState) -> str:
    """
    Determines the next step based on verification result.
    """
    if state["is_verified"]:
        return "final_answer"
    else:
        return "generate_strategy"


def create_graph():
    workflow = StateGraph(CivilComplaintState)

    # Add nodes
    workflow.add_node("civi_complaint", civil_complaint_node)
    workflow.add_node("refine_query", refine_query_node)
    workflow.add_node("generate_strategy", generate_strategy_node)
    workflow.add_node("draft_reply", draft_reply_node)
    workflow.add_node("verify_reply", verify_reply_node)
    workflow.add_node("final_answer", final_answer_node)

    # Define edges
    workflow.set_entry_point("civi_complaint")
    workflow.add_edge("civi_complaint", "refine_query")
    workflow.add_edge("refine_query", "generate_strategy")
    workflow.add_edge("generate_strategy", "draft_reply")
    workflow.add_edge("draft_reply", "verify_reply")

    # Conditional edge
    workflow.add_conditional_edges(
        "verify_reply",
        should_continue,
        {"final_answer": "final_answer", "generate_strategy": "generate_strategy"},
    )

    workflow.add_edge("final_answer", END)

    return workflow.compile()
