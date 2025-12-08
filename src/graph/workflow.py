from langgraph.graph import StateGraph, END
from src.models.state import CivilComplaintState

from src.nodes.civil_complaint_node import civil_complaint_node
from src.nodes.refine_query_node import refine_query_node
from src.nodes.draft_reply_node import draft_reply_node


def create_graph():
    workflow = StateGraph(CivilComplaintState)

    # Add nodes
    workflow.add_node("civi_complaint", civil_complaint_node)
    workflow.add_node("refine_query", refine_query_node)
    workflow.add_node("draft_reply", draft_reply_node)

    # Define edges
    workflow.set_entry_point("civi_complaint")
    workflow.add_edge("civi_complaint", "refine_query")
    workflow.add_edge("refine_query", "draft_reply")
    workflow.add_edge("draft_reply", END)

    return workflow.compile()
