from src.models.state import CivilComplaintState

def civi_complaint_node(state: CivilComplaintState) -> CivilComplaintState:
    """
    Initial processing of the civil complaint.
    """
    print("---CIVIL COMPLAINT NODE---")
    # In a real app, this might classify the complaint or do initial setup
    return {"retry_count": 0}
