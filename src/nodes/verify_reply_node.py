import random
from src.models.state import CivilComplaintState


def verify_reply_node(state: CivilComplaintState) -> CivilComplaintState:
    """
    Verifies the drafted reply.
    Has a 0.5 probability of rejecting the reply (sending back to strategy).
    """
    print("---VERIFY REPLY NODE---")

    # 0.5 probability
    is_verified = random.random() > 0.5

    if is_verified:
        feedback = "Verification Passed."
        print("Verification: PASSED")
    else:
        feedback = "Verification Failed. Please retry."
        print("Verification: FAILED (Retrying...)")

    return {
        "verification_feedback": feedback,
        "is_verified": is_verified,
        "retry_count": state.get("retry_count", 0) + 1,
    }