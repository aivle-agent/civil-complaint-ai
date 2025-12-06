import csv
import os
import random
from src.models.state import CivilComplaintState


def civil_complaint_node(state: CivilComplaintState) -> CivilComplaintState:
    """
    Initial processing of the civil complaint.
    If user_question is missing, load a random one from data/examples.csv.
    """
    print("---CIVIL COMPLAINT NODE---")

    if not state.get("user_question"):
        # Try to find the CSV file
        # Assuming run from root, but let's be safe with relative paths if needed
        # Or just assume data/examples.csv exists relative to CWD
        csv_path = "data/examples.csv"
        if os.path.exists(csv_path):
            try:
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    if rows:
                        selected = random.choice(rows)
                        print(f"Loaded random question from CSV: {selected.get('title')}")
                        return {
                            "user_question": selected.get("user_question"),
                            "retry_count": 0
                        }
            except Exception as e:
                print(f"Error loading examples.csv: {e}")
        else:
            print(f"Warning: {csv_path} not found.")

    # In a real app, this might classify the complaint or do initial setup
    return {"retry_count": 0}
