import sys
import os

# Ensure the src directory is in the python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.graph.workflow import create_graph  # noqa: E402


def main():
    # 1. Create the graph
    app = create_graph()

    # 2. Define initial state
    initial_state = {
        "user_question": "아파트 층간소음 민원은 어떻게 넣나요?",
        "retry_count": 0,
    }

    print(f"Initial Question: {initial_state['user_question']}")
    print("-" * 50)

    # 3. Run the workflow
    # invoke returns the final state
    result = app.invoke(initial_state)

    # 4. Print results
    print("-" * 50)
    print("Workflow Finished!")
    print(f"Final Answer: {result.get('final_answer')}")
    print(f"Total Retries (Verification Failures): {result.get('retry_count')}")
    print(f"Last Verification Feedback: {result.get('verification_feedback')}")


if __name__ == "__main__":
    main()
