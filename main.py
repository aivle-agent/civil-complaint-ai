import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from src.graph.workflow import create_graph

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the graph
workflow = create_graph()

class ProposalRequest(BaseModel):
    prplTitl: str
    prplCntnCl: str
    btmtIdeaCl: str
    expcEfctCl: str

@app.post("/api/submit-proposal")
async def submit_proposal(request: ProposalRequest):
    try:
        # Construct the user question from the form fields
        user_question = (
            f"제목: {request.prplTitl}\n"
            f"현황 및 문제점: {request.prplCntnCl}\n"
            f"개선방안: {request.btmtIdeaCl}\n"
            f"기대효과: {request.expcEfctCl}"
        )

        initial_state = {
            "user_question": user_question,
            "retry_count": 0,
            "is_verified": False  # Initialize required field
        }

        # Run the graph
        # invoke returns the final state
        result = workflow.invoke(initial_state)
        
        return {
            "status": "success",
            "refined_question": result.get("refined_question"),
            "final_answer": result.get("final_answer"),
            "draft_answer": result.get("draft_answer"),
            "strategy": result.get("strategy")
        }

    except Exception as e:
        print(f"Error processing proposal: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
