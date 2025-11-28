from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uuid
from agent_logic import HealthAgent
from alzheimer_agent import AlzheimerAgent

app = FastAPI(title="Agentic AI Health Assistant")

# In-memory session storage
sessions: Dict[str, Any] = {}

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    session_id: str
    response: Dict[str, Any]

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    General Health Agent Chat Endpoint.
    """
    session_id = request.session_id
    if not session_id or session_id not in sessions:
        session_id = str(uuid.uuid4())
        sessions[session_id] = HealthAgent(session_id=session_id)
    
    agent = sessions[session_id]
    if not isinstance(agent, HealthAgent):
         # If session exists but is wrong type, overwrite or error. Overwriting for now.
         sessions[session_id] = HealthAgent(session_id=session_id)
         agent = sessions[session_id]

    try:
        agent_response = agent.step(request.message)
        return ChatResponse(session_id=session_id, response=agent_response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/alzheimer", response_model=ChatResponse)
async def alzheimer_chat_endpoint(request: ChatRequest):
    """
    Alzheimer's Triage Agent Chat Endpoint.
    """
    session_id = request.session_id
    if not session_id or session_id not in sessions:
        session_id = str(uuid.uuid4())
        sessions[session_id] = AlzheimerAgent(session_id=session_id)
    
    agent = sessions[session_id]
    if not isinstance(agent, AlzheimerAgent):
         sessions[session_id] = AlzheimerAgent(session_id=session_id)
         agent = sessions[session_id]

    try:
        agent_response = agent.step(request.message)
        return ChatResponse(session_id=session_id, response=agent_response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Agentic AI Health Assistant API is running."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
