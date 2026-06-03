import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.graph import build_graph

# Ensure your project root is in the python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 1. Initialize the compiled LangGraph application
graph_app = build_graph()

# 2. Initialize the FastAPI app (This is exactly what Uvicorn is looking for!)
app = FastAPI(title="CRAG System API")

# 3. Define the data structure for incoming requests
class QueryRequest(BaseModel):
    question: str

# 4. Create a basic health-check route so Hugging Face knows it's working
@app.get("/")
def read_root():
    return {"status": "CRAG System API is running successfully!"}

# 5. Create the API endpoint to handle the actual chat logic
@app.post("/chat")
def chat_endpoint(request: QueryRequest):
    try:
        inputs = {"question": request.question}
        
        # We use invoke() instead of stream() here to easily return the final result in one API response
        final_state = graph_app.invoke(inputs)
        
        return {
            "response": final_state.get("generation", "No generation produced.")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
