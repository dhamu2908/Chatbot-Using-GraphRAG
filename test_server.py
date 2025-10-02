from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="Test GraphRAG Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"], 
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Test server is running!"}

@app.get("/status")
def status():
    return {"status": "ok", "message": "Server is working"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)