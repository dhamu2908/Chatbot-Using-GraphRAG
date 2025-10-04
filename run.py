import requests
import json
from app.main import app
import uvicorn

def test_chatbot():
    # Add sample document
    sample_doc = {
        "title": "AI and Databases",
        "content": open("data/sample_data.txt", "r").read()
    }
    
    response = requests.post("http://localhost:8000/documents", json=sample_doc)
    print("Document added:", response.json())
    
    # Test chat
    chat_data = {"question": "What is Artificial Intelligence?"}
    response = requests.post("http://localhost:8000/chat", json=chat_data)
    print("Chat response:", json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    # Start the server
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)