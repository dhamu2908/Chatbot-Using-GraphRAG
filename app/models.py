from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    question: str
    context: Optional[str] = None

class ChatResponse(BaseModel):
    question: str
    answer: str
    context: Optional[List[str]] = None

class Document(BaseModel):
    title: str
    content: str

class Chunk(BaseModel):
    id: str
    content: str
    document_id: str
    keywords: List[str]