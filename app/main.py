from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import traceback
import shutil

class ChatRequest(BaseModel):
    question: str
    context: Optional[str] = None

class SourceDocument(BaseModel):
    title: str
    doc_id: str

class ChatResponse(BaseModel):
    question: str
    answer: str
    context: Optional[List[str]] = None
    sources: Optional[List[SourceDocument]] = None

class Document(BaseModel):
    title: str
    content: str

app = FastAPI(title="GraphRAG Chatbot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

rag_system = None
postgres = None
file_storage = {}

print("Initializing GraphRAG system...")
try:
    from rag import GraphRAGSystem
    rag_system = GraphRAGSystem()
    print("GraphRAG system initialized")
except Exception as e:
    print(f"GraphRAG initialization failed: {e}")

try:
    from database import PostgresConnection
    print("Connecting to PostgreSQL...")
    postgres = PostgresConnection()
    postgres.connect()
    print("PostgreSQL connected!")
except Exception as e:
    print(f"PostgreSQL connection failed: {e}")
    postgres = None

@app.get("/")
async def root():
    return {"message": "GraphRAG Chatbot API is running!"}

@app.get("/status")
async def get_status():
    neo4j_status = "disconnected"
    if rag_system and hasattr(rag_system, 'neo4j'):
        try:
            if hasattr(rag_system.neo4j, 'driver') and rag_system.neo4j.driver:
                with rag_system.neo4j.driver.session() as session:
                    session.run("RETURN 1")
                neo4j_status = "connected"
        except Exception:
            neo4j_status = "disconnected"
    
    postgres_status = "disconnected"
    if postgres and postgres.conn:
        try:
            with postgres.conn.cursor() as cursor:
                cursor.execute("SELECT 1")
            postgres_status = "connected"
        except Exception:
            postgres_status = "disconnected"
    
    ollama_status = "disconnected"
    model_name = "unknown"
    try:
        import ollama
        ollama.list()
        ollama_status = "connected"
        if rag_system and hasattr(rag_system, 'model_name'):
            model_name = rag_system.model_name
    except Exception:
        pass
    
    return {
        "message": "GraphRAG Chatbot API is running!",
        "neo4j": neo4j_status,
        "postgresql": postgres_status,
        "ollama": ollama_status,
        "model": model_name
    }

@app.get("/documents")
async def list_documents():
    if not rag_system:
        return {"documents": [], "count": 0, "message": "GraphRAG system not available"}
    
    try:
        if hasattr(rag_system, 'neo4j') and hasattr(rag_system.neo4j, 'driver') and rag_system.neo4j.driver:
            with rag_system.neo4j.driver.session() as session:
                result = session.run("MATCH (d:Document) RETURN d.title as title, d.id as id ORDER BY d.title")
                documents = [{"id": record["id"], "title": record["title"]} for record in result]
            return {"documents": documents, "count": len(documents)}
        else:
            return {"documents": [], "count": 0, "message": "Neo4j not connected"}
    except Exception as e:
        print(f"Error listing documents: {e}")
        return {"documents": [], "count": 0, "error": str(e)}

@app.get("/document/{doc_id}")
async def get_document(doc_id: str):
    """Get document file by ID for download/viewing"""
    if doc_id in file_storage:
        file_path = file_storage[doc_id]
        if os.path.exists(file_path):
            return FileResponse(
                file_path,
                media_type='application/octet-stream',
                filename=os.path.basename(file_path)
            )
    raise HTTPException(status_code=404, detail="Document file not found")

def extract_text_from_file(content: bytes, filename: str) -> str:
    file_ext = os.path.splitext(filename.lower())[1]
    
    if file_ext in ['.txt', '.md']:
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                text = content.decode(encoding, errors='ignore')
                if text.strip():
                    return text
            except:
                continue
        raise ValueError("Could not decode text file")
    
    elif file_ext == '.pdf':
        try:
            import PyPDF2
            from io import BytesIO
            
            pdf_file = BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except ImportError:
            raise ValueError("PyPDF2 not installed. Install with: pip install PyPDF2")
        except Exception as e:
            raise ValueError(f"Failed to parse PDF: {str(e)}")
    
    elif file_ext == '.docx':
        try:
            import docx
            from io import BytesIO
            
            doc_file = BytesIO(content)
            doc = docx.Document(doc_file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text.strip()
        except ImportError:
            raise ValueError("python-docx not installed. Install with: pip install python-docx")
        except Exception as e:
            raise ValueError(f"Failed to parse DOCX: {str(e)}")
    
    elif file_ext == '.doc':
        raise ValueError("Legacy .doc format not supported. Please convert to .docx or .txt format")
    
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

@app.post("/upload-files")
async def upload_files(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    if not rag_system:
        raise HTTPException(status_code=503, detail="GraphRAG system not available")
    
    results = []
    failed_files = []
    
    for file in files:
        try:
            print(f"Processing: {file.filename}")
            
            if not file.filename:
                failed_files.append({"name": "unnamed_file", "reason": "No filename"})
                continue
            
            file_ext = os.path.splitext(file.filename.lower())[1]
            allowed_exts = {'.txt', '.md', '.pdf', '.docx'}
            
            if file_ext not in allowed_exts:
                failed_files.append({
                    "name": file.filename,
                    "reason": f"Unsupported format: {file_ext}. Supported: .txt, .md, .pdf, .docx"
                })
                continue
            
            content = await file.read()
            if not content:
                failed_files.append({"name": file.filename, "reason": "Empty file"})
                continue
            
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, 'wb') as f:
                f.write(content)
            
            try:
                text_content = extract_text_from_file(content, file.filename)
                
                if not text_content or len(text_content.strip()) < 10:
                    failed_files.append({"name": file.filename, "reason": "No readable text found"})
                    continue
                
            except ValueError as e:
                failed_files.append({"name": file.filename, "reason": str(e)})
                continue
            except Exception as e:
                failed_files.append({"name": file.filename, "reason": f"Parse error: {str(e)}"})
                continue
            
            try:
                rag_system.store_document(file.filename, text_content)
                
                doc_id = None
                if hasattr(rag_system, 'neo4j') and hasattr(rag_system.neo4j, 'driver') and rag_system.neo4j.driver:
                    with rag_system.neo4j.driver.session() as session:
                        result = session.run(
                            "MATCH (d:Document {title: $title}) RETURN d.id as id",
                            title=file.filename
                        )
                        record = result.single()
                        if record:
                            doc_id = record["id"]
                            file_storage[doc_id] = file_path
                
                results.append(file.filename)
                print(f"Stored: {file.filename}")
                
                if postgres and postgres.conn:
                    try:
                        with postgres.conn.cursor() as cursor:
                            cursor.execute(
                                "INSERT INTO documents (title, content) VALUES (%s, %s)",
                                (file.filename, text_content[:5000])
                            )
                            postgres.conn.commit()
                    except Exception as e:
                        print(f"PostgreSQL backup warning: {e}")
                        
            except Exception as e:
                print(f"Storage error for {file.filename}: {e}")
                failed_files.append({"name": file.filename, "reason": f"Storage failed: {str(e)}"})
                
        except Exception as e:
            print(f"Unexpected error processing {file.filename}: {e}")
            failed_files.append({"name": file.filename, "reason": f"Processing error: {str(e)}"})
    
    if not results and failed_files:
        error_summary = "; ".join([f"{f['name']}: {f['reason']}" for f in failed_files[:3]])
        raise HTTPException(status_code=400, detail=f"All files failed. Errors: {error_summary}")
    
    message = f"Successfully processed {len(results)}/{len(files)} file(s)"
    if failed_files:
        message += f". {len(failed_files)} file(s) failed"
    
    return {
        "message": message,
        "processed_files": results,
        "failed_files": failed_files,
        "success_count": len(results),
        "failure_count": len(failed_files)
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not rag_system:
        raise HTTPException(status_code=503, detail="GraphRAG system not available")
    
    try:
        print(f"Chat request: {request.question}")
        result = rag_system.chat(request.question)
        
        print(f"Chat result keys: {result.keys()}")
        print(f"Sources in result: {result.get('sources', [])}")
        
        if postgres and postgres.conn:
            try:
                with postgres.conn.cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO chat_history (question, answer) VALUES (%s, %s)",
                        (result["question"], result["answer"])
                    )
                    postgres.conn.commit()
            except Exception as e:
                print(f"Chat history save failed: {e}")
        
        sources = []
        if result.get("sources"):
            sources = [SourceDocument(title=s["title"], doc_id=s["doc_id"]) for s in result["sources"]]
            print(f"Returning {len(sources)} sources to frontend")
        else:
            print("No sources found in result")
        
        response = ChatResponse(
            question=result["question"],
            answer=result["answer"],
            context=result.get("context", []),
            sources=sources
        )
        
        print(f"Final response has {len(response.sources) if response.sources else 0} sources")
        
        return response
        
    except Exception as e:
        print(f"Chat error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.delete("/documents")
async def clear_documents():
    if not rag_system:
        raise HTTPException(status_code=503, detail="GraphRAG system not available")
    
    try:
        if hasattr(rag_system, 'neo4j') and hasattr(rag_system.neo4j, 'driver') and rag_system.neo4j.driver:
            with rag_system.neo4j.driver.session() as session:
                count_result = session.run("MATCH (d:Document) RETURN count(d) as count")
                doc_count = count_result.single()["count"]
                
                session.run("MATCH (n) DETACH DELETE n")
                
                if postgres and postgres.conn:
                    with postgres.conn.cursor() as cursor:
                        cursor.execute("DELETE FROM documents")
                        postgres.conn.commit()
                
                for file_path in file_storage.values():
                    if os.path.exists(file_path):
                        os.remove(file_path)
                file_storage.clear()
                
                return {"message": f"Cleared {doc_count} documents"}
        else:
            return {"message": "No Neo4j connection to clear"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clear failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")