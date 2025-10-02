# Advanced GraphRAG Chatbot with Semantic Search

A production-ready document Q&A system combining Graph databases (Neo4j), semantic embeddings, and Large Language Models for intelligent, context-aware responses with source attribution.

## Features

- **Semantic Search with Embeddings**: 768-dimensional vector embeddings for meaning-based retrieval
- **Multi-Stage Retrieval**: Query understanding → Embedding search → Re-ranking → Answer generation
- **Source Attribution**: Clickable links to original documents, ranked by contribution
- **Query Intent Detection**: Automatically detects question types (who/what/when/where/why/how)
- **GPU Acceleration**: Optimized for NVIDIA GPUs (4GB+) with 5-10x speedup
- **Hybrid Architecture**: Combines keyword matching with semantic similarity
- **Answer Validation**: Verifies responses are relevant and grounded in context
- **Document Structure Awareness**: Handles legal documents, articles, and plain text
- **Auto-Cleanup**: Removes documents when session ends

---

## System Architecture

```
Document Upload Flow:
User Upload → Text Extraction → Sentence-Aware Chunking → Embedding Generation → Neo4j Storage

Query Flow:
User Query → Intent Analysis → Query Embedding → Semantic Search → Re-ranking → Answer Generation → Response
     ↓              ↓                ↓                 ↓              ↓              ↓
  (LLM)          (LLM)       (Embedding Model)    (Cosine Sim)   (Algorithm)      (LLM)
```

### Technology Stack

**Backend:**
- FastAPI (Python web framework)
- Neo4j 5.13 (Graph database)
- PostgreSQL 13 (Relational database)
- Ollama (LLM inference engine)

**AI Models:**
- **Main LLM**: phi3:mini (3.8B) or gemma:4b for query understanding & answer generation
- **Embeddings**: nomic-embed-text (768-dim vectors) for semantic search

**Frontend:**
- Pure HTML/CSS/JavaScript

### Graph Database Structure

```cypher
(Document {id, title, doc_type})
  -[:CONTAINS]→
(Chunk {id, content, embedding[768]})
  -[:HAS_KEYWORD]→
(Keyword {name})
```

---

## Prerequisites

### Required Software

1. **Python 3.8+**
2. **Docker & Docker Compose**
3. **Ollama** - Download from https://ollama.com/download
4. **NVIDIA GPU (Recommended)** - 4GB+ VRAM with CUDA 11.8+ drivers

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8GB | 16GB |
| GPU VRAM | 4GB | 8GB+ |
| Storage | 10GB | 20GB |
| CPU | 4 cores | 8+ cores |

---

## Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Install Ollama Models

```bash
# Embedding model (required)
ollama pull nomic-embed-text

# Choose LLM based on GPU:
# For 4GB GPU (RTX 2050/3050)
ollama pull phi3:mini

# For 8GB+ GPU
ollama pull gemma:4b
```

### Step 3: Start Databases

```bash
docker-compose up -d

# Verify
docker ps
```

### Step 4: Configure Environment

Create `.env` file:

```env
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password123

# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=graphrag
POSTGRES_USER=admin
POSTGRES_PASSWORD=password123

# Ollama
MODEL_NAME=phi3:mini
OLLAMA_BASE_URL=http://localhost:11434
```

### Step 5: Start Server

```bash
python start_server.py
```

### Step 6: Open Interface

Open `chatbot_ui.html` in your browser.

---

## How It Works

### 1. Document Processing

Upload → Extract Text → Chunk into ~150 words → Generate 768-dim embeddings → Store in Neo4j

### 2. Query Understanding

When you ask "Who is Ambedkar?":
- LLM extracts intent: "who" (person identification)
- Identifies entities: ["Ambedkar"]
- Generates variations: "B.R. Ambedkar", "Dr. Ambedkar", "Ambedkar was"

### 3. Semantic Retrieval (Two-Stage)

**Stage 1: Candidate Retrieval**
- Generate embeddings for query variations
- Search Neo4j for keyword matches
- Get 15 initial candidates

**Stage 2: Re-ranking**
- Calculate cosine similarity with query embeddings
- Apply relevance signals:
  - +0.2: Direct answer patterns ("Ambedkar is...")
  - +0.1: Entity matches
  - -0.15: Indirect mentions ("someone said Ambedkar...")
- Return top 5 chunks

### 4. Answer Generation

Top chunks + Question → LLM → Answer with sources

### 5. Source Attribution

Documents ranked by contribution (similarity score × content length)

---

## Configuration

### GPU Optimization

**For 4GB GPU:**
```env
MODEL_NAME=phi3:mini
```

**For 8GB+ GPU:**
```env
MODEL_NAME=gemma:4b
```

### Performance Comparison

| Model | VRAM | Speed | Quality |
|-------|------|-------|---------|
| phi3:mini | 2.5GB | 40-60 tok/s | Excellent |
| gemma:4b | 3.5GB | 30-50 tok/s | Superior |

---

## Usage Guide

### Basic Workflow

1. **Upload Documents** - Select .txt, .pdf, .docx, or .md files
2. **Ask Questions** - Type in input box and press Enter
3. **Access Sources** - Click document links below answers

### Supported Question Types

- Identity: "Who is [person]?"
- Definition: "What is [concept]?"
- Factual: "When was [event]?"
- Process: "How to [action]?"

---

## API Endpoints

### Base URL
```
http://localhost:8000
```

### Key Endpoints

**Health Check**
```http
GET /
```

**System Status**
```http
GET /status
```

**Upload Documents**
```http
POST /upload-files
Content-Type: multipart/form-data
```

**Chat**
```http
POST /chat
Content-Type: application/json

{
  "question": "Who is Ambedkar?"
}
```

**List Documents**
```http
GET /documents
```

**Get Document**
```http
GET /document/{doc_id}
```

**Clear Documents**
```http
DELETE /documents
```

---

## Performance & Scaling

### Response Times

| Operation | CPU | GPU (4GB) |
|-----------|-----|-----------|
| Upload (per doc) | 10-15s | 2-3s |
| Query processing | 20-30s | 3-5s |

### Scalability

| Documents | Chunks | Upload Time | Query Time |
|-----------|--------|-------------|------------|
| 10 | ~100 | 30s | 3s |
| 100 | ~1,000 | 5min | 4s |
| 1,000 | ~10,000 | 45min | 6s |
| 10,000 | ~100,000 | 7hrs | 10s |

---

## Troubleshooting

### Server Won't Start

**Port in use:**
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

### GPU Not Being Used

**Check GPU:**
```bash
nvidia-smi
```

**Fix:** Reinstall Ollama from https://ollama.com/download

### Wrong Answers

1. Verify correct document uploaded
2. Check terminal logs
3. Try rephrasing question
4. Clear and re-upload:
   ```bash
   docker-compose down -v
   docker-compose up -d
   ```

### Slow Performance

1. Use GPU if available
2. Switch to smaller model: `MODEL_NAME=phi3:mini`
3. Close other applications

### Out of Memory

1. Use smaller model: `ollama pull phi3:mini`
2. Reduce retrieval: Set `final_k=3` in `rag.py`

---

## Project Structure

```
graphrag-chatbot/
├── app/
│   ├── main.py              # FastAPI server
│   ├── rag.py               # RAG logic with embeddings
│   ├── database.py          # Database connections
│   └── models.py            # Data models
├── uploaded_files/          # Stored documents
├── .env                     # Configuration
├── docker-compose.yml       # Database containers
├── requirements.txt         # Dependencies
├── start_server.py          # Server launcher
├── chatbot_ui.html          # Web interface
└── README.md               # This file
```

---

## Advanced Features

### Custom Prompting

Edit prompts in `rag.py`:
```python
prompt = f"""Your custom instructions...
Context: {context_text}
Question: {question}
Answer:"""
```

### Adding Document Types

In `main.py`, add parsers for new formats:
```python
if file_ext == '.csv':
    import pandas as pd
    return pd.read_csv(BytesIO(content)).to_string()
```

---

## Architecture Details

### GraphRAG Implementation

1. **Graph Storage**: Documents as interconnected nodes in Neo4j
2. **Semantic Search**: 768-dim vector embeddings for meaning-based retrieval
3. **Re-ranking**: Multi-signal relevance scoring
4. **Source Tracking**: Contribution-based document ranking

### Tech Stack

- **Backend**: FastAPI (Python)
- **Graph DB**: Neo4j
- **Relational DB**: PostgreSQL
- **AI Runtime**: Ollama
- **Frontend**: HTML/CSS/JavaScript
- **Containers**: Docker Compose

---

## License

MIT License - free to use and modify.

## Contributing

1. Fork repository
2. Create feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

## Support

**For issues:**
1. Check troubleshooting section
2. Review terminal logs
3. Verify system status: http://localhost:8000/status
4. Ensure all services running: `docker ps` and `nvidia-smi`

**Resources:**
- Ollama: https://ollama.com/docs
- Neo4j: https://neo4j.com/docs
- FastAPI: https://fastapi.tiangolo.com

---

