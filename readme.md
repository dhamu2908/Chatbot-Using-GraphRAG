# GraphRAG Chatbot

A document question-answering system that uses Neo4j graph database, semantic embeddings, and Large Language Models to provide accurate answers with source attribution.

## Features

- Upload documents (TXT, PDF, DOCX, MD) and ask questions about their content
- Semantic search using 768-dimensional embeddings
- Source attribution - see which documents contributed to each answer
- Fuzzy matching for handling typos and misspellings
- Clean, modern web interface
- Auto-cleanup of old data on server restart

## Architecture

```
Document Upload → Text Extraction → Chunking → Embedding Generation → Neo4j Storage
Query → Entity Detection → Fuzzy Matching → Semantic Search → Re-ranking → Answer Generation
```

**Technology Stack:**
- Backend: FastAPI (Python)
- Graph Database: Neo4j 5.13
- Relational Database: PostgreSQL 13
- AI Runtime: Ollama
- Embedding Model: nomic-embed-text (768-dim)
- LLM: phi3:mini or gemma:1b
- Frontend: HTML/CSS/JavaScript

## Prerequisites

1. **Python 3.8+**
2. **Docker & Docker Compose**
3. **Ollama** - [Download here](https://ollama.com/download)

## Installation

### Step 1: Clone and Setup

```bash
cd your-project-directory
pip install -r requirements.txt
```

### Step 2: Install Ollama Models

```bash
# Embedding model (required)
ollama pull nomic-embed-text

# Choose one LLM:
ollama pull phi3:mini    # For 4GB GPU
ollama pull gemma:1b     # For 2GB GPU or CPU-only
```

### Step 3: Configure Environment

Create a `.env` file in the project root:

```env
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password123

# PostgreSQL Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=graphrag
POSTGRES_USER=admin
POSTGRES_PASSWORD=password123

# Ollama Configuration
MODEL_NAME=phi3:mini
OLLAMA_BASE_URL=http://localhost:11434
```

### Step 4: Start Databases

```bash
# Start Neo4j and PostgreSQL
docker-compose up -d

# Verify containers are running
docker ps
```

You should see `neo4j-graphrag` and `postgres-graphrag` containers.

### Step 5: Start Server

```bash
python start_server.py
```

Wait for the message: "Started." This confirms Neo4j is ready.

### Step 6: Open Interface

Open `chatbot_ui.html` in your web browser.

## Project Structure

```
graphrag-chatbot/
├── app/
│   ├── main.py          # FastAPI server
│   ├── rag.py           # RAG logic with embeddings
│   ├── database.py      # Database connections
│   └── models.py        # Data models
├── uploaded_files/      # Stored documents
├── .env                 # Configuration
├── docker-compose.yml   # Database containers
├── requirements.txt     # Python dependencies
├── start_server.py      # Server launcher
├── chatbot_ui.html      # Web interface
└── README.md           # This file
```

## Usage

### Upload Documents

1. Click "Choose Files" in the web interface
2. Select one or more documents (TXT, PDF, DOCX, MD)
3. Click "Upload Documents"
4. Wait for processing to complete

### Ask Questions

1. Type your question in the input box
2. Press Enter or click the send button
3. View the answer and source documents

### Clear Data

Click "Clear All Documents" to remove all uploaded documents and start fresh.

## API Endpoints

### Check Status
```http
GET http://localhost:8000/status
```

Returns connection status for Neo4j, PostgreSQL, and Ollama.

### Upload Documents
```http
POST http://localhost:8000/upload-files
Content-Type: multipart/form-data
```

### Ask Question
```http
POST http://localhost:8000/chat
Content-Type: application/json

{
  "question": "What is artificial intelligence?"
}
```

### List Documents
```http
GET http://localhost:8000/documents
```

### Clear All Documents
```http
DELETE http://localhost:8000/documents
```

### Debug Graph Status
```http
GET http://localhost:8000/debug/graph-status
```

Shows node counts and document list in Neo4j.

## Troubleshooting

### Server Won't Start

**Port already in use:**
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

### Neo4j Authentication Error

```bash
# Reset database completely
docker-compose down -v
docker-compose up -d

# Wait 20 seconds for Neo4j to initialize
python start_server.py
```

### Out of Memory Error

If you see "MemoryPoolOutOfMemoryError" when clearing documents:

```bash
# Nuclear option - complete reset
docker-compose down -v
docker-compose up -d
```

This removes all data but takes only 30 seconds.

### Database Not Clearing

The updated code deletes in batches of 10,000 nodes to avoid memory issues. For large databases, this may take several minutes. Check the terminal for progress messages.

### Ollama Not Found

```bash
# Check if Ollama is running
ollama list

# If not installed, download from https://ollama.com/download
```

### Import Errors

Make sure you're running the server from the project root directory:

```bash
cd /path/to/project
python start_server.py
```

Not from inside the `app/` directory.

## Configuration

### Change LLM Model

Edit `.env`:
```env
MODEL_NAME=gemma:1b    # or phi3:mini, gemma:4b, etc.
```

Then restart the server.

### Adjust Retrieval Parameters

In `app/rag.py`, modify the `retrieve_with_reranking` function:
- `initial_k`: Number of candidate chunks (default: 20)
- `final_k`: Number of chunks to use for answer (default: 8)
- `batch_size`: Fuzzy match threshold (default: 0.7 = 70% similarity)

## Performance

### Response Times (GPU)
- Document upload: 2-5 seconds per document
- Query processing: 3-5 seconds

### Response Times (CPU Only)
- Document upload: 10-15 seconds per document
- Query processing: 15-30 seconds

### Scalability
- 10 documents: ~1 minute upload, 3s queries
- 100 documents: ~10 minutes upload, 4s queries
- 1,000 documents: ~2 hours upload, 6s queries

## Key Features Explained

### Fuzzy Matching

The system handles typos and misspellings using Levenshtein distance:
- "Shubman Gill" matches "Shubman Gil" (70%+ similarity)
- "Ambedkar" matches "Ambedker", "Amdedkar"

### Entity Detection

Automatically extracts names, places, and concepts:
- Capitalized phrases: "Shubman Gill"
- Single names: "Chahal", "Ambedkar"
- Lowercase names: "yuzi chahal" → "Yuzi Chahal"

### Semantic Search

Uses 768-dimensional embeddings to find relevant chunks based on meaning, not just keywords.

### Re-ranking Algorithm

Combines multiple signals:
- Semantic similarity (cosine distance)
- Keyword matches from graph
- Direct text content matches
- Fuzzy match scores

## Database Schema

### Neo4j Graph Structure

```cypher
(Document {id, title})
  -[:CONTAINS]→
(Chunk {id, content, embedding[768]})
  -[:HAS_KEYWORD]→
(Keyword {name})
```

### PostgreSQL Tables

```sql
documents (id, title, content, created_at)
chat_history (id, question, answer, created_at)
```

## Maintenance

### Clear Old Data on Startup

The server automatically clears the database when it starts to ensure a clean state.

To disable this, comment out the cleanup call in `start_server.py`:
```python
# clear_database_on_startup()  # Commented out
```

### View Database Contents

**Neo4j Browser:** http://localhost:7474
- Username: neo4j
- Password: password123

**PostgreSQL:**
```bash
docker exec -it postgres-graphrag psql -U admin -d graphrag
```

### Backup Data

```bash
# Backup Neo4j
docker exec neo4j-graphrag neo4j-admin database dump neo4j --to-path=/backups

# Backup PostgreSQL
docker exec postgres-graphrag pg_dump -U admin graphrag > backup.sql
```

## Advanced Configuration

### Increase Neo4j Memory

Edit `docker-compose.yml`:
```yaml
neo4j:
  environment:
    - NEO4J_dbms_memory_heap_max__size=2G
    - NEO4J_dbms_memory_pagecache_size=1G
```

Then restart:
```bash
docker-compose down
docker-compose up -d
```

### Change Chunk Size

In `app/rag.py`, modify `chunk_text()`:
```python
def chunk_text(self, text: str, max_chunk_size: int = 150):
    # Change 150 to your desired size
```

Smaller chunks = more precise but less context.
Larger chunks = more context but less precise.

## Support

For issues:
1. Check the troubleshooting section above
2. Review terminal logs for error messages
3. Check database status: http://localhost:8000/debug/graph-status
4. Verify services: `docker ps` and `ollama list`

## License

MIT License - free to use and modify.

## Credits

Built with:
- [FastAPI](https://fastapi.tiangolo.com/)
- [Neo4j](https://neo4j.com/)
- [Ollama](https://ollama.com/)
- [PostgreSQL](https://www.postgresql.org/)
