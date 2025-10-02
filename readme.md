# GraphRAG Chatbot

A sophisticated chatbot that combines Graph databases (Neo4j) with Retrieval-Augmented Generation (RAG) using Ollama for intelligent, context-aware responses.

## Features

- **GraphRAG Architecture**: Uses Neo4j to store documents in graph format with relationships
- **AI-Powered**: Integrates with Ollama for natural language generation
- **Modern Web UI**: Beautiful, responsive chat interface
- **Real-time Status**: Monitor system health and connections
- **Persistent Storage**: PostgreSQL for chat history and backups
- **Easy Setup**: Automated installation and configuration

## System Architecture

```
User Query → Keyword Extraction → Graph Search → Context Retrieval → AI Generation → Response
                    ↓                    ↓              ↓               ↓
               (Ollama)            (Neo4j Graph)   (Document Chunks)   (Ollama)
```

## Quick Start

### Prerequisites

- Python 3.8+
- Docker & Docker Compose
- Ollama installed

### 1. Automated Setup

```bash
# Clone/download the project
# Navigate to project directory

# Run the complete setup
python setup.py
```

### 2. Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start databases
docker-compose up -d

# Install Ollama model
ollama pull gemma:1b

# Start the server
python start_server.py
```

### 3. Open the Chat Interface

Open `chatbot_ui.html` in your web browser

## Project Structure

```
graphrag-chatbot/
├── app/                     # Main application package
│   ├── __init__.py         # Package initializer
│   ├── main.py             # FastAPI application
│   ├── rag.py              # GraphRAG implementation
│   ├── database.py         # Database connections
│   └── models.py           # Data models
├── data/
│   └── sample_data.txt     # Sample documents
├── .env                    # Environment configuration
├── docker-compose.yml      # Database services
├── requirements.txt        # Python dependencies
├── setup.py               # Automated setup script
├── start_server.py        # Server startup
├── chatbot_ui.html        # Web interface
└── README.md              # This file
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/status` | GET | System status (Neo4j, PostgreSQL, Ollama) |
| `/chat` | POST | Send message to chatbot |
| `/documents` | POST | Add new document to knowledge base |
| `/history` | GET | Retrieve chat history |
| `/add-sample-data` | POST | Load sample documents for testing |

## How It Works

### 1. Document Storage
- Documents are chunked into smaller pieces
- Keywords are extracted using Ollama
- Data is stored in Neo4j as a graph:
  ```
  Document → Contains → Chunks → Has_Keyword → Keywords
  ```

### 2. Query Processing
- User question is analyzed for keywords
- Graph database is searched for relevant chunks
- Context is retrieved based on keyword relationships

### 3. Response Generation
- Retrieved context is combined with user question
- Ollama generates contextually aware response
- Answer is returned with source information

## Configuration

### Environment Variables (.env)

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
MODEL_NAME=gemma:1b
OLLAMA_BASE_URL=http://localhost:11434
```

### Docker Services

The system uses Docker Compose to manage:
- **Neo4j**: Graph database for document relationships
- **PostgreSQL**: Relational database for chat history

## Usage Examples

### Adding Documents via API

```bash
curl -X POST "http://localhost:8000/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "My Document",
    "content": "Document content here..."
  }'
```

### Chatting via API

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is artificial intelligence?"
  }'
```

### Using the Web Interface

1. Open `chatbot_ui.html` in your browser
2. Click "Add Sample Documents" to load test data
3. Type your question and press Enter
4. View contextual responses with source information

## Troubleshooting

### Common Issues

**Server won't start:**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check if ports 8000, 7474, 7687, 5432 are available
- Verify Docker is running: `docker ps`

**Chatbot gives generic responses:**
- Make sure Neo4j is connected (check `/status` endpoint)
- Add documents using "Add Sample Documents" or `/documents` endpoint
- Verify Ollama is running: `ollama list`

**Database connection errors:**
- Check Docker containers: `docker-compose ps`
- Restart databases: `docker-compose restart`
- Verify credentials in `.env` file

### Checking System Health

Visit `http://localhost:8000/status` to see:
- Neo4j connection status
- PostgreSQL connection status  
- Ollama model availability
- Current configuration

## Development

### Running in Development Mode

```bash
# Start with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Testing the System

```bash
# Run the setup script with testing
python setup.py

# Or test manually
python -c "
import requests
response = requests.get('http://localhost:8000/status')
print(response.json())
"
```

## Architecture Details

### GraphRAG Implementation

This system implements a true GraphRAG architecture:

1. **Graph Storage**: Documents stored as interconnected nodes in Neo4j
2. **Relationship Traversal**: Queries traverse keyword relationships
3. **Context Ranking**: Results ranked by relevance and connection strength
4. **Augmented Generation**: Retrieved context enhances AI responses

### Tech Stack

- **Backend**: FastAPI (Python)
- **Graph Database**: Neo4j
- **Relational Database**: PostgreSQL  
- **AI Model**: Ollama (Gemma)
- **Frontend**: HTML/CSS/JavaScript
- **Containerization**: Docker Compose

## License

MIT License - feel free to use and modify for your projects.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Verify system status at `/status` endpoint
3. Review logs in the terminal
4. Ensure all services are running properly

## Screenshots from the project

<p align="center">
  <img src="https://github.com/user-attachments/assets/911e3ba7-8c1e-416c-95b1-fec9f0e9e5c5" alt="Chatbot Screenshot" width="600"/>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/df1f54f8-7d49-499e-a0c9-b7b6c577266a" alt="Chatbot Screenshot" width="600"/>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/1b402ac0-dae3-41ac-80e4-f370339db8d7" alt="Chatbot Screenshot" width="600"/>
</p>




