# Chat With PDF Backend

A sophisticated multi-agent RAG (Retrieval-Augmented Generation) system that enables intelligent question-answering over academic papers. Built with FastAPI, LangGraph, and LangChain, this system provides a "Chat With PDF" assistant capable of handling complex queries, performing web searches, and providing clarification for ambiguous questions.

## 🚀 Features

- **Multi-Agent Architecture**: Uses LangGraph to orchestrate specialized agents for optimal responses
- **PDF-First RAG**: Intelligent retrieval and generation from academic papers
- **Web Search Integration**: Falls back to web search for current information not in PDFs
- **Clarification Agent**: Handles ambiguous queries by requesting specific details
- **Session Memory**: Maintains conversation context within user sessions
- **RESTful API**: Clean API endpoints for integration
- **Docker Support**: Easy deployment with Docker and docker-compose

## 🏗️ Architecture Overview

The system employs a multi-agent architecture using LangGraph for orchestration:

### Agents

1. **Query Router**: Analyzes incoming questions and routes them to the appropriate agent
2. **PDF Agent**: Searches and answers questions from embedded academic papers
3. **Web Search Agent**: Performs web searches for current information using Tavily API
4. **Clarification Agent**: Detects ambiguous questions and requests clarification

### Agent Workflow

```mermaid
graph TD
    A[User Question] --> B[Query Router]
    B --> C{Route Decision}
    C -->|Research Question| D[PDF Agent]
    C -->|Current Events| E[Web Search Agent]
    C -->|Ambiguous| F[Clarification Agent]
    D --> G[Vector Search]
    G --> H[Generate Answer]
    E --> I[Web Search]
    I --> J[Synthesize Results]
    F --> K[Generate Clarification]
    H --> L[Final Response]
    J --> L
    K --> L
```

### Technology Stack

- **FastAPI**: High-performance web framework
- **LangGraph**: Agent orchestration and workflow management
- **LangChain**: LLM integration and document processing
- **ChromaDB**: Vector database for document embeddings
- **OpenAI GPT-4o mini**: Large language model for clarification detection
- **OpenAI GPT-4**: Large language model for generation
- **Tavily API**: Web search capabilities
- **Docker**: Containerization and deployment

## 📋 Requirements

- Python 3.11+
- Docker and docker-compose
- OpenAI API key
- Tavily API key (optional, will use mock results without it)

## 🛠️ Installation & Setup

### 1. Clone and Setup

```bash
git clone <repository-url>
cd arcfusion-test
```

### 2. Environment Configuration

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```env
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### 3. Local Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Ingest PDF documents
python scripts/ingest_pdfs.py

# Run the application
python -m uvicorn app.main:app --reload
```

### 4. Docker Deployment

```bash
# Build and run with docker-compose
docker-compose up --build

# Or run just the application
docker build -t chat-with-pdf .
docker run -p 8000:8000 --env-file .env chat-with-pdf
```

> **Note**: If you encounter NumPy compatibility errors (`np.float_` was removed in NumPy 2.0), add `numpy<2.0.0` to your requirements.txt file to pin NumPy to a compatible version.

## 🐳 Running with Docker Compose

```bash
# Start the system
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the system
docker-compose down
```

The application will be available at `http://localhost:8000`

> **Important**: The Docker setup automatically runs the PDF ingestion process during startup. This ensures the vector database is populated before the API becomes available.

## 📚 API Documentation

Once running, visit:
- **Interactive API Docs**: `http://localhost:8000/docs`
- **OpenAPI Schema**: `http://localhost:8000/openapi.json`

### Key Endpoints

#### Ask a Question
```bash
POST /api/v1/ask
{
    "question": "What prompt template gave the highest zero-shot accuracy on Spider in Zhang et al. (2024)?",
    "session_id": "optional-session-id"
}
```

#### Clear Session Memory
```bash
POST /api/v1/clear-memory
{
    "session_id": "session-id-to-clear"
}
```

#### Health Check
```bash
GET /api/v1/health
```

#### Get All Chunks
```bash
GET /api/v1/chunks?limit=10
```

## 🧪 Testing the System

### Example Queries

1. **PDF-Only Query**:
   ```
   "Which prompt template gave the highest zero-shot accuracy on Spider in Zhang et al. (2024)?"
   ```

2. **Ambiguous Query** (triggers clarification):
   ```
   "How many examples are enough for good accuracy?"
   ```

3. **Web Search Query**:
   ```
   "What did OpenAI release this month?"
   ```

### Real-World Test Scenarios

The system handles the scenarios mentioned in the assignment:

1. **Ambiguous Questions**: Detects vague terms and requests clarification
2. **PDF-Only Queries**: Searches embedded research papers for specific findings
3. **Out-of-Scope Queries**: Routes to web search for current information

## 🔧 System Configuration

### PDF Processing

- **Chunk Size**: 1000 characters (configurable)
- **Chunk Overlap**: 200 characters (configurable)
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`

### Agent Settings

- **Router Model**: `gpt-4-turbo-preview`
- **Clarification Model**: `gpt-4o-mini` (optimized for clarification detection)
- **PDF/Web Agent Model**: `gpt-4-turbo-preview`
- **Max Iterations**: 10
- **Session TTL**: 1 hour
- **Max Session Memory**: 50 messages

### Vector Database

- **Engine**: ChromaDB
- **Persistence**: Local file system
- **Similarity Search**: Cosine similarity

## 📁 Project Structure

```
arcfusion-test/
├── app/
│   ├── agents/           # Multi-agent implementation
│   │   ├── router.py     # Query routing logic
│   │   ├── pdf_agent.py  # PDF retrieval agent
│   │   ├── web_agent.py  # Web search agent
│   │   ├── clarification.py # Clarification agent
│   │   └── graph.py      # LangGraph orchestration
│   ├── api/
│   │   └── endpoints.py  # FastAPI routes
│   ├── models/
│   │   └── schemas.py    # Pydantic models
│   ├── services/
│   │   ├── pdf_ingestion.py # PDF processing
│   │   ├── vectorstore.py    # Vector database
│   │   └── session.py        # Session management
│   ├── config.py         # Configuration
│   └── main.py          # FastAPI application
├── scripts/
│   └── ingest_pdfs.py   # PDF ingestion script
├── papers/              # PDF documents
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 🔄 How It Works

### 1. Document Ingestion
- PDFs are loaded and chunked using LangChain's text splitters
- Chunks are embedded using SentenceTransformers
- Embeddings are stored in ChromaDB with metadata

### 2. Query Processing
- User questions are routed through the LangGraph workflow
- Router determines the best agent based on query analysis
- Selected agent processes the query and generates a response
- Session memory maintains conversation context

### 3. Multi-Agent Coordination
- **Router Agent**: Analyzes queries and determines routing
- **PDF Agent**: Searches vector database and generates answers
- **Web Agent**: Performs web search and synthesizes results  
- **Clarification Agent**: Handles ambiguous queries

### 4. Response Generation
- Agents use OpenAI GPT-4 for natural language generation
- Sources are tracked and returned with responses
- Confidence scores help indicate answer reliability

## 🔍 System Trade-offs

### Architecture Trade-offs

| Design Choice | Benefits | Drawbacks | Considerations |
|---------------|----------|-----------|----------------|
| **Multi-Agent Architecture** | - Specialized handling for different query types<br>- Clear separation of concerns<br>- Modular and extensible | - Increased complexity<br>- Higher latency due to routing<br>- More API calls | Best for systems with diverse query types requiring specialized handling |
| **Synchronous API** | - Simpler implementation<br>- Direct response flow<br>- Easier debugging | - Longer response times<br>- Potential timeouts for complex queries | Consider async endpoints for production with long-running queries |
| **In-Memory Session Management** | - Fast access<br>- Simple implementation<br>- No external dependencies | - Limited by server memory<br>- Lost on server restart<br>- Not scalable across instances | Suitable for prototyping; use Redis/DB for production |
| **Local Vector Database** | - Simple setup<br>- No external service dependencies<br>- Good for development | - Limited scalability<br>- Performance bottlenecks with large datasets<br>- No built-in redundancy | Works well for small to medium document collections |

### Model Selection Trade-offs

| Model Choice | Benefits | Drawbacks | Use Cases |
|--------------|----------|-----------|-----------|
| **GPT-4-turbo for main agents** | - High accuracy<br>- Sophisticated reasoning<br>- Good context handling | - Higher cost<br>- Longer latency<br>- API rate limits | Complex queries requiring deep understanding |
| **GPT-4o-mini for clarification** | - Lower cost<br>- Faster response times<br>- Sufficient for detection tasks | - Less sophisticated reasoning<br>- Smaller context window | Pattern recognition and classification tasks |
| **Sentence Transformers for embeddings** | - Open source<br>- Fast embedding generation<br>- No API costs | - Less powerful than latest models<br>- Fixed model size | Document indexing and retrieval |

### Deployment Trade-offs

| Deployment Option | Benefits | Drawbacks | Best For |
|-------------------|----------|-----------|----------|
| **Docker Compose** | - Simple setup<br>- Consistent environment<br>- Good for development | - Limited scalability<br>- Manual scaling<br>- Basic orchestration | Development, testing, small deployments |
| **Kubernetes** | - Horizontal scaling<br>- Advanced orchestration<br>- Production-ready | - Complex setup<br>- Resource overhead<br>- Learning curve | Production environments with scaling needs |
| **Serverless** | - Auto-scaling<br>- Pay-per-use<br>- No infrastructure management | - Cold start latency<br>- Timeout limitations<br>- Vendor lock-in | Sporadic usage patterns, cost optimization |

## 🛣️ Next Steps & Roadmap

### Short-term Improvements (1-2 Months)

1. **Performance Optimization**
   - **Memory Usage**: Implement streaming responses to reduce memory footprint
   - **Caching Layer**: Add Redis caching for frequent queries and embeddings
   - **Batch Processing**: Optimize PDF ingestion with parallel processing
   - **Action**: Implement Redis caching and streaming responses

2. **Reliability Enhancements**
   - **Error Recovery**: Add retry mechanisms for API calls
   - **Fallback Strategies**: Implement model fallbacks for API rate limits
   - **Monitoring**: Add comprehensive logging and alerting
   - **Action**: Set up monitoring dashboard and implement retry logic

3. **User Experience**
   - **Response Quality**: Fine-tune prompts for more accurate answers
   - **Feedback Loop**: Add user feedback collection mechanism
   - **Action**: Implement A/B testing framework for prompt optimization

### Mid-term Goals (3-6 Months)

1. **Scalability**
   - **Distributed Architecture**: Split services into microservices
   - **Load Balancing**: Implement horizontal scaling for API endpoints
   - **Database Migration**: Move from local ChromaDB to managed vector database
   - **Action**: Refactor into microservices with Kubernetes deployment

2. **Feature Expansion**
   - **Document Management**: Add API for document upload and management
   - **User Management**: Implement authentication and authorization
   - **Multi-Collection Support**: Allow searching across different document sets
   - **Action**: Build document management API with user permissions

3. **AI Enhancements**
   - **Model Evaluation**: Benchmark different LLMs for cost/performance
   - **Custom Fine-tuning**: Train domain-specific models for improved accuracy
   - **Hybrid Search**: Implement combined keyword and semantic search
   - **Action**: Set up evaluation pipeline and fine-tune models

### Long-term Vision (6+ Months)

1. **Enterprise Integration**
   - **SSO Integration**: Support enterprise authentication systems
   - **Compliance Features**: Add audit logs and compliance reporting
   - **Data Governance**: Implement data retention and privacy controls
   - **Action**: Develop enterprise integration framework

2. **Advanced Intelligence**
   - **Autonomous Agents**: Implement agentic workflows for complex tasks
   - **Multi-modal Support**: Add support for images, audio, and video content
   - **Personalization**: User-specific response tuning and preferences
   - **Action**: Research and prototype autonomous agent framework

3. **Ecosystem Development**
   - **Plugin System**: Create extensible plugin architecture
   - **Developer SDK**: Build tools for custom agent development
   - **Marketplace**: Allow sharing of custom agents and workflows
   - **Action**: Design plugin architecture and developer documentation

## 🐛 Troubleshooting

### Common Issues

1. **Vector store not found**: Run `python scripts/ingest_pdfs.py` to ingest documents
2. **API key errors**: Ensure `OPENAI_API_KEY` is set in environment
3. **Permission errors**: Check file permissions for data directories
4. **Memory issues**: Reduce chunk size or use smaller embedding models
5. **NumPy compatibility errors**: Add `numpy<2.0.0` to requirements.txt

### Logs

Check application logs:
```bash
# Docker
docker-compose logs -f chat-with-pdf

# Local
tail -f logs/app.log
```

## 📄 License

This project is developed for the ArcFusion technical assignment.

## 🤝 Contributing

This is a technical assessment project. For questions or clarifications, please reach out to the development team.

---

**Note**: This system demonstrates production-ready architecture patterns and can be extended for real-world applications with appropriate scaling and security considerations. 