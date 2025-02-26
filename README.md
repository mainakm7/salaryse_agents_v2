# SalarySe_Agents_v2

An AI agent system built with FastAPI and open-source LLMs, supporting API calling, RAG-based answering, and general conversation capabilities. The project provides a modular and extensible architecture for various AI workflows.

If the response is returning an API, the string will be stored in a key called 'api'.

## Architecture

### Project Structure
```
root/
├── data/               # Data folder for RAG files in .csv format.
├── metadata/           # Data folder for RAG files in .csv format after metadata tagging.
├── api_agents/         # Different API worker agents. Each specializes in different products.
├── manager_agent.py    # Manager LLM for routing queries to worker agents.
├── worker_agents.py    # Different specialized worker agents.
├── api_manager.py      # A sub-manager LLM for routing API-based queries to different sub-workers specializing in different product APIs.
├── graphbuilder.py     # LangGraph-based AI workflow management.
└── ai_app.py           # FastAPI backend implementation.
```

### Components
- **Manager Agent**: Routes incoming queries to appropriate worker agents (RAG, database, chat, API supervisor).
- **Worker Agents**: Specialized agents for different tasks.
- **API Worker Agents**: Specialized agents for different product APIs.
- **LangGraph**: Manages AI workflow execution.
- **FastAPI Backend**: Provides REST API endpoints.

## Setup

### Prerequisites

1. **API Keys**
   - Nomic API Key (for embeddings)
   - Groq API Key (optional, for Groq deployment)

2. **AWS Configuration**
   - AWS credentials for Amazon Bedrock
   - DocumentDB configuration for vector storage

### Environment Variables

Create a `.env` file in the root directory:
```bash
NOMIC_API_KEY=your_nomic_api_key
DOCUMENTDB_URI=your_documentdb_uri
DOCUMENTDB_USERNAME=your_documentdb_username
DOCUMENTDB_CLUSTER_NAME=your_documentdb_cluster_name
AWS_REGION=aws_region_of_your_documentdb_cluster
GROQ_API_KEY=your_groq_api_key  # Optional
```

### Installation

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Key dependencies:
   - FastAPI
   - Langchain
   - LangGraph
   - Langchain-aws
   - python-dotenv

### Model Setup

#### Local Models (Ollama)
1. Install Ollama.
2. Pull required models:
   ```bash
   ollama pull gemma2:27b
   # or
   ollama pull llama2:70b
   ```

#### AWS Bedrock Models
Update the AWS profile in your code:
```python
llm = ChatBedrock(credentials_profile_name="your_profile_name")
```

## Usage

### Data

The RAG data files should be stored in a subdirectory called `data` in `.csv` format for proper loading.

### Vector Store Setup

Choose and initialize either DocumentDB or ChromaDB:

```bash
# For DocumentDB
python rag_retriever_documentdb.py

# For ChromaDB (local storage)
python rag_retriever_chroma.py
```

### Starting the Server

Launch the FastAPI server:
```bash
uvicorn ai_app:app --reload
```

Access the API documentation at `http://localhost:8000/docs`

### API Usage

Send queries as JSON to the endpoint:
```json
{
    "query": "Your query here",
    "thread_id": "user_thread_id"  // Defaults to 1 if not provided
}
```

## Vector Store Options

### DocumentDB (Default)
- Used as the primary vector store.
- Requires AWS configuration.
- Suitable for production deployments.

### ChromaDB (Alternative)
- Local storage option.
- Good for development and testing.
- No cloud dependencies.

## Deployment Options

### Local Development
- Follow the setup instructions above.
- Suitable for testing and development.

### Groq Deployment
1. Add Groq API key to `.env`.
2. Configure hardware-specific settings in `ai_app.py`.
3. Deploy using standard Groq deployment procedures.

## Error Handling

The system includes robust error handling for:
- Invalid API keys.
- Model loading failures.
- Vector store connection issues.
- Query processing errors.

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request.

## License

This project is licensed under the Apache License.

## Future Roadmap

- Multi-modal support.
- Additional LLM integrations.
- Enhanced RAG capabilities.
- Performance optimizations.
- Frontend development.

