# PDF QA with RAG & Self-Reflection

This project lets you upload PDF documents and ask questions about their content using AI. It's like having a conversation with your documents 😄

Built as a self-learning project, it uses Retrieval-Augmented Generation (RAG) to find relevant information from your PDFs and generate accurate answers. The simple web interface allows to upload files and start chatting right away.

Under the hood, it includes some features like self-reflection (the AI double-checks its own answers) and different ways to find the best information in your documents.

## Features

### Document Processing

- **Automatic Chunking**: Upload PDFs that are automatically split into manageable chunks with overlap
- **Smart Embeddings**: Utilizes OpenAI's embeddings for semantic understanding
- **Source Tracking**: Preserves document metadata for accurate source attribution

### Retrieval Strategies

- **Stuff** - Fast and simple: concatenates all relevant chunks into a single context
- **Map Reduce** - Processes chunks separately then combines results for comprehensive answers
- **Refine** - Iteratively improves answers by refining with each relevant chunk

### AI-Powered Features

- **Contextual Compression**: Optional LLM-based compression to focus on most relevant content
- **Self-Reflection**: 
  - Automatically evaluates answer quality
  - Regenerates responses that don't meet quality thresholds
  - Provides transparent reasoning for evaluations
- **Conversational Memory**: Maintains context across questions for natural follow-ups

### Analytics & Integration

- **Langfuse & LangSmith Integration**: Detailed monitoring and analytics of LLM usage
- **Web Search**: Augments answers with real-time information when needed
- **Performance Tracking**: Monitors response quality and retrieval effectiveness
- **Usage Analytics**: Tracks user interactions and system performance

## Tech Stack

- **Backend**: Python 3.9+
- **LLM**: OpenAI GPT-3.5-turbo
- **Document Processing**: PyMuPDF, LangChain
- **Vector Database**: Chroma DB
- **Analytics**: Langfuse, LangSmith
- **Search**: Tavily Search API
- **Web Framework**: Streamlit
- **Containerization**: Docker

## Project Structure

```
.
├── app/                     # Main application directory
│   ├── chroma_db/           # Chroma vector database storage directory (will be created on startup)
│   ├── pdfs/                # PDF document storage directory (uploaded PDFs will be stored here)
│   ├── .env                 # Environment variables (must be copied from .env.example and filled in)
│   ├── .env.example         # Example environment variables
│   ├── analytics.py         # Analytics and metrics dashboard
│   ├── app.py               # Main Streamlit application
│   ├── config.py            # Configuration settings
│   ├── eval_results.csv     # Evaluation results storage file (will be created after receiving responses)
│   ├── evaluation.py        # Response evaluation logic
│   ├── langfuse_utils.py    # Langfuse tracing functions
│   ├── llm_eval_metrics.csv # LLM evaluation metrics storage file (will be created after receiving responses)
│   ├── rag_pipeline.py      # Core RAG implementation
│   ├── rag_tools.py         # RAG-specific utility functions
│   ├── requirements.txt     # Python dependencies
│   ├── self_reflection.py   # Self-reflection implementation
│   └── web_tools.py         # Web search tools integration
├── docker-compose.yml       # Docker Compose configuration
├── Dockerfile               # Docker configuration for the application
├── README.md                # Project documentation
```

## Getting Started

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/)
- OpenAI API key (LLM usage)
- Langfuse credentials (analytics tool)
- LangChain API key (another analytics tool)
- Tavily API key (web search functionality)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/igorsuhinin/rag-pdf-qa.git
   cd rag-pdf-qa
   ```

2. Create an `.env` file in the `app` directory with your API keys:
   ```bash
   cd app
   cp .env.example .env
   ```
   Then edit the `.env` file and fill in your API keys:
   ```
   OPENAI_API_KEY=sk-proj-...
   LANGCHAIN_API_KEY=lsv2_pt_...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   TAVILY_API_KEY=tvly-dev-...
   ```

### Running with Docker Compose

The application is configured to run with Docker Compose, which will set up all necessary services.

1. Build and start the containers:
   ```bash
   docker-compose up --build
   ```

2. The application will be available at: [http://localhost:8501](http://localhost:8501)

3. To stop the application, press `Ctrl+C` in the terminal or run:
   ```bash
   docker-compose down
   ```

### Local Development (Without Docker)

If you prefer to run the application locally:

1. Ensure you have Python 3.9+ installed
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r app/requirements.txt
   ```
4. Run the application:
   ```bash
   cd app
   streamlit run app.py
   ```
5. Access the application at: [http://localhost:8501](http://localhost:8501)

## License

This project is licensed under the [MIT License](LICENSE).
