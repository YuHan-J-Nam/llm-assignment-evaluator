# RAG System

A production-ready modular Retrieval-Augmented Generation (RAG) system for processing PDF documents, embedding text, and implementing semantic search with quality checks.

## Features

- **PDF Text Extraction**: Extract text from PDF files with multiple fallback methods
- **Text Embedding**: Generate embeddings using OpenAI, Hugging Face, BGE models, or Korean-specific KURE model
- **Vector Database**: Store and query embeddings using Chroma or FAISS
- **Quality Checking**: Evaluate embedding quality and retrieval relevance
- **Data Retrieval**: Search for relevant documents based on queries

## System Architecture

The system is organized into modular components:

- `pdf_extractor.py`: Handles PDF text extraction and chunking
- `text_embedder.py`: Generates embeddings using various models
- `vector_store.py`: Manages vector database operations
- `quality_checker.py`: Performs quality checks on embeddings and results
- `data_retriever.py`: Retrieves relevant documents based on queries
- `main.py`: Entry point that ties all components together

## Installation

1. Clone the repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. If you want to use quality checking features with specific LLM providers, uncomment and install the relevant optional dependencies in `requirements.txt`:

```
# For OpenAI quality checking
pip install langchain_openai

# For Google Generative AI quality checking
pip install google-generativeai langchain_google_genai

# For Anthropic (Claude) quality checking
pip install anthropic langchain_anthropic
```

4. Set up API keys in a `.env` file:

```
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

## Usage

### Process PDF Files

Place your PDF files in the `pdf_data` folder, then run:

```bash
python -m rag_system.main --process
```

This will extract text from all PDFs, generate embeddings, and store them in the vector database.

### Search Documents

Search for documents related to a query:

```bash
python -m rag_system.main --query "Your search query here"
```

### Advanced Options

```bash
python -m rag_system.main --help
```

This will show all available command-line options:

- `--pdf_folder`: Path to the folder containing PDF files
- `--embedding_model`: Type of embedding model to use (openai, huggingface, bge, korean)
- `--vector_db`: Type of vector database (chroma, faiss)
- `--process`: Process PDF files and update vector store
- `--clear`: Clear the vector store
- `--query`: Search query
- `--k`: Number of results to return

## Using as a Library

You can also use the RAG system as a library in your own code:

```python
from rag_system.main import RAGSystem

# Initialize the system
rag = RAGSystem(
    pdf_folder="path/to/pdfs",
    embedding_model="openai",  # or "huggingface", "bge", "korean"
    vector_db="chroma"  # or "faiss"
)

# For Korean text specifically
rag = RAGSystem(
    pdf_folder="path/to/pdfs",
    embedding_model="korean",  # Uses nlpai-lab/KURE-v1 model
    vector_db="chroma"
)

# Process PDF files
num_docs = rag.process_pdf_files()

# Search for documents
results = rag.search("Your query here", k=5)

# Get context for a query
context = rag.get_context("Your query here")

# Validate retrieval quality
validation = rag.validate_retrieval(["Query 1", "Query 2"])
```

## Configuration

Edit `config.py` to adjust system settings:

- API keys
- Embedding model parameters
- Vector database configuration
- PDF processing settings
- Quality check thresholds

## Korean Language Support

The system includes dedicated support for Korean language through the "korean" embedding model option, which uses `nlpai-lab/KURE-v1`, a model specialized for Korean text embeddings.

To use it:
```bash
python -m rag_system.main --embedding_model korean --process
```

## Modular Dependencies

The system is designed with modular dependencies:
- **Core functionality** (PDF extraction, embedding, vector search) has required dependencies
- **Quality checking features** have optional dependencies that are only loaded when needed

This design allows you to use the system without installing all dependencies if you don't need certain features.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 