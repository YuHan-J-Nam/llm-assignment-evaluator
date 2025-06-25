# LLM API Project Structure

## Overview
This project is a comprehensive LLM (Large Language Model) API system for automated essay evaluation and assessment. It supports multiple AI providers (OpenAI, Anthropic Claude, Google Gemini) and includes RAG (Retrieval-Augmented Generation) capabilities for enhanced evaluation.

## Root Directory Structure

```
LLM_API/
├── .env                              # Environment variables and API keys
├── .git/                             # Git repository data
├── .gitignore                        # Git ignore file
├── LICENSE                           # Project license
├── README.md                         # Project documentation
├── requirements.txt                  # Python dependencies
├── TASK_LOG.md                       # Development task log
├── PROJECT_STRUCTURE.md              # This file
├── 
├── # Core Application Files
├── llm_api_client.py                 # Main LLM API client
├── llm_api_client.ipynb             # Jupyter notebook for API testing
├── create_checklist.py              # Checklist creation functionality
├── evaluate_assignment.py           # Assignment evaluation logic
├── widgets_core.py                  # Core widget functionality
├── UI.ipynb                         # User interface notebook
├── 
├── # Sample Data
├── student_submission_sample.txt    # Sample student submission for testing
├── essay_evaluation_scores.csv     # Evaluation scores dataset
├── 
├── # Core Modules
├── api_utils/                       # API utility modules
├── rag_utils/                       # RAG (Retrieval-Augmented Generation) utilities
├── 
├── # Output Directories
├── auto_evaluation_results/         # Automated evaluation results
├── checklists/                      # Generated evaluation checklists
├── evaluations/                     # Individual evaluation results
├── logs/                           # Application logs
├── temp/                           # Temporary files
├── test/                           # Test files
├── __pycache__/                    # Python cache files
├── 
└── summarize_submission 관련/       # Submission summarization related files
```

## Detailed Module Structure

### 1. `api_utils/` - API Integration Module
Contains utilities for interacting with different LLM providers:

```
api_utils/
├── __init__.py
├── config.py                    # Configuration management
├── anthropic_api.py            # Anthropic Claude API integration
├── gemini_api.py               # Google Gemini API integration
├── openai_api.py               # OpenAI API integration
├── logging_utils.py            # Logging utilities
├── pdf_utils.py                # PDF processing utilities
├── schema_manager.py           # Schema management for API responses
└── token_counter.py            # Token counting utilities
```

### 2. `rag_utils/` - RAG Implementation Module
Implements Retrieval-Augmented Generation capabilities:

```
rag_utils/
├── __init__.py
├── get_embedding_function.py   # Embedding function using Korean model (KURE-v1)
├── populate_database.py       # Vector database population script
├── query_data.py              # Data querying functionality
├── cache_folder/              # Embedding cache storage
├── chroma/                    # ChromaDB vector database
├── data/                      # Source documents for RAG
└── __pycache__/              # Python cache files
```

### 3. `auto_evaluation_results/` - Evaluation Results Storage
Stores automated evaluation results from different models:

```
auto_evaluation_results/
├── auto_evaluation_results.json                    # Main results file
├── auto_evaluation_results_light_models.json       # Light models results
├── auto_evaluation_results_light_models1.json      # Additional light models results
├── auto_evaluation_results_medium_models.json      # Medium models results
├── ESSAY_74077_responses.txt                       # Individual essay responses
├── ESSAY_78504_responses.txt
├── ESSAY_83769_responses.txt
├── processed_results_json.zip                      # Compressed results
├── evaluation_results_json/                        # JSON format results
├── previous_attempt/                               # Previous evaluation attempts
└── processed_results_json/                        # Processed results
```

### 4. `checklists/` - Evaluation Checklists
Contains generated evaluation checklists for different topics and models:

```
checklists/
├── claude_평가기준_비혼주의자에 대한 본인의 의견_*.json      # Claude checklists
├── gemini_평가기준_비혼주의자에 대한 본인의 의견_*.json      # Gemini checklists  
├── openai_평가기준_비혼주의자에 대한 본인의 의견_*.json      # OpenAI checklists
├── claude-3-7-sonnet-20250219_평가기준_*.json              # Specific model checklists
├── gemini-2.0-flash_평가기준_*.json
├── gpt-4.1_평가기준_*.json
├── 논술형_checklist.json                                   # Essay-type checklist
└── 수필형_checklist.json                                   # Narrative-type checklist
```

### 5. `evaluations/` - Individual Evaluation Results
Stores detailed evaluation results for individual submissions:

```
evaluations/
├── Claude_평가결과_*.json          # Claude evaluation results
├── Gemini_평가결과_*.json          # Gemini evaluation results
├── OpenAI_평가결과_*.json          # OpenAI evaluation results
├── claude-3-opus-20240229_*.json   # Specific Claude model results
├── gemini-2.0-flash_*.json         # Specific Gemini model results
├── o4-mini_*.json                  # OpenAI o4-mini model results
└── test.json                       # Test evaluation file
```

## Key Features

### Multi-Provider LLM Support
- **OpenAI**: GPT-4, GPT-4-mini, o4-mini models
- **Anthropic**: Claude-3-Opus, Claude-3-Sonnet models  
- **Google**: Gemini-2.0-Flash, Gemini-2.5-Flash models

### RAG Implementation
- **Vector Database**: ChromaDB for document storage and retrieval
- **Embeddings**: Korean language model (KURE-v1) for semantic understanding
- **Document Processing**: PDF loading and text chunking capabilities

### Evaluation System
- **Automated Scoring**: Multi-criteria evaluation system
- **Checklist Generation**: Dynamic creation of evaluation criteria
- **Batch Processing**: Support for multiple submission evaluation
- **Results Export**: JSON and CSV format results

### User Interface
- **Jupyter Notebooks**: Interactive development and testing environment
- **Widget System**: Custom UI components for evaluation workflow
- **Logging**: Comprehensive logging system for debugging and monitoring

## Technology Stack

### Core Dependencies
- **LangChain**: Framework for LLM application development
- **ChromaDB**: Vector database for RAG implementation
- **Sentence Transformers**: For Korean text embeddings (KURE-v1)
- **Pandas**: Data manipulation and analysis
- **Jupyter**: Interactive development environment

### API Integrations
- **OpenAI API**: GPT model access
- **Anthropic API**: Claude model access  
- **Google Gemini API**: Gemini model access

## Usage Patterns

### 1. Evaluation Workflow
1. Load student submissions
2. Generate evaluation checklists using `create_checklist.py`
3. Run evaluations using `evaluate_assignment.py`
4. Results stored in `evaluations/` and `auto_evaluation_results/`

### 2. RAG Setup
1. Place source documents in `rag_utils/data/`
2. Run `populate_database.py` to create vector embeddings
3. Use `query_data.py` for document retrieval

### 3. API Testing
1. Use `llm_api_client.ipynb` for interactive testing
2. Configure API keys in `.env` file
3. Test different models and parameters

## Configuration

### Environment Variables (.env)
```
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GEMINI_API_KEY=your_gemini_key
```

### Model Configuration
- Model selection and parameters in `api_utils/config.py`
- Embedding model configuration in `rag_utils/get_embedding_function.py`
- Evaluation criteria in checklist JSON files

## Development Notes

- **Korean Language Support**: Optimized for Korean text processing and evaluation
- **Modular Design**: Separated concerns for different AI providers and functionalities
- **Extensible Architecture**: Easy to add new LLM providers or evaluation criteria
- **Comprehensive Logging**: Detailed logs for debugging and performance monitoring
