"""
Custom exceptions for the RAG system.
"""


class RAGBaseException(Exception):
    """Base exception for all RAG system exceptions."""
    pass


class PDFExtractionError(RAGBaseException):
    """Exception raised when PDF text extraction fails."""
    pass


class EmbeddingError(RAGBaseException):
    """Exception raised when text embedding fails."""
    pass


class VectorStoreError(RAGBaseException):
    """Exception raised when vector store operations fail."""
    pass


class QualityCheckError(RAGBaseException):
    """Exception raised when quality checks fail."""
    pass


class RetrievalError(RAGBaseException):
    """Exception raised when data retrieval fails."""
    pass


class ConfigurationError(RAGBaseException):
    """Exception raised when configuration is invalid."""
    pass 