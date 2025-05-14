"""
Module for embedding text using Langchain with various models.
"""
import time
from typing import List, Dict, Any, Union, Optional

from langchain.embeddings.base import Embeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from rag_system.config import OPENAI_API_KEY, EMBEDDING_MODEL
from rag_system.utils.logger import setup_logger
from rag_system.utils.exceptions import EmbeddingError, ConfigurationError


logger = setup_logger(__name__)


class TextEmbedder:
    """
    Class for embedding text using various models through Langchain.
    """
    
    SUPPORTED_MODELS = {
        "openai": {
            "class": OpenAIEmbeddings,
            "requires_api_key": True,
            "default_model": "text-embedding-ada-002",
        },
        "huggingface": {
            "class": HuggingFaceEmbeddings,
            "requires_api_key": False,
            "default_model": "sentence-transformers/all-mpnet-base-v2",
        },
        "bge": {
            "class": HuggingFaceBgeEmbeddings,
            "requires_api_key": False,
            "default_model": "BAAI/bge-small-en-v1.5",
        },
        "korean": {
            "class": HuggingFaceEmbeddings,
            "requires_api_key": False,
            "default_model": "nlpai-lab/KURE-v1",
        }
    }
    
    def __init__(self, model_type: str = "openai", model_name: Optional[str] = None, 
                 api_key: Optional[str] = None, batch_size: int = 16):
        """
        Initialize the TextEmbedder.
        
        Args:
            model_type (str): Type of embedding model to use. Options: "openai", "huggingface", "bge", "korean".
            model_name (Optional[str]): Name of specific model to use. If None, uses default for model_type.
            api_key (Optional[str]): API key for the model service. If None, uses default from config.
            batch_size (int): Batch size for embedding generation.
            
        Raises:
            ConfigurationError: If model_type is not supported or required API key is missing.
        """
        self.model_type = model_type.lower()
        self.batch_size = batch_size
        
        if self.model_type not in self.SUPPORTED_MODELS:
            raise ConfigurationError(f"Unsupported model type: {model_type}. "
                                    f"Supported types: {list(self.SUPPORTED_MODELS.keys())}")
        
        model_info = self.SUPPORTED_MODELS[self.model_type]
        self.model_name = model_name or model_info["default_model"]
        
        # Check if API key is required and available
        if model_info["requires_api_key"]:
            self.api_key = api_key or OPENAI_API_KEY
            if not self.api_key:
                raise ConfigurationError(f"{self.model_type} requires an API key, but none was provided")
        else:
            self.api_key = None
        
        # Initialize the embedding model
        self.embedding_model = self._init_embedding_model()
        logger.info(f"TextEmbedder initialized with model: {self.model_type} ({self.model_name})")
    
    def _init_embedding_model(self) -> Embeddings:
        """
        Initialize the embedding model based on model_type.
        
        Returns:
            Embeddings: Langchain embeddings object.
            
        Raises:
            ConfigurationError: If model initialization fails.
        """
        model_info = self.SUPPORTED_MODELS[self.model_type]
        model_class = model_info["class"]
        
        try:
            if self.model_type == "openai":
                return model_class(
                    model=self.model_name,
                    openai_api_key=self.api_key,
                )
            elif self.model_type in ["huggingface", "korean"]:
                return model_class(
                    model_name=self.model_name,
                )
            elif self.model_type == "bge":
                return model_class(
                    model_name=self.model_name,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True}
                )
            else:
                raise ConfigurationError(f"Embedding model initialization not implemented for {self.model_type}")
        
        except Exception as e:
            logger.error(f"Error initializing embedding model: {str(e)}")
            raise ConfigurationError(f"Error initializing embedding model: {str(e)}")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embeddings for a single text.
        
        Args:
            text (str): Text to embed.
            
        Returns:
            List[float]: Embedding vector.
            
        Raises:
            EmbeddingError: If embedding generation fails.
        """
        if not text.strip():
            logger.warning("Attempted to embed empty text")
            return []
        
        try:
            embedding = self.embedding_model.embed_query(text)
            return embedding
        
        except Exception as e:
            logger.error(f"Error embedding text: {str(e)}")
            raise EmbeddingError(f"Error embedding text: {str(e)}")
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts (List[str]): List of texts to embed.
            
        Returns:
            List[List[float]]: List of embedding vectors.
            
        Raises:
            EmbeddingError: If embedding generation fails.
        """
        if not texts:
            logger.warning("Attempted to embed empty list of texts")
            return []
        
        # Filter out empty texts
        texts = [text for text in texts if text.strip()]
        if not texts:
            logger.warning("All texts were empty after filtering")
            return []
        
        try:
            # Process in batches to avoid overloading the API
            all_embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                logger.info(f"Embedding batch {i//self.batch_size + 1}/{(len(texts)-1)//self.batch_size + 1} "
                           f"({len(batch)} texts)")
                
                batch_start = time.time()
                batch_embeddings = self.embedding_model.embed_documents(batch)
                batch_duration = time.time() - batch_start
                
                all_embeddings.extend(batch_embeddings)
                logger.info(f"Batch embedded in {batch_duration:.2f} seconds")
                
                # Add a small delay to avoid hitting rate limits
                if i + self.batch_size < len(texts) and self.model_type == "openai":
                    time.sleep(0.5)
            
            return all_embeddings
        
        except Exception as e:
            logger.error(f"Error embedding texts: {str(e)}")
            raise EmbeddingError(f"Error embedding texts: {str(e)}")
    
    def embed_documents(self, 
                        documents: List[Dict[str, Any]],
                        text_key: str = "text") -> List[Dict[str, Any]]:
        """
        Generate embeddings for documents and add them to the documents.
        
        Args:
            documents (List[Dict[str, Any]]): List of document dictionaries.
            text_key (str): Key for the text field in the documents.
            
        Returns:
            List[Dict[str, Any]]: Documents with added embeddings.
            
        Raises:
            EmbeddingError: If embedding generation fails.
        """
        if not documents:
            logger.warning("Attempted to embed empty list of documents")
            return []
        
        texts = [doc.get(text_key, "") for doc in documents]
        valid_indices = [i for i, text in enumerate(texts) if text.strip()]
        valid_texts = [texts[i] for i in valid_indices]
        
        try:
            embeddings = self.embed_texts(valid_texts)
            
            # Add embeddings to documents
            result = []
            embedding_idx = 0
            for i, doc in enumerate(documents):
                if i in valid_indices:
                    doc_copy = doc.copy()
                    doc_copy["embedding"] = embeddings[embedding_idx]
                    embedding_idx += 1
                    result.append(doc_copy)
                else:
                    logger.warning(f"Skipping document {i} due to empty text")
            
            return result
        
        except Exception as e:
            logger.error(f"Error embedding documents: {str(e)}")
            raise EmbeddingError(f"Error embedding documents: {str(e)}")
    
    def get_document_embeddings(self, chunks: List[str], 
                               metadata: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Generate embeddings for text chunks and combine with metadata.
        
        Args:
            chunks (List[str]): List of text chunks to embed.
            metadata (List[Dict[str, Any]], optional): Metadata for each chunk.
            
        Returns:
            List[Dict[str, Any]]: List of documents with text, metadata, and embeddings.
            
        Raises:
            EmbeddingError: If embedding generation fails.
        """
        if not chunks:
            logger.warning("Attempted to embed empty list of chunks")
            return []
        
        # Create documents with text and metadata
        documents = []
        for i, chunk in enumerate(chunks):
            doc = {"text": chunk}
            if metadata and i < len(metadata):
                doc["metadata"] = metadata[i]
            elif metadata:
                doc["metadata"] = metadata[-1]  # Use last metadata if index out of range
            else:
                doc["metadata"] = {"index": i}
            documents.append(doc)
        
        # Generate embeddings
        return self.embed_documents(documents) 