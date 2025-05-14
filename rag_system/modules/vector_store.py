"""
Module for vector database operations.
"""
import os
import json
import uuid
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
from langchain.vectorstores import FAISS, Chroma
from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document

from rag_system.config import VECTOR_DB_PATH, VECTOR_DB_TYPE
from rag_system.utils.logger import setup_logger
from rag_system.utils.exceptions import VectorStoreError, ConfigurationError


logger = setup_logger(__name__)


class VectorStore:
    """
    Class for vector database operations.
    """
    
    SUPPORTED_STORES = ["faiss", "chroma"]
    
    def __init__(self, embedding_model: Embeddings, db_type: str = VECTOR_DB_TYPE, 
                 db_path: str = VECTOR_DB_PATH):
        """
        Initialize the VectorStore.
        
        Args:
            embedding_model (Embeddings): Langchain embeddings object.
            db_type (str): Type of vector database. Options: "faiss", "chroma".
            db_path (str): Path to store the vector database.
            
        Raises:
            ConfigurationError: If db_type is not supported.
        """
        self.embedding_model = embedding_model
        self.db_type = db_type.lower()
        self.db_path = db_path
        
        if self.db_type not in self.SUPPORTED_STORES:
            raise ConfigurationError(f"Unsupported vector database type: {db_type}. "
                                    f"Supported types: {self.SUPPORTED_STORES}")
        
        # Ensure the database directory exists
        os.makedirs(self.db_path, exist_ok=True)
        
        # Initialize the vector store
        self.store = self._init_vector_store()
        logger.info(f"VectorStore initialized with type: {self.db_type} at path: {self.db_path}")
    
    def _init_vector_store(self) -> Union[FAISS, Chroma]:
        """
        Initialize the vector store based on db_type.
        
        Returns:
            Union[FAISS, Chroma]: Vector store object.
            
        Raises:
            VectorStoreError: If vector store initialization fails.
        """
        try:
            store_path = os.path.join(self.db_path, self.db_type)
            os.makedirs(store_path, exist_ok=True)
            
            if self.db_type == "faiss":
                # Check if FAISS index exists
                index_path = os.path.join(store_path, "index.faiss")
                docstore_path = os.path.join(store_path, "docstore.json")
                
                if os.path.exists(index_path) and os.path.exists(docstore_path):
                    logger.info("Loading existing FAISS index")
                    return FAISS.load_local(store_path, self.embedding_model, "index")
                else:
                    logger.info("Initializing new FAISS index")
                    return FAISS.from_texts(["Initialization text"], self.embedding_model)
            
            elif self.db_type == "chroma":
                logger.info("Initializing or loading Chroma database")
                return Chroma(
                    embedding_function=self.embedding_model,
                    persist_directory=store_path,
                )
            
            else:
                raise VectorStoreError(f"Vector store initialization not implemented for {self.db_type}")
        
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise VectorStoreError(f"Error initializing vector store: {str(e)}")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents (List[Dict[str, Any]]): List of document dictionaries with text, metadata, and embeddings.
            
        Returns:
            List[str]: List of document IDs.
            
        Raises:
            VectorStoreError: If document addition fails.
        """
        if not documents:
            logger.warning("Attempted to add empty list of documents")
            return []
        
        try:
            # Convert to Langchain Document format
            langchain_docs = []
            for doc in documents:
                langchain_doc = Document(
                    page_content=doc.get("text", ""),
                    metadata=doc.get("metadata", {})
                )
                langchain_docs.append(langchain_doc)
            
            ids = [str(uuid.uuid4()) for _ in range(len(langchain_docs))]
            
            if self.db_type == "faiss":
                self.store.add_documents(langchain_docs)
                # Save the updated index
                self.store.save_local(os.path.join(self.db_path, self.db_type), "index")
            
            elif self.db_type == "chroma":
                if hasattr(documents[0], "embedding") and documents[0]["embedding"]:
                    # If documents have embeddings, use them
                    embeddings = [doc.get("embedding", []) for doc in documents]
                    self.store.add_documents(
                        documents=langchain_docs,
                        ids=ids,
                        embeddings=embeddings
                    )
                else:
                    # Let Chroma calculate embeddings
                    self.store.add_documents(
                        documents=langchain_docs,
                        ids=ids
                    )
                self.store.persist()
            
            logger.info(f"Added {len(documents)} documents to vector store")
            return ids
        
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise VectorStoreError(f"Error adding documents to vector store: {str(e)}")
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents in the vector store.
        
        Args:
            query (str): Query text.
            k (int): Number of results to return.
            
        Returns:
            List[Dict[str, Any]]: List of similar documents with similarity scores.
            
        Raises:
            VectorStoreError: If search fails.
        """
        if not query.strip():
            logger.warning("Attempted to search with empty query")
            return []
        
        try:
            if self.db_type == "faiss":
                docs_and_scores = self.store.similarity_search_with_score(query, k=k)
                results = []
                
                for doc, score in docs_and_scores:
                    results.append({
                        "text": doc.page_content,
                        "metadata": doc.metadata,
                        "score": float(score)
                    })
                
                return results
            
            elif self.db_type == "chroma":
                docs_and_scores = self.store.similarity_search_with_relevance_scores(query, k=k)
                results = []
                
                for doc, score in docs_and_scores:
                    results.append({
                        "text": doc.page_content,
                        "metadata": doc.metadata,
                        "score": float(score)
                    })
                
                return results
            
            else:
                raise VectorStoreError(f"Search not implemented for {self.db_type}")
        
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            raise VectorStoreError(f"Error searching vector store: {str(e)}")
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """
        Delete documents from the vector store.
        
        Args:
            document_ids (List[str]): List of document IDs to delete.
            
        Returns:
            bool: True if deletion was successful.
            
        Raises:
            VectorStoreError: If deletion fails.
        """
        if not document_ids:
            logger.warning("Attempted to delete empty list of document IDs")
            return True
        
        try:
            if self.db_type == "chroma":
                # Chroma supports direct deletion by ID
                self.store.delete(document_ids)
                self.store.persist()
                logger.info(f"Deleted {len(document_ids)} documents from Chroma")
                return True
            
            elif self.db_type == "faiss":
                # FAISS does not support deletion, so we need to rebuild the index
                logger.warning("FAISS does not support direct deletion. "
                              "Consider using a different vector store for production use.")
                return False
            
            else:
                raise VectorStoreError(f"Deletion not implemented for {self.db_type}")
        
        except Exception as e:
            logger.error(f"Error deleting documents from vector store: {str(e)}")
            raise VectorStoreError(f"Error deleting documents from vector store: {str(e)}")
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        Get all documents from the vector store.
        
        Returns:
            List[Dict[str, Any]]: List of all documents.
            
        Raises:
            VectorStoreError: If retrieval fails.
        """
        try:
            if self.db_type == "chroma":
                # Chroma supports getting all documents
                result = self.store.get()
                
                if not result or "documents" not in result:
                    return []
                
                docs = []
                for i, doc in enumerate(result["documents"]):
                    docs.append({
                        "id": result["ids"][i] if "ids" in result else str(uuid.uuid4()),
                        "text": doc,
                        "metadata": result["metadatas"][i] if "metadatas" in result else {}
                    })
                
                return docs
            
            elif self.db_type == "faiss":
                # FAISS does not support getting all documents directly
                logger.warning("FAISS does not support retrieving all documents directly.")
                return []
            
            else:
                raise VectorStoreError(f"Get all documents not implemented for {self.db_type}")
        
        except Exception as e:
            logger.error(f"Error getting all documents from vector store: {str(e)}")
            raise VectorStoreError(f"Error getting all documents from vector store: {str(e)}")
    
    def clear(self) -> bool:
        """
        Clear all documents from the vector store.
        
        Returns:
            bool: True if clearing was successful.
            
        Raises:
            VectorStoreError: If clearing fails.
        """
        try:
            if self.db_type == "chroma":
                # Get all document IDs and delete them
                result = self.store.get()
                if result and "ids" in result and result["ids"]:
                    self.store.delete(result["ids"])
                    self.store.persist()
                logger.info("Cleared all documents from Chroma")
                return True
            
            elif self.db_type == "faiss":
                # For FAISS, we recreate the index
                store_path = os.path.join(self.db_path, self.db_type)
                index_path = os.path.join(store_path, "index.faiss")
                docstore_path = os.path.join(store_path, "docstore.json")
                
                if os.path.exists(index_path):
                    os.remove(index_path)
                if os.path.exists(docstore_path):
                    os.remove(docstore_path)
                
                # Reinitialize
                self.store = FAISS.from_texts(["Initialization text"], self.embedding_model)
                logger.info("Cleared FAISS index by recreating it")
                return True
            
            else:
                raise VectorStoreError(f"Clearing not implemented for {self.db_type}")
        
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            raise VectorStoreError(f"Error clearing vector store: {str(e)}")
    
    def save(self) -> bool:
        """
        Save the current state of the vector store.
        
        Returns:
            bool: True if saving was successful.
            
        Raises:
            VectorStoreError: If saving fails.
        """
        try:
            if self.db_type == "faiss":
                store_path = os.path.join(self.db_path, self.db_type)
                self.store.save_local(store_path, "index")
                logger.info(f"Saved FAISS index to {store_path}")
                return True
            
            elif self.db_type == "chroma":
                self.store.persist()
                logger.info("Persisted Chroma database")
                return True
            
            else:
                raise VectorStoreError(f"Saving not implemented for {self.db_type}")
        
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise VectorStoreError(f"Error saving vector store: {str(e)}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dict[str, Any]: Dictionary of statistics.
            
        Raises:
            VectorStoreError: If statistics retrieval fails.
        """
        try:
            stats = {
                "db_type": self.db_type,
                "db_path": self.db_path,
            }
            
            if self.db_type == "chroma":
                result = self.store.get()
                if result and "ids" in result:
                    stats["document_count"] = len(result["ids"])
                else:
                    stats["document_count"] = 0
            
            elif self.db_type == "faiss":
                # FAISS does not provide direct document count
                store_path = os.path.join(self.db_path, self.db_type)
                docstore_path = os.path.join(store_path, "docstore.json")
                
                if os.path.exists(docstore_path):
                    with open(docstore_path, 'r') as f:
                        docstore = json.load(f)
                    stats["document_count"] = len(docstore)
                else:
                    stats["document_count"] = 0
            
            logger.info(f"Vector store statistics: {stats}")
            return stats
        
        except Exception as e:
            logger.error(f"Error getting vector store statistics: {str(e)}")
            raise VectorStoreError(f"Error getting vector store statistics: {str(e)}") 