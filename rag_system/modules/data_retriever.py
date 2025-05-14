"""
Module for retrieving data based on user queries.
"""
from typing import List, Dict, Any, Optional, Union, Callable

from rag_system.modules.vector_store import VectorStore
from rag_system.modules.text_embedder import TextEmbedder
from rag_system.utils.logger import setup_logger
from rag_system.utils.exceptions import RetrievalError


logger = setup_logger(__name__)


class DataRetriever:
    """
    Class for retrieving data based on user queries.
    """
    
    def __init__(self, vector_store: VectorStore, embedder: TextEmbedder,
                 result_count: int = 5, score_threshold: float = 0.7):
        """
        Initialize the DataRetriever.
        
        Args:
            vector_store (VectorStore): Vector store for document retrieval.
            embedder (TextEmbedder): Text embedder for query embedding.
            result_count (int): Default number of results to return.
            score_threshold (float): Minimum similarity score threshold.
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.result_count = result_count
        self.score_threshold = score_threshold
        
        logger.info("DataRetriever initialized")
    
    def search(self, query: str, k: Optional[int] = None,
              threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Search for documents based on a query.
        
        Args:
            query (str): Query string.
            k (Optional[int]): Number of results to return. If None, uses default.
            threshold (Optional[float]): Minimum similarity score. If None, uses default.
            
        Returns:
            List[Dict[str, Any]]: List of relevant documents.
            
        Raises:
            RetrievalError: If search fails.
        """
        if not query.strip():
            logger.warning("Empty query provided for search")
            return []
        
        k = k or self.result_count
        threshold = threshold or self.score_threshold
        
        try:
            # Search in vector store
            results = self.vector_store.search(query, k=k)
            
            # Filter by threshold
            filtered_results = [
                result for result in results
                if result.get("score", 0) >= threshold
            ]
            
            logger.info(f"Retrieved {len(filtered_results)} results for query: {query[:50]}...")
            return filtered_results
        
        except Exception as e:
            logger.error(f"Error searching for query '{query}': {str(e)}")
            raise RetrievalError(f"Error searching for query: {str(e)}")
    
    def get_related_context(self, query: str, k: Optional[int] = None,
                           threshold: Optional[float] = None) -> str:
        """
        Get related context as concatenated text for a query.
        
        Args:
            query (str): Query string.
            k (Optional[int]): Number of results to include. If None, uses default.
            threshold (Optional[float]): Minimum similarity score. If None, uses default.
            
        Returns:
            str: Concatenated context from relevant documents.
            
        Raises:
            RetrievalError: If retrieval fails.
        """
        try:
            results = self.search(query, k=k, threshold=threshold)
            
            if not results:
                logger.warning(f"No relevant context found for query: {query[:50]}...")
                return ""
            
            # Extract and join text from results
            context_parts = []
            for i, result in enumerate(results):
                text = result.get("text", "").strip()
                
                if text:
                    source = f"[Source {i+1}"
                    if "metadata" in result and "source" in result["metadata"]:
                        source += f": {result['metadata']['source']}"
                    source += "]"
                    
                    context_parts.append(f"{source}\n{text}\n")
            
            return "\n".join(context_parts)
        
        except Exception as e:
            logger.error(f"Error getting context for query '{query}': {str(e)}")
            raise RetrievalError(f"Error getting context for query: {str(e)}")
    
    def get_query_embedding(self, query: str) -> List[float]:
        """
        Get embedding for a query.
        
        Args:
            query (str): Query string.
            
        Returns:
            List[float]: Query embedding vector.
            
        Raises:
            RetrievalError: If embedding fails.
        """
        try:
            embedding = self.embedder.embed_text(query)
            return embedding
        
        except Exception as e:
            logger.error(f"Error embedding query '{query}': {str(e)}")
            raise RetrievalError(f"Error embedding query: {str(e)}")
    
    def search_with_reranking(self, query: str, 
                             k: Optional[int] = None,
                             reranker: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """
        Search with reranking of results.
        
        Args:
            query (str): Query string.
            k (Optional[int]): Number of results to return. If None, uses default.
            reranker (Optional[Callable]): Function for reranking results.
                Should take (query, results) and return reranked results.
            
        Returns:
            List[Dict[str, Any]]: List of reranked relevant documents.
            
        Raises:
            RetrievalError: If search fails.
        """
        try:
            # Get initial results with a higher k to allow for filtering
            initial_k = (k or self.result_count) * 3
            results = self.search(query, k=initial_k, threshold=0)
            
            if not results:
                return []
            
            # Apply reranking if provided
            if reranker:
                try:
                    reranked_results = reranker(query, results)
                    logger.info(f"Reranked {len(reranked_results)} results for query: {query[:50]}...")
                    
                    # Limit to k results
                    return reranked_results[:k or self.result_count]
                
                except Exception as e:
                    logger.error(f"Error reranking results: {str(e)}")
                    # Fall back to original results if reranking fails
                    return results[:k or self.result_count]
            
            return results[:k or self.result_count]
        
        except Exception as e:
            logger.error(f"Error searching with reranking for query '{query}': {str(e)}")
            raise RetrievalError(f"Error searching with reranking: {str(e)}")
    
    def search_by_metadata(self, query: str, metadata_filter: Dict[str, Any],
                          k: Optional[int] = None,
                          threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Search with metadata filtering.
        
        Args:
            query (str): Query string.
            metadata_filter (Dict[str, Any]): Metadata for filtering results.
            k (Optional[int]): Number of results to return. If None, uses default.
            threshold (Optional[float]): Minimum similarity score. If None, uses default.
            
        Returns:
            List[Dict[str, Any]]: List of filtered relevant documents.
            
        Raises:
            RetrievalError: If search fails.
        """
        try:
            # Get initial results with a higher k to allow for filtering
            initial_k = (k or self.result_count) * 5
            results = self.search(query, k=initial_k, threshold=threshold)
            
            if not results:
                return []
            
            # Filter by metadata
            filtered_results = []
            for result in results:
                result_metadata = result.get("metadata", {})
                match = True
                
                # Check if all filter conditions match
                for key, value in metadata_filter.items():
                    if key not in result_metadata or result_metadata[key] != value:
                        match = False
                        break
                
                if match:
                    filtered_results.append(result)
            
            logger.info(f"Filtered to {len(filtered_results)} results based on metadata")
            
            # Limit to k results
            return filtered_results[:k or self.result_count]
        
        except Exception as e:
            logger.error(f"Error searching with metadata filter for query '{query}': {str(e)}")
            raise RetrievalError(f"Error searching with metadata filter: {str(e)}")
    
    def get_query_plan(self, query: str) -> Dict[str, Any]:
        """
        Create a query plan with analysis of the query.
        
        Args:
            query (str): Query string.
            
        Returns:
            Dict[str, Any]: Query plan with analysis.
            
        Raises:
            RetrievalError: If plan creation fails.
        """
        try:
            return {
                "query": query,
                "embedding_available": True,
                "default_k": self.result_count,
                "threshold": self.score_threshold,
                "metadata_filtering_available": True,
                "reranking_available": True,
            }
        
        except Exception as e:
            logger.error(f"Error creating query plan for '{query}': {str(e)}")
            raise RetrievalError(f"Error creating query plan: {str(e)}")
    
    def hybrid_search(self, query: str, k: Optional[int] = None,
                     keyword_weight: float = 0.3) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector search with keyword search.
        
        This is a placeholder for hybrid search functionality which would
        combine vector search with BM25 or similar keyword search.
        
        Args:
            query (str): Query string.
            k (Optional[int]): Number of results to return. If None, uses default.
            keyword_weight (float): Weight of keyword search vs. vector search.
            
        Returns:
            List[Dict[str, Any]]: List of relevant documents.
            
        Raises:
            RetrievalError: If search fails.
        """
        # Note: This is a placeholder. Actual implementation would require
        # a keyword search index like ElasticSearch or a BM25 implementation.
        logger.warning("Hybrid search is not fully implemented, falling back to vector search")
        return self.search(query, k=k) 