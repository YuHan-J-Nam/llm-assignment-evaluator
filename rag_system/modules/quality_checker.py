"""
Module for quality checking of embeddings and retrieval results.
"""
import random
import time
import importlib
from typing import List, Dict, Any, Union, Optional, Tuple

import numpy as np

from rag_system.config import (
    OPENAI_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY,
    QUALITY_CHECK_SAMPLES, SIMILARITY_THRESHOLD
)
from rag_system.utils.logger import setup_logger
from rag_system.utils.exceptions import QualityCheckError, ConfigurationError


logger = setup_logger(__name__)


class QualityChecker:
    """
    Class for checking the quality of embeddings and retrieval results.
    """
    
    LLM_PROVIDERS = {
        "openai": {
            "module": "langchain_openai",
            "class_name": "ChatOpenAI",
            "api_key_var": OPENAI_API_KEY,
            "default_model": "gpt-3.5-turbo",
        },
        "google": {
            "module": "langchain_google_genai",
            "class_name": "ChatGoogleGenerativeAI",
            "api_key_var": GOOGLE_API_KEY,
            "default_model": "gemini-pro",
        },
        "anthropic": {
            "module": "langchain_anthropic",
            "class_name": "ChatAnthropic",
            "api_key_var": ANTHROPIC_API_KEY,
            "default_model": "claude-3-haiku-20240307",
        }
    }
    
    def __init__(self, similarity_threshold: float = SIMILARITY_THRESHOLD, 
                 sample_size: int = QUALITY_CHECK_SAMPLES,
                 llm_provider: str = "openai",
                 llm_model: Optional[str] = None):
        """
        Initialize the QualityChecker.
        
        Args:
            similarity_threshold (float): Threshold for similarity checks.
            sample_size (int): Number of samples to check.
            llm_provider (str): LLM provider for semantic quality checks.
            llm_model (Optional[str]): Specific LLM model to use.
            
        Raises:
            ConfigurationError: If LLM provider is not supported or API key is missing.
        """
        self.similarity_threshold = similarity_threshold
        self.sample_size = sample_size
        self.llm_provider = llm_provider.lower()
        
        if self.llm_provider not in self.LLM_PROVIDERS:
            raise ConfigurationError(f"Unsupported LLM provider: {llm_provider}. "
                                    f"Supported providers: {list(self.LLM_PROVIDERS.keys())}")
        
        provider_info = self.LLM_PROVIDERS[self.llm_provider]
        self.llm_model = llm_model or provider_info["default_model"]
        
        # Check if API key is available
        self.api_key = provider_info["api_key_var"]
        if not self.api_key:
            raise ConfigurationError(f"{self.llm_provider} requires an API key, but none was provided")
        
        # Initialize the LLM (lazy loading)
        self.llm = None
        logger.info(f"QualityChecker initialized with {self.llm_provider} ({self.llm_model})")
    
    def _init_llm(self):
        """
        Initialize the LLM based on provider (lazy loading).
        
        Returns:
            Any: LLM object.
            
        Raises:
            ConfigurationError: If LLM initialization fails.
        """
        if self.llm is not None:
            return self.llm
            
        provider_info = self.LLM_PROVIDERS[self.llm_provider]
        
        try:
            # Dynamically import the module
            module_name = provider_info["module"]
            class_name = provider_info["class_name"]
            
            try:
                module = importlib.import_module(module_name)
                llm_class = getattr(module, class_name)
            except (ImportError, AttributeError) as e:
                raise ConfigurationError(
                    f"Could not import {class_name} from {module_name}. "
                    f"Please install the package with: pip install {module_name}. Error: {str(e)}"
                )
            
            if self.llm_provider == "openai":
                self.llm = llm_class(
                    model=self.llm_model,
                    openai_api_key=self.api_key,
                    temperature=0
                )
            
            elif self.llm_provider == "google":
                self.llm = llm_class(
                    model=self.llm_model,
                    google_api_key=self.api_key,
                    temperature=0
                )
            
            elif self.llm_provider == "anthropic":
                self.llm = llm_class(
                    model=self.llm_model,
                    anthropic_api_key=self.api_key,
                    temperature=0
                )
            
            else:
                raise ConfigurationError(f"LLM initialization not implemented for {self.llm_provider}")
                
            return self.llm
        
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            raise ConfigurationError(f"Error initializing LLM: {str(e)}")
    
    @staticmethod
    def cosine_similarity(vector1: List[float], vector2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vector1 (List[float]): First vector.
            vector2 (List[float]): Second vector.
            
        Returns:
            float: Cosine similarity value between -1 and 1.
        """
        if not vector1 or not vector2:
            return 0.0
        
        # Convert to numpy arrays for efficient calculation
        v1 = np.array(vector1)
        v2 = np.array(vector2)
        
        # Calculate cosine similarity
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        
        return dot_product / (norm_v1 * norm_v2)
    
    def check_embedding_consistency(self, embeddings: List[List[float]]) -> Dict[str, Any]:
        """
        Check the consistency of embeddings.
        
        Args:
            embeddings (List[List[float]]): List of embedding vectors.
            
        Returns:
            Dict[str, Any]: Results of consistency checks.
            
        Raises:
            QualityCheckError: If consistency check fails.
        """
        if not embeddings:
            logger.warning("Empty embeddings list provided for consistency check")
            return {"passed": False, "reason": "Empty embeddings list"}
        
        # Check embedding dimensions
        dims = [len(emb) for emb in embeddings]
        consistent_dims = all(d == dims[0] for d in dims)
        
        if not consistent_dims:
            return {
                "passed": False,
                "reason": f"Inconsistent embedding dimensions: {dims}"
            }
        
        # Check for zero vectors
        zero_vectors = [i for i, emb in enumerate(embeddings) if np.linalg.norm(np.array(emb)) < 1e-6]
        
        if zero_vectors:
            return {
                "passed": False,
                "reason": f"Zero vectors found at indices: {zero_vectors}"
            }
        
        # Check for duplicate vectors
        duplicates = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity = self.cosine_similarity(embeddings[i], embeddings[j])
                if abs(similarity - 1.0) < 1e-6:
                    duplicates.append((i, j, similarity))
        
        if duplicates:
            return {
                "passed": False,
                "reason": f"Duplicate vectors found: {duplicates[:5]}..."
            }
        
        return {
            "passed": True,
            "dimension": dims[0],
            "vector_count": len(embeddings)
        }
    
    def check_query_result_relevance(self, query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check the relevance of query results using an LLM.
        
        Args:
            query (str): Query string.
            results (List[Dict[str, Any]]): Search results.
            
        Returns:
            Dict[str, Any]: Results of relevance checks.
            
        Raises:
            QualityCheckError: If relevance check fails.
        """
        if not results:
            logger.warning("Empty results list provided for relevance check")
            return {"passed": False, "reason": "No results to evaluate"}
        
        try:
            # Initialize LLM if not already initialized
            llm = self._init_llm()
            
            # Sample a subset of results if there are too many
            sample_size = min(self.sample_size, len(results))
            sampled_results = random.sample(results, sample_size)
            
            relevance_scores = []
            relevant_count = 0
            
            for i, result in enumerate(sampled_results):
                text = result.get("text", "")
                
                if not text.strip():
                    relevance_scores.append(0.0)
                    continue
                
                # Use LLM to evaluate relevance
                prompt = f"""
                Task: Evaluate if the following text is relevant to the query.
                
                Query: {query}
                
                Text: {text[:1000]}  # Truncate to avoid token limits
                
                Is this text relevant to the query? Respond with ONLY a number between 0 and 1,
                where 0 means completely irrelevant and 1 means highly relevant.
                """
                
                response = llm.invoke(prompt).content.strip()
                
                try:
                    relevance = float(response)
                    # Clamp to [0, 1]
                    relevance = max(0.0, min(1.0, relevance))
                except ValueError:
                    logger.warning(f"LLM returned non-numeric response: {response}")
                    relevance = 0.0
                
                relevance_scores.append(relevance)
                
                if relevance >= self.similarity_threshold:
                    relevant_count += 1
                
                # Add a small delay to avoid rate limits
                if i < len(sampled_results) - 1:
                    time.sleep(0.5)
            
            avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
            relevance_ratio = relevant_count / sample_size if sample_size > 0 else 0.0
            
            passed = relevance_ratio >= 0.7  # At least 70% of results should be relevant
            
            return {
                "passed": passed,
                "average_relevance": avg_relevance,
                "relevant_ratio": relevance_ratio,
                "relevant_count": relevant_count,
                "sample_size": sample_size,
                "relevance_scores": relevance_scores
            }
        
        except Exception as e:
            logger.error(f"Error checking query result relevance: {str(e)}")
            raise QualityCheckError(f"Error checking query result relevance: {str(e)}")
    
    def evaluate_embedding_quality(self, 
                                  texts: List[str], 
                                  embeddings: List[List[float]]) -> Dict[str, Any]:
        """
        Evaluate the overall quality of text embeddings.
        
        Args:
            texts (List[str]): List of text strings.
            embeddings (List[List[float]]): Corresponding embedding vectors.
            
        Returns:
            Dict[str, Any]: Overall quality assessment.
            
        Raises:
            QualityCheckError: If evaluation fails.
        """
        if not texts or not embeddings:
            logger.warning("Empty texts or embeddings list provided for quality evaluation")
            return {"passed": False, "reason": "Empty texts or embeddings list"}
        
        if len(texts) != len(embeddings):
            return {
                "passed": False,
                "reason": f"Number of texts ({len(texts)}) and embeddings ({len(embeddings)}) do not match"
            }
        
        # Check embedding consistency
        consistency_result = self.check_embedding_consistency(embeddings)
        if not consistency_result["passed"]:
            return consistency_result
        
        # Check semantic similarity of similar texts
        semantic_checks = []
        
        try:
            # Initialize LLM if not already initialized
            llm = self._init_llm()
            
            # Sample pairs for semantic checks
            sample_size = min(self.sample_size, len(texts))
            indices = random.sample(range(len(texts)), sample_size)
            
            for i in indices:
                # Create a semantically similar text using the LLM
                prompt = f"""
                Task: Rephrase the following text while keeping the same meaning.
                
                Original text: {texts[i][:500]}  # Truncate to avoid token limits
                
                Rephrased text (keep the core meaning but use different words):
                """
                
                rephrased_text = llm.invoke(prompt).content.strip()
                
                # Get embedding for the rephrased text (this would require passing the embedder)
                # For demonstration, we'll check if similar texts have similar embeddings
                # by comparing within our existing dataset
                
                # Find semantically similar texts in our dataset
                similar_texts = []
                for j, text in enumerate(texts):
                    if i != j:
                        similarity = self.cosine_similarity(embeddings[i], embeddings[j])
                        similar_texts.append((j, similarity))
                
                # Sort by similarity (highest first)
                similar_texts.sort(key=lambda x: x[1], reverse=True)
                
                semantic_checks.append({
                    "original_idx": i,
                    "original_text": texts[i][:100] + "...",
                    "rephrased_text": rephrased_text[:100] + "...",
                    "similar_texts": [
                        {"idx": idx, "similarity": sim, "text": texts[idx][:50] + "..."}
                        for idx, sim in similar_texts[:3]
                    ]
                })
            
            # Evaluate overall quality
            avg_similarity = sum(check["similar_texts"][0]["similarity"] 
                               for check in semantic_checks) / len(semantic_checks)
            
            quality_score = avg_similarity
            passed = quality_score >= self.similarity_threshold
            
            return {
                "passed": passed,
                "quality_score": quality_score,
                "consistency_check": consistency_result,
                "semantic_checks": semantic_checks[:3],  # Include a few examples
                "threshold": self.similarity_threshold
            }
        
        except Exception as e:
            logger.error(f"Error evaluating embedding quality: {str(e)}")
            raise QualityCheckError(f"Error evaluating embedding quality: {str(e)}")
    
    def validate_retrieval_pipeline(self, 
                                   sample_queries: List[str], 
                                   retrieval_func: callable,
                                   k: int = 5) -> Dict[str, Any]:
        """
        Validate the end-to-end retrieval pipeline.
        
        Args:
            sample_queries (List[str]): List of sample queries.
            retrieval_func (callable): Function that takes a query and returns results.
            k (int): Number of results to retrieve.
            
        Returns:
            Dict[str, Any]: Validation results.
            
        Raises:
            QualityCheckError: If validation fails.
        """
        if not sample_queries:
            logger.warning("Empty queries list provided for pipeline validation")
            return {"passed": False, "reason": "No queries to validate with"}
        
        try:
            # Initialize LLM if not already initialized
            llm = self._init_llm()
            
            # Run queries through the retrieval pipeline
            query_results = []
            relevance_checks = []
            
            for i, query in enumerate(sample_queries):
                try:
                    results = retrieval_func(query, k)
                    
                    # Check relevance
                    relevance = self.check_query_result_relevance(query, results)
                    relevance_checks.append(relevance)
                    
                    query_results.append({
                        "query": query,
                        "results_count": len(results),
                        "relevance": relevance
                    })
                
                except Exception as e:
                    logger.error(f"Error processing query {query}: {str(e)}")
                    query_results.append({
                        "query": query,
                        "error": str(e)
                    })
                
                # Add a small delay between queries
                if i < len(sample_queries) - 1:
                    time.sleep(1)
            
            # Calculate average relevance
            relevance_scores = [check["average_relevance"] for check in relevance_checks if "average_relevance" in check]
            avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
            
            # Calculate success rate (queries without errors)
            success_rate = sum(1 for result in query_results if "error" not in result) / len(query_results)
            
            passed = avg_relevance >= self.similarity_threshold and success_rate >= 0.9
            
            return {
                "passed": passed,
                "average_relevance": avg_relevance,
                "success_rate": success_rate,
                "threshold": self.similarity_threshold,
                "queries_tested": len(sample_queries),
                "query_results": query_results
            }
        
        except Exception as e:
            logger.error(f"Error validating retrieval pipeline: {str(e)}")
            raise QualityCheckError(f"Error validating retrieval pipeline: {str(e)}") 