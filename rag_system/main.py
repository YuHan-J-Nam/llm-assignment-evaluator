"""
Main entry point for the RAG system.
"""
import os
import argparse
from typing import List, Dict, Any, Optional

from rag_system.modules.pdf_extractor import PDFExtractor
from rag_system.modules.text_embedder import TextEmbedder
from rag_system.modules.vector_store import VectorStore
from rag_system.modules.quality_checker import QualityChecker
from rag_system.modules.data_retriever import DataRetriever
from rag_system.config import PDF_DATA_FOLDER
from rag_system.utils.logger import setup_logger
from rag_system.utils.exceptions import RAGBaseException


logger = setup_logger("main")


class RAGSystem:
    """
    Main class for the RAG (Retrieval-Augmented Generation) system.
    """
    
    def __init__(self, 
                pdf_folder: str = PDF_DATA_FOLDER,
                embedding_model: str = "openai",
                vector_db: str = "chroma"):
        """
        Initialize the RAG system.
        
        Args:
            pdf_folder (str): Path to the folder containing PDF files.
            embedding_model (str): Type of embedding model to use.
            vector_db (str): Type of vector database to use.
        """
        logger.info(f"Initializing RAG system with {embedding_model} embeddings and {vector_db} vector database")
        
        # Initialize components
        self.pdf_extractor = PDFExtractor(pdf_folder=pdf_folder)
        self.embedder = TextEmbedder(model_type=embedding_model)
        self.vector_store = VectorStore(embedding_model=self.embedder.embedding_model, db_type=vector_db)
        self.quality_checker = QualityChecker()
        self.data_retriever = DataRetriever(vector_store=self.vector_store, embedder=self.embedder)
        
        logger.info("RAG system initialized successfully")
    
    def process_pdf_files(self, specific_file: Optional[str] = None) -> int:
        """
        Process PDF files, extract text, and store in vector database.
        
        Args:
            specific_file (Optional[str]): Path to a specific PDF file to process.
                                         If None, all PDFs in the folder are processed.
        
        Returns:
            int: Number of documents added to the vector store.
        """
        try:
            # Extract text from PDFs and chunk
            logger.info("Extracting and chunking text from PDFs")
            pdf_chunks = self.pdf_extractor.extract_and_chunk(specific_file)
            
            total_docs = 0
            
            for pdf_path, chunks in pdf_chunks:
                if not chunks:
                    logger.warning(f"No chunks extracted from {pdf_path}")
                    continue
                
                # Create metadata for chunks
                pdf_name = os.path.basename(pdf_path)
                metadata = [{"source": pdf_name, "chunk_index": i, "path": pdf_path} for i in range(len(chunks))]
                
                # Generate embeddings
                logger.info(f"Generating embeddings for {len(chunks)} chunks from {pdf_name}")
                embedded_docs = self.embedder.get_document_embeddings(chunks, metadata)
                
                # Add to vector store
                logger.info(f"Adding {len(embedded_docs)} documents to vector store")
                self.vector_store.add_documents(embedded_docs)
                
                total_docs += len(embedded_docs)
                
                # Perform quality checks
                self.run_quality_checks(chunks, [doc.get("embedding", []) for doc in embedded_docs])
            
            logger.info(f"Processed {len(pdf_chunks)} PDF files, added {total_docs} documents to vector store")
            return total_docs
        
        except Exception as e:
            logger.error(f"Error processing PDF files: {str(e)}")
            raise
    
    def run_quality_checks(self, texts: List[str], embeddings: List[List[float]]) -> Dict[str, Any]:
        """
        Run quality checks on embeddings.
        
        Args:
            texts (List[str]): List of text chunks.
            embeddings (List[List[float]]): List of embedding vectors.
        
        Returns:
            Dict[str, Any]: Quality check results.
        """
        try:
            logger.info("Running quality checks on embeddings")
            quality_results = self.quality_checker.evaluate_embedding_quality(texts, embeddings)
            
            if quality_results.get("passed", False):
                logger.info(f"Quality checks passed with score: {quality_results.get('quality_score', 0)}")
            else:
                logger.warning(f"Quality checks failed: {quality_results.get('reason', 'Unknown reason')}")
            
            return quality_results
        
        except Exception as e:
            logger.error(f"Error running quality checks: {str(e)}")
            return {"passed": False, "error": str(e)}
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant documents based on a query.
        
        Args:
            query (str): Query string.
            k (int): Number of results to return.
        
        Returns:
            List[Dict[str, Any]]: List of relevant documents.
        """
        try:
            logger.info(f"Searching for query: {query}")
            results = self.data_retriever.search(query, k=k)
            logger.info(f"Found {len(results)} results for query")
            return results
        
        except Exception as e:
            logger.error(f"Error searching for query '{query}': {str(e)}")
            raise
    
    def get_context(self, query: str, k: int = 5) -> str:
        """
        Get context for a query as a string.
        
        Args:
            query (str): Query string.
            k (int): Number of results to include.
        
        Returns:
            str: Context string from relevant documents.
        """
        try:
            logger.info(f"Getting context for query: {query}")
            context = self.data_retriever.get_related_context(query, k=k)
            return context
        
        except Exception as e:
            logger.error(f"Error getting context for query '{query}': {str(e)}")
            raise
    
    def validate_retrieval(self, sample_queries: List[str]) -> Dict[str, Any]:
        """
        Validate the retrieval pipeline with sample queries.
        
        Args:
            sample_queries (List[str]): List of sample queries.
        
        Returns:
            Dict[str, Any]: Validation results.
        """
        try:
            logger.info(f"Validating retrieval with {len(sample_queries)} sample queries")
            results = self.quality_checker.validate_retrieval_pipeline(
                sample_queries, self.data_retriever.search
            )
            
            if results.get("passed", False):
                logger.info("Retrieval validation passed")
            else:
                logger.warning("Retrieval validation failed")
            
            return results
        
        except Exception as e:
            logger.error(f"Error validating retrieval: {str(e)}")
            return {"passed": False, "error": str(e)}
    
    def clear_vector_store(self) -> bool:
        """
        Clear all documents from the vector store.
        
        Returns:
            bool: True if successful.
        """
        try:
            logger.info("Clearing vector store")
            success = self.vector_store.clear()
            if success:
                logger.info("Vector store cleared successfully")
            else:
                logger.warning("Failed to clear vector store")
            return success
        
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            raise


def main():
    """
    Main function for running the RAG system from the command line.
    """
    parser = argparse.ArgumentParser(description="RAG System")
    parser.add_argument("--pdf_folder", type=str, default=PDF_DATA_FOLDER,
                        help="Path to the folder containing PDF files")
    parser.add_argument("--embedding_model", type=str, default="openai",
                        choices=["openai", "huggingface", "bge", "korean"],
                        help="Type of embedding model to use")
    parser.add_argument("--vector_db", type=str, default="chroma",
                        choices=["chroma", "faiss"],
                        help="Type of vector database to use")
    parser.add_argument("--process", action="store_true",
                        help="Process PDF files and update vector store")
    parser.add_argument("--clear", action="store_true",
                        help="Clear the vector store")
    parser.add_argument("--query", type=str,
                        help="Search query")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of results to return")
    
    args = parser.parse_args()
    
    try:
        # Initialize the RAG system
        rag_system = RAGSystem(
            pdf_folder=args.pdf_folder,
            embedding_model=args.embedding_model,
            vector_db=args.vector_db
        )
        
        # Process command-line arguments
        if args.clear:
            rag_system.clear_vector_store()
        
        if args.process:
            num_docs = rag_system.process_pdf_files()
            print(f"Processed PDFs and added {num_docs} documents to vector store")
        
        if args.query:
            results = rag_system.search(args.query, k=args.k)
            print(f"\nSearch results for: {args.query}")
            print(f"Found {len(results)} results")
            
            for i, result in enumerate(results):
                print(f"\nResult {i+1} (score: {result.get('score', 0):.4f}):")
                
                # Get source information
                source = "Unknown source"
                if "metadata" in result and "source" in result["metadata"]:
                    source = result["metadata"]["source"]
                
                print(f"Source: {source}")
                
                # Truncate text for display
                text = result.get("text", "")
                if len(text) > 300:
                    text = text[:300] + "..."
                
                print(f"Text: {text}")
        
        if not any([args.clear, args.process, args.query]):
            parser.print_help()
    
    except RAGBaseException as e:
        logger.error(f"RAG system error: {str(e)}")
        print(f"Error: {str(e)}")
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    main() 