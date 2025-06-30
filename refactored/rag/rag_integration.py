"""RAG integration for checklist creation"""
import os
import torch
from transformers import AutoTokenizer, AutoModel
from .opensearch_client import OpenSearchClient, EmbeddingClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RAGChecklistEnhancer:
    """Enhances checklist creation with RAG-retrieved context"""
    
    def __init__(self):
        self.opensearch_client = None
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize OpenSearch and embedding clients if environment variables are available"""
        try:
            host = os.getenv("OPENSEARCH_HOST")
            port = os.getenv("OPENSEARCH_PORT")
            user = os.getenv("OPENSEARCH_USER")
            password = os.getenv("OPENSEARCH_PASS")

            # Debug output for environment variables
            print(f"OpenSearch Host: {host}, Port: {port}, User: {user}")
            
            if not all([host, port, user, password]):
                print("OpenSearch credentials not found in environment. RAG functionality disabled.")
                return
            
            # Initialize embedding client
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer, model = self._load_model_and_tokenizer("nlpai-lab/KURE-v1", torch.device(device))
            embedding_client = EmbeddingClient(
                api_key=None, 
                device_type=device, 
                tokenizer=tokenizer, 
                model=model
            )
            
            # Initialize OpenSearch client
            auth = (user, password)
            self.opensearch_client = OpenSearchClient(host, int(port), auth, embedding_client)
            print("✓ RAG functionality enabled")
            
        except Exception as e:
            print(f"Failed to initialize RAG clients: {e}")
            self.opensearch_client = None
    
    def _load_model_and_tokenizer(self, model_name: str, device: torch.device):
        """Load tokenizer and model for embeddings"""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        return tokenizer, model
    
    def get_enhanced_prompt(self, original_prompt: str, subject: str, assessment_type: str, 
                          assessment_title: str, assessment_description: str) -> str:
        """Enhance the original prompt with RAG-retrieved context"""
        if not self.opensearch_client:
            return original_prompt
        
        try:
            # Create search query combining assessment components
            search_query = f"{assessment_type} {assessment_title} {assessment_description}"
            
            # Retrieve similar documents
            contexts = self.opensearch_client.search_similar_doc(
                user_query=search_query,
                index_name="korean",
                subject=subject,
                size=5,
                similarity_threshold=0.935
            )
            
            if not contexts:
                print("ⓘ No relevant documents found for RAG enhancement")
                return original_prompt
            
            # Format contexts for prompt inclusion
            formatted_contexts = self._format_contexts(contexts)
            
            # Create enhanced prompt
            enhanced_prompt = f"""다음은 OpenSearch를 통해 검색된 참고 문서입니다. 이 문서들을 참고하여 '{assessment_title}' 수행평가에 대한 평가 기준(체크리스트)을 생성해주세요.

[참고 문서들]
{formatted_contexts}

이 수행평가는 {subject} 과목의 {assessment_type} 유형의 수행평가입니다.
수행평가에 대한 설명은 다음과 같습니다:
{assessment_description}

{original_prompt}"""
            
            print(f"✓ RAG enhancement applied - {len(contexts)} documents retrieved")
            return enhanced_prompt
            
        except Exception as e:
            print(f"RAG enhancement failed: {e}")
            return original_prompt
    
    def _format_contexts(self, contexts):
        """Format retrieved contexts for prompt inclusion"""
        context_strings = []
        for i, ctx in enumerate(contexts):
            formatted = f"""[문서 {i+1}]
평가유형: {ctx.get('assessment_type', 'N/A')}
성취기준: {ctx.get('achievement_standard', 'N/A')}
개발 방향: {ctx.get('development_notes', 'N/A')}
채점 기준: {ctx.get('scoring_criteria', 'N/A')}"""
            context_strings.append(formatted)
        return "\n\n".join(context_strings)
    
    def is_available(self) -> bool:
        """Check if RAG functionality is available"""
        return self.opensearch_client is not None
