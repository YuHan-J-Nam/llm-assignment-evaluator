"""OpenSearch client and embedding functionality adapted for refactored structure"""
import os
from opensearchpy import OpenSearch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import torch

# Load environment variables
load_dotenv()

class OpenSearchClient:
    """OpenSearch client for document retrieval"""
    
    def __init__(self, host, port, auth, embedding_client):
        self.host = host
        self.port = port
        self.auth = auth
        self.embedding_client = embedding_client 

        # OpenSearch client configuration
        self.client = OpenSearch(
            hosts=[{'host': host, 'port': port}],
            http_auth=auth,
            use_ssl=True,
            timeout=80,
            verify_certs=False
        )
    
    def search_similar_doc(self, user_query, index_name, subject, size=5, top_k=1, similarity_threshold=0.935):
        """Perform similarity-based document search"""
        
        # Embed the query text
        embedded_user_query = self.embedding_client.embed_query(user_query)
        
        # Search query
        search_body = {
            "size": size,
            "query": {
                "bool": {
                    "must": [
                        {"match": {"subject_name": subject}},
                        {
                            "knn": {
                                "embedding": {
                                    "vector": embedded_user_query[0].tolist(),
                                    "k": top_k
                                }
                            }
                        }
                    ]
                }
            }
        }
        
        # Execute search
        try:
            response = self.client.search(index=index_name, body=search_body)
            hits = response["hits"]["hits"]
            
            fields = ["assessment_type", "achievement_standard", "scoring_criteria", "development_notes"]
            contexts = []
            
            for hit in hits:
                # Select documents that exceed similarity threshold
                similarity_score = hit['_score']
                if similarity_score >= similarity_threshold:
                    # Refine search results
                    source = hit.get("_source", {})
                    context = {field: source.get(field) if source.get(field) is not None else None for field in fields}
                    contexts.append(context)

            return contexts
        
        except Exception as e:
            print(f"[ERROR] Error during similarity document search: {e}")
            return []


class EmbeddingClient:
    """Client for text embedding using KURE model"""
    
    def __init__(self, api_key, device_type, tokenizer, model):
        self.api_key = api_key
        self.device_type = device_type
        self.device = torch.device(device_type)
        self.tokenizer = tokenizer
        self.model = model.to(self.device)
    
    def mean_pooling(self, model_output, attention_mask):
        """Create representative vector for segments"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)

    def embed_query(self, user_query):
        """Embed user query into vector representation"""
        self.model.eval()
        # Tokenize user query
        enc = self.tokenizer(user_query, padding=True, truncation=True, return_tensors="pt")
        # Move tensors to device (GPU or CPU)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        # Generate embeddings
        with torch.no_grad():
            out = self.model(**enc)
        # Create representative vector using mean pooling
        emb = self.mean_pooling(out, enc["attention_mask"])
        return emb.cpu().numpy()  # shape (1, D)
