from financials.pipeline.embeddings import EmbeddingService
from financials.pipeline.weaviate_client import WeaviateVectorStore
from financials.chat.optimized_financial_extractor import OptimizedFinancialExtractor
import os

class ChatSystem:
    """Chat class for financial data analysis"""
    def __init__(self, collection_name: str = "Financials"):
        self.extractor = OptimizedFinancialExtractor()
        self.vectordb = WeaviateVectorStore(collection_name=collection_name)
        self.embedding_service = EmbeddingService()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        # Conversation history (should a memory)
        self.conversation_history = []
    


    def info(self):
        """Get information about the chat system"""
        collections = self.vectordb.get_collections()
        return f"ChatSystem initialized with Weaviate\n Collections Available: {collections}\n"
    
    def chat(self, query):
        """Chat with the financial data analysis system"""
        vector = self.embedding_service.embed_query(query)
        matches = self.vectordb.similarity_search(vector, k=4)

        self.conversation_history.append({"role": "user", "content": query})
        message = [match.page_content for match in matches]
        self.conversation_history.append({"role": "assistant", "content": message})
        return message
    