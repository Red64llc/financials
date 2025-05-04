
from financials.pipeline.weaviate_client import WeaviateVectorStore
from financials.chat.optimized_financial_extractor import OptimizedFinancialExtractor

class ChatSystem:
    """Chat class for financial data analysis"""
    def __init__(self):
        self.extractor = OptimizedFinancialExtractor()
        self.vectordb = WeaviateVectorStore()

        # Conversation history
        self.conversation_history = []
    
    def info(self):
        """Get information about the chat system"""
        collections = self.vectordb.get_collections()
        return f"ChatSystem initialized with Weaviate\n Collections Available: {collections}\n"
    
    def chat(self, query):
        """Chat with the financial data analysis system"""
        # message = self.extractor.chat(query)
        message = "to be implement soon..."
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": message})
        return message
    