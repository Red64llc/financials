from financials.pipeline.embeddings import EmbeddingService
from financials.pipeline.weaviate_client import WeaviateVectorStore
from financials.chat.optimized_financial_extractor import OptimizedFinancialExtractor
import os
import pandas as pd
import logging

class ChatSystem:
    """Chat class for financial data analysis"""
    def __init__(self, collection_name: str = "Financials", logger: logging.Logger = None):
        logger = logger or logging.getLogger("ChatSystem")
        self.logger = logger
        self.logger.info("Initializing ChatSystem...")
        self.extractor = OptimizedFinancialExtractor()
        self.vectordb = WeaviateVectorStore(collection_name=collection_name)
        self.embedding_service = EmbeddingService()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        # Conversation history (should a memory)
        self.conversation_history = []
        logger.info("ChatSystem initialized successfully")

    def _format_financial_data_response(self, financial_data, user_query):
        """Format financial data into a natural language response"""
        # Create a natural language response based on the financial data
        if financial_data.empty:
            return "I couldn't find specific revenue data related to your query."
        
        # Format basic information about periods found
        periods = financial_data['period'].tolist()
        if len(periods) == 1:
            period_info = f"I found revenue data for {periods[0]}."
        else:
            period_info = f"I found revenue data for {len(periods)} periods from {periods[0]} to {periods[-1]}."
        
        # Format revenue information
        revenue_info = "Here's the revenue information:\n\n"
        for _, row in financial_data.iterrows():
            period = row['period']
            value = row['value']
            unit = row['unit']
            growth = row.get('growth_rate', None)
            
            revenue_line = f"• {period}: {value:,.2f} {unit}"
            if growth is not None and not pd.isna(growth):
                growth_pct = growth * 100
                direction = "increase" if growth_pct > 0 else "decrease"
                revenue_line += f" ({abs(growth_pct):.2f}% {direction} from previous period)"
            
            revenue_info += revenue_line + "\n"
        
        # Add trend information if available
        trend_info = ""
        if len(periods) > 1 and 'growth_rate' in financial_data.columns:
            avg_growth = financial_data['growth_rate'].mean() * 100
            trend = "increasing" if avg_growth > 0 else "decreasing"
            trend_info = f"\nOverall, revenue has been {trend} at an average rate of {abs(avg_growth):.2f}% per period."
        
        # Complete response
        response = f"{period_info}\n\n{revenue_info}{trend_info}"
        
        return response    
    def info(self):
        """Get information about the chat system"""
        count = self.vectordb.get_info()
        return f"ChatSystem initialized with Weaviate\n #Documents: {count}\n"
    
    def chat(
        self, 
        user_message: str,
        strict_mode: bool = False,
        k: int = 4
    ) -> str:
        """Process a user message and generate a response"""
        # Store user message
        self.conversation_history.append({"role": "user", "content": user_message})
        self.logger.info(f"User message: {user_message}")
        # Check if this is a query about financial data
        financial_keywords = ['revenue', 'sales', 'financial', 'earnings', 'growth']
        is_financial_query = any(keyword in user_message.lower() for keyword in financial_keywords)
        
        if is_financial_query or not strict_mode:
            # first gets the documents that are relevant the query
            query = user_message
            vector = self.embedding_service.embed_query(query)
            documents = self.vectordb.similarity_search(vector, k=10)
            
            self.logger.info(f"Found {len(documents)} relevant documents for query: {query}")
            # then extract the financial data from the documents
            # this is much faster than extracting from the files direclty    
            financial_data = self.extractor.extract(documents)
            
            if financial_data is not None:
                # Convert financial data to a response
                response = self._format_financial_data_response(financial_data, user_message)
            else:
                response = "I couldn't find relevant financial data for your query."
        else:
            # For non-financial queries, just use regular RAG without extraction
            # Here you would implement standard RAG response generation
            response = "Ask me about financial data :-)"
        
        # Store assistant response
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response    