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
        self.embedder = EmbeddingService()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        # Conversation history (should a memory)
        self.conversation_history = []
        logger.info("ChatSystem initialized successfully")

    def _format_financial_data_response(self, financial_data, user_query):
        """Format financial data into a natural language response
        
        Args:
            financial_data: DataFrame containing the extracted financial data
            user_query: The user's query
        Returns:
            response: A natural language response based on the financial data
        Format of the financial data:
            period,confidence,unit,source,value,prev_value,growth_rate
        """
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
        revenue_info = "Here's the revenue information (period, value, unit, source, growth_rate):\n\n"
        for _, row in financial_data.iterrows():
            period = row['period']
            value = row['value']
            unit = row['unit']
            growth = row.get('growth_rate', None)
            source = row['source']
            
            revenue_line = f"• {period}: {value:,.2f} {unit} from {source}\n"
            if growth is not None and not pd.isna(growth):
                growth_pct = growth * 100
                direction = "increase" if growth_pct > 0 else "decrease"
                revenue_line += f" ({abs(growth_pct):.2f}% {direction} from previous period)\n"
            
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

    def retrieve_relevant_chunks(self, query):
        vector = self.embedder.embed_query(query)
        chunks = self.vectordb.retrieve_relevant_chunks(vector)
        self.logger.info(f"==> Found {len(chunks)} relevant chunks for query: {query}")
        return chunks

    def _find_relevant_entities(self, query, entity_results):
        """Find entities relevant to the query"""
        query_lower = query.lower()
        relevant_entities = []
        
        # Check for explicit entity mentions
        for entity_key in entity_results:
            entity_parts = entity_key.split('_')
            entity_type = entity_parts[0]
            
            # Check if the entity type is mentioned in the query
            if entity_type in query_lower:
                relevant_entities.append(entity_key)
                continue
                
            # Check if subtypes are mentioned
            if len(entity_parts) > 1:
                entity_subtype = entity_parts[1]
                if entity_subtype in query_lower:
                    relevant_entities.append(entity_key)
                    continue
            
            # Check for synonyms and related terms
            if entity_type == 'revenue' and any(term in query_lower for term in ['sales', 'turnover', 'top line']):
                relevant_entities.append(entity_key)
            elif entity_type == 'profit' and any(term in query_lower for term in ['earnings', 'income', 'bottom line']):
                relevant_entities.append(entity_key)
            elif entity_type == 'cost' and any(term in query_lower for term in ['expense', 'expenditure', 'spending']):
                relevant_entities.append(entity_key)
        
        # If no specific entities found, check for financial terms
        if not relevant_entities:
            financial_terms = [
                'financial', 'performance', 'metric', 'measure', 'figure', 'number', 
                'report', 'statement', 'result'
            ]
            
            if any(term in query_lower for term in financial_terms):
                # Return the first few entities as potentially relevant
                relevant_entities = list(entity_results.keys())[:3]
        
        return relevant_entities
    
    def _generate_entity_response(self, query, relevant_entities, entity_results):
        """Generate a response for the relevant entities"""
        responses = []
        
        for entity_key in relevant_entities:
            entity_data = entity_results[entity_key]['aggregate']
            
            if entity_data.empty:
                continue
                
            # Get entity type and subtype
            entity_parts = entity_key.split('_')
            entity_type = entity_parts[0]
            entity_subtype = entity_parts[1] if len(entity_parts) > 1 else None
            
            # Generate a response for this entity
            entity_response = self._format_entity_data(entity_type, entity_subtype, entity_data)
            responses.append(entity_response)
            
            # Check if dimensions were requested
            if any(term in query.lower() for term in ['segment', 'region', 'breakdown', 'by']):
                dimension_data = entity_results[entity_key]['dimensions']
                if dimension_data:
                    for dim_key, dim_data in dimension_data.items():
                        dim_parts = dim_key.split('_')
                        dim_type = dim_parts[0]
                        dim_value = dim_parts[1] if len(dim_parts) > 1 else None
                        
                        if not dim_data.empty:
                            dimension_response = self._format_dimension_data(
                                entity_type, entity_subtype, dim_type, dim_value, dim_data
                            )
                            responses.append(dimension_response)
        
        # Combine the responses
        if responses:
            final_response = "Based on the financial data I found:\n\n"
            final_response += "\n\n".join(responses)
        else:
            final_response = "I found some financial entities, but couldn't extract specific data relevant to your query."
        
        return final_response
    
    def _format_entity_data(self, entity_type, entity_subtype, entity_data):
        """Format entity data into a natural language response"""
        # Sort by period
        try:
            entity_data['period_dt'] = entity_data['period'].apply(self.extractor._period_to_sortable_date)
            sorted_data = entity_data.sort_values('period_dt')
            sorted_data = sorted_data.drop('period_dt', axis=1)
        except:
            sorted_data = entity_data.sort_values('period')
        
        # Format entity name
        if entity_subtype and entity_subtype != 'unknown':
            entity_name = f"{entity_subtype}"
        else:
            entity_name = f"{entity_type}"
        
        # Capitalize first letter
        entity_name = entity_name.capitalize()
        
        # Prepare response
        response = f"**{entity_name}**\n"
        
        # Add latest value
        latest_row = sorted_data.iloc[-1]
        latest_period = latest_row['period']
        latest_value = latest_row['value']
        unit = latest_row['unit']
        
        response += f"The most recent {entity_name.lower()} ({latest_period}) is {latest_value:,.2f} {unit}."
        
        # Add growth information if available
        if 'growth_rate' in latest_row and not pd.isna(latest_row['growth_rate']):
            growth_rate = latest_row['growth_rate'] * 100
            direction = "increase" if growth_rate > 0 else "decrease"
            response += f" This represents a {abs(growth_rate):.2f}% {direction} from the previous period."
        
        # Add trend information if we have multiple periods
        if len(sorted_data) > 2:
            # Calculate average growth
            if 'growth_rate' in sorted_data.columns:
                avg_growth = sorted_data['growth_rate'].mean() * 100
                trend = "increasing" if avg_growth > 0 else "decreasing"
                response += f"\n\nOver the periods analyzed, {entity_name.lower()} has been {trend} at an average rate of {abs(avg_growth):.2f}% per period."
        
        return response
    
    def _format_dimension_data(self, entity_type, entity_subtype, dim_type, dim_value, dim_data):
        """Format dimension breakdown data"""
        # Sort by period
        try:
            dim_data['period_dt'] = dim_data['period'].apply(self.extractor._period_to_sortable_date)
            sorted_data = dim_data.sort_values('period_dt')
            sorted_data = sorted_data.drop('period_dt', axis=1)
        except:
            sorted_data = dim_data.sort_values('period')
        
        # Format entity and dimension names
        if entity_subtype and entity_subtype != 'unknown':
            entity_name = f"{entity_subtype}"
        else:
            entity_name = f"{entity_type}"
        
        entity_name = entity_name.capitalize()
        dim_type_formatted = dim_type.replace('_', ' ').capitalize()
        
        # Prepare response
        response = f"**{entity_name} breakdown by {dim_type_formatted}: {dim_value.capitalize()}**\n"
        
        # Add latest value
        latest_row = sorted_data.iloc[-1]
        latest_period = latest_row['period']
        latest_value = latest_row['value']
        unit = latest_row['unit']
        
        response += f"For {dim_value.capitalize()}, the most recent {entity_name.lower()} ({latest_period}) is {latest_value:,.2f} {unit}."
        
        # Add growth information if available
        if 'growth_rate' in latest_row and not pd.isna(latest_row['growth_rate']):
            growth_rate = latest_row['growth_rate'] * 100
            direction = "increase" if growth_rate > 0 else "decrease"
            response += f" This represents a {abs(growth_rate):.2f}% {direction} from the previous period."
        
        return response

    def generate_response(self, query, extraction_results):
        """Generate a response based on the query and extraction results"""
        if not extraction_results or 'entity_results' not in extraction_results:
            return "I couldn't find relevant financial data for your query."
        
        # Extract relevant parts from the results
        entity_results = extraction_results['entity_results']
        summary = extraction_results['summary']
        
        # Determine which entities are relevant to the query
        relevant_entities = self._find_relevant_entities(query, entity_results)
        
        if not relevant_entities:
            return "I found some financial data, but none seems directly relevant to your query."
        
        # Generate a response focused on the relevant entities
        response = self._generate_entity_response(query, relevant_entities, entity_results)
        
        return response        

    def process_query(self, query):
        """Process a financial query using RAG and enhanced extraction"""
        # Step 1: Retrieve relevant chunks from vector database
        relevant_chunks = self.retrieve_relevant_chunks(query)
        
        # Step 2: Process the chunks with the enhanced financial extractor
        extraction_results = self.extractor.extract(relevant_chunks)
        
        # Step 3: Generate a response based on the extracted data
        response = self.generate_response(query, extraction_results)
        
        return response

    def chat(
        self, 
        user_message: str,
        strict_mode: bool = False,
        k: int = 4
    ) -> str:
        """Process a user message and generate a response"""
        self.logger.info(f"User message: {user_message}")

        self.conversation_history.append({"role": "user", "content": user_message})
        response = self.process_query(user_message)
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response    