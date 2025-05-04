# About RAG Input/Output


# Example usage
# def main():
#     """Example of using the PDF to Vector DB pipeline."""
    
#     # Configuration
#     openai_api_key = os.getenv("OPENAI_API_KEY")
#     pdf_path = "example.pdf"
    
#     # Initialize pipeline
#     pipeline = PDFProcessor(
#         openai_api_key=openai_api_key,
#         chunk_size=1000,
#         chunk_overlap=200,
#         db_directory="./chroma_db"
#     )
    
#     try:
#         # Process PDF and store in vector database
#         vectorstore = pipeline.process_pdf_to_vectordb(pdf_path)
        
#         # Example query
#         query = "What are the main topics discussed in this document?"
#         results = vectorstore.similarity_search(query, k=3)
        
#         for i, doc in enumerate(results):
#             print(f"\n--- Result {i+1} ---")
#             print(f"Content: {doc.page_content[:200]}...")
#             print(f"Metadata: {doc.metadata}")
            
#     except Exception as e:
#         logger.error(f"Pipeline failed: {str(e)}")




