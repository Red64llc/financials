import os
from typing import List, Dict, Any
import logging

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from financials.pipeline.embeddings import EmbeddingService
from financials.pipeline.weaviate_client import WeaviateVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    Pipeline for processing PDF documents and storing them in a vector database.
    
    The pipeline consists of:
    1. Loading PDF with unstructured
    2. Extracting and processing elements
    3. Chunking the content
    4. Generating embeddings
    5. Storing in vector database
    """
    
    def __init__(
        self,
        openai_api_key: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        db_host: str = None,
        db_port: int = None,
        db_grpc_port: int = None
    ):
        """Initialize the pipeline with required configurations."""
        self.openai_api_key = openai_api_key
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Get connection details from environment variables or use defaults/passed values
        self.db_host = db_host or os.getenv("WEAVIATE_HOST", "localhost")
        self.db_port = db_port or int(os.getenv("WEAVIATE_PORT", "8080"))
        self.db_grpc_port = db_grpc_port or int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
        
        # Initialize embeddings service
        self.embedding_service = EmbeddingService(
            embedding_type="openai",
            api_key=openai_api_key
        )
        
        # Initialize Weaviate vector store
        self.vector_store = WeaviateVectorStore(
            host=self.db_host,  # Use self.db_host
            port=self.db_port,  # Use self.db_port
            grpc_port=self.db_grpc_port,  # Use self.db_grpc_port
            auth_config={"X-OpenAI-Api-Key": openai_api_key}
        )
        
        # Initialize text splitter as backup
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_pdf_with_unstructured(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Step 1: Load PDF using unstructured package.
        
        Unstructured provides several partitioning strategies:
        - 'auto': Automatically detect the best strategy
        - 'hi_res': High resolution partitioning (slower but more accurate)
        - 'ocr_only': Use OCR for scanned documents
        - 'fast': Faster processing with lower accuracy
        """
        try:
            logger.info(f"Loading PDF: {pdf_path}")
            
            # Basic PDF loading with unstructured
            elements = partition_pdf(
                filename=pdf_path,
                strategy="auto",  # Can be changed to 'hi_res', 'ocr_only', 'fast'
                
                # Additional parameters for enhanced extraction
                extract_images_in_pdf=False,  # Set to True to extract images
                infer_table_structure=True,   # Extract tables
                languages=["eng"],            # Specify language for OCR
                
                # Advanced OCR options (if using OCR)
                pdf_infer_table_structure=True,
                hi_res_model_name="yolox",  # For better table detection
                
                # Layout analysis
                max_characters=5000000,  # Max characters per page
                
                # Metadata extraction
                extract_image_block_types=["Image", "Table"],
                include_page_breaks=True
            )
            
            # Convert elements to dictionaries for easier processing
            element_dicts = [element.to_dict() for element in elements]
            
            logger.info(f"Extracted {len(elements)} elements from PDF")
            return element_dicts
            
        except Exception as e:
            logger.error(f"Error loading PDF: {str(e)}")
            raise
    
    def process_elements(self, elements: List[Dict[str, Any]]) -> List[Document]:
        """
        Step 2: Process and structure the extracted elements.
        
        Element types from unstructured:
        - Title: Document or section titles
        - NarrativeText: Regular text content
        - Table: Structured table data
        - ListItem: Bullet points or numbered lists
        - Image: Image elements (if extracted)
        - Header/Footer: Page headers and footers
        - Figure: Figure captions
        - Formula: Mathematical expressions
        - CompositeElement: Mixed content
        """
        documents = []
        current_page = 1
        page_content = []
        
        for element in elements:
            element_type = element.get('type', 'Unknown')
            text = element.get('text', '')
            metadata = element.get('metadata', {})
            
            # Extract page number if available
            if 'page_number' in metadata:
                page_num = metadata['page_number']
                
                # If we've moved to a new page, save the current page content
                if page_num != current_page and page_content:
                    documents.append(Document(
                        page_content='\n'.join(page_content),
                        metadata={
                            'page': current_page,
                            'source': metadata.get('filename', 'unknown')
                        }
                    ))
                    page_content = []
                    current_page = page_num
            
            # Format content based on element type
            if element_type == 'Title':
                formatted_text = f"# {text}\n"
            elif element_type == 'Table':
                # Format table data if available
                formatted_text = f"[TABLE]\n{text}\n"
            elif element_type == 'ListItem':
                formatted_text = f"- {text}\n"
            else:
                formatted_text = text
            
            page_content.append(formatted_text)
        
        # Don't forget the last page
        if page_content:
            documents.append(Document(
                page_content='\n'.join(page_content),
                metadata={
                    'page': current_page,
                    'source': metadata.get('filename', 'unknown')
                }
            ))
        
        logger.info(f"Created {len(documents)} document chunks")
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Step 3: Chunk the documents for better retrieval.
        
        Two chunking approaches:
        1. Unstructured's built-in chunking
        2. LangChain's text splitter
        """
        chunked_documents = []
        
        for doc in documents:
            # Option 1: Use unstructured's chunking by title
            # This works better for well-structured documents
            try:
                # Convert document to unstructured elements
                elements = partition_pdf(text=doc.page_content)
                chunked_elements = chunk_by_title(
                    elements,
                    max_characters=self.chunk_size,
                    overlap=self.chunk_overlap,
                    combine_text_under_n_chars=50
                )
                
                # Convert back to documents
                for i, chunk in enumerate(chunked_elements):
                    chunk_doc = Document(
                        page_content=chunk.text,
                        metadata={
                            **doc.metadata,
                            'chunk_id': i,
                            'chunk_method': 'unstructured_title'
                        }
                    )
                    chunked_documents.append(chunk_doc)
                    
            except Exception as e:
                logger.warning(f"Failed chunking with unstructured, falling back to text splitter: {e}")
                
                # Option 2: Fallback to LangChain's text splitter
                chunks = self.text_splitter.split_documents([doc])
                for i, chunk in enumerate(chunks):
                    chunk.metadata['chunk_id'] = i
                    chunk.metadata['chunk_method'] = 'recursive_text_splitter'
                    chunked_documents.append(chunk)
        
        logger.info(f"Created {len(chunked_documents)} chunks")
        return chunked_documents
    
    def generate_embeddings(self, documents: List[Document]) -> List[List[float]]:
        """
        Step 4: Generate embeddings for the document chunks.
        
        Uses the embedding service which supports multiple backends:
        - OpenAI embeddings
        - HuggingFace embeddings
        - Cohere embeddings
        - Local models like BERT
        """
        texts = [doc.page_content for doc in documents]
        
        try:
            embeddings = self.embedding_service.embed_documents(texts)
            logger.info(f"Generated embeddings for {len(texts)} documents")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def store_in_vectordb(
        self, 
        documents: List[Document],
        embeddings: List[List[float]],
        collection_name: str = "pdf_documents"
    ) -> int:
        """
        Step 5: Store documents and embeddings in vector database.
        
        Weaviate is used as the vector database.
        """
        try:
            # Set the collection name if different from default
            if collection_name != self.vector_store.collection_name:
                self.vector_store.collection_name = collection_name
                self.vector_store.collection = self.vector_store._init_collection()
            
            # Add documents to the vector store
            doc_ids = self.vector_store.add_documents(documents, embeddings)
            
            logger.info(f"Stored {len(documents)} documents in Weaviate vector database")
            return len(doc_ids)
            
        except Exception as e:
            logger.error(f"Error storing in vector database: {str(e)}")
            raise
    
    def process_pdf_to_vectordb(
        self, 
        pdf_path: str, 
        collection_name: str = "pdf_documents"
    ) -> int:
        """
        Complete pipeline: Process PDF and store in vector database.
        """
        # Step 1: Load PDF with unstructured
        elements = self.load_pdf_with_unstructured(pdf_path)
        
        # Step 2: Process elements
        documents = self.process_elements(elements)
        
        # Step 3: Chunk documents
        chunked_documents = self.chunk_documents(documents)
        
        # Step 4: Generate embeddings
        texts = [doc.page_content for doc in chunked_documents]
        embeddings = self.embedding_service.embed_documents(texts)
        
        # Step 5: Store in vector database
        nb_docs = self.store_in_vectordb(chunked_documents, embeddings, collection_name)
        
        logger.info(f"PDF processing pipeline completed successfully with {nb_docs} documents")
        return nb_docs
