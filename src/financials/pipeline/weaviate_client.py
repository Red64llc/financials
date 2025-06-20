"""
Weaviate client module for vector database operations.

This module provides a wrapper around the Weaviate client to store and
retrieve documents with vector embeddings.
"""
import os
import logging
from typing import List, Dict, Any, Optional, Union

import weaviate
from weaviate.client import WeaviateClient
from weaviate.collections import Collection
import weaviate.classes as wvc
from weaviate.classes.init import Auth
from langchain.schema import Document

logger = logging.getLogger(__name__)

class WeaviateVectorStore:
    """
    Weaviate vector store for storing and retrieving documents with embeddings.
    
    This class provides a high-level interface to interact with Weaviate,
    optimized for document storage and retrieval in the financial data analysis context.
    """
    
    def __init__(
        self,
        client: Optional[WeaviateClient] = None,
        collection_name: str = "pdf_documents",
        host: str = "localhost",
        port: int = 8080,
        grpc_port: int = 50051,
        embedding_dim: int = 1536,  # OpenAI embedding dimension
        auth_config: Optional[Dict[str, Any]] = None,
        batch_size: int = 100
    ):
        """Initialize the Weaviate vector store.
        
        Args:
            client: An existing Weaviate client instance (optional)
            collection_name: Name of the collection to store documents in
            host: Weaviate host (if client not provided)
            port: Weaviate HTTP port (if client not provided)
            grpc_port: Weaviate gRPC port (if client not provided)
            embedding_dim: Dimension of the embeddings
            auth_config: Authentication configuration (optional)
            batch_size: Batch size for operations
        """
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        
        # Set up client
        self._init_client(client, host, port, grpc_port, auth_config)
        
        # Set up collection
        self.collection = self._init_collection()
    
    def _init_client(
        self,
        client: Optional[WeaviateClient],
        host: str,
        port: int,
        grpc_port: int,
        auth_config: Optional[Dict[str, Any]]
    ) -> None:
        """Initialize the Weaviate client."""
        if client:
            self.client = client
            return
        if host is None or port is None or grpc_port is None:
            raise ValueError("host, port, and grpc_port must be provided if client is not provided")
        
        # Set up connection based on environment (local, docker or cloud)
        if host == "weaviate" or host == "localhost" or host.startswith("127.0.0.1"):
            logger.info(f"Connecting to local Weaviate at {host}:{port}")
            self.client = weaviate.connect_to_local(
                host=host,
                port=port,
                grpc_port=grpc_port,
                headers=auth_config
            )
        else:
            logger.info(f"Connecting to remote Weaviate at {host}:{port}")
            # For cloud/remote deployment
            cluster_url = f"http://{host}:{port}"
            
            # Configure authentication credentials if provided
            auth_credentials = None
            if auth_config and "X-OpenAI-Api-Key" in auth_config:
                # Extract OpenAI API key if needed for vectorizer
                openai_api_key = auth_config.get("X-OpenAI-Api-Key")
                # Additional headers for vectorizer modules if needed
                additional_headers = {k: v for k, v in auth_config.items() if k != "X-OpenAI-Api-Key"}
                
                # If there's an API key for Weaviate Cloud in auth_config
                if "Authorization" in auth_config:
                    weaviate_api_key = auth_config.get("Authorization").replace("Bearer ", "")
                    auth_credentials = Auth.api_key(weaviate_api_key)
                    
            # Connect to Weaviate Cloud
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=cluster_url,
                auth_credentials=auth_credentials,
                headers=auth_config  # Pass any additional headers
            )
        
        logger.info(f"Weaviate client initialized, connecting to {host}:{port}")
    
    def _init_collection(self) -> Collection:
        """Initialize or get the collection."""
        # Check if collection exists
        try:
            collection = self.client.collections.get(self.collection_name)
            logger.info(f"Retrieved existing collection: {self.collection_name}")
            return collection
        except Exception as e:
            logger.info(f"Collection {self.collection_name} not found, creating new: {str(e)}")
        
        # Create a new collection with schema
        collection = self.client.collections.create(
            name=self.collection_name,
            properties=[
                # Document content
                {
                    "name": "content",
                    "data_type": ["text"],
                    "description": "Document content text",
                    "tokenization": "word",
                    "indexed": True,
                },
                # Metadata fields
                {
                    "name": "source",
                    "data_type": ["text"],
                    "description": "Source document",
                    "indexed": True,
                },
                {
                    "name": "page",
                    "data_type": ["int"],
                    "description": "Page number",
                    "indexed": True,
                },
                {
                    "name": "chunk_id",
                    "data_type": ["int"],
                    "description": "Chunk identifier",
                    "indexed": True,
                },
                {
                    "name": "chunk_method",
                    "data_type": ["text"],
                    "description": "Method used for chunking",
                    "indexed": True,
                }
            ],
            # Vector index configuration
            vectorizer_config=wvc.config.Configure.Vectorizer.none(
                vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                    distance_metric=wvc.config.Configure.VectorDistance.cosine
                )
            )
        )
        
        logger.info(f"Created new collection: {self.collection_name}")
        return collection
    
    def get_info(self):
        collection = self.client.collections.get("Financials")
        response = collection.aggregate.over_all(total_count=True)
        return response.total_count
    
    def get_collections(self):
        """Get list of collections in the Weaviate instance."""
        collections = self.client.collections.list_all(simple=True)
        return collections
    
    def add_documents(
        self,
        documents: List[Document],
        embeddings: List[List[float]]
    ) -> List[str]:
        """Add documents with embeddings to the vector store.
        
        Args:
            documents: List of langchain Document objects
            embeddings: List of embedding vectors (must match documents length)
            
        Returns:
            List of document IDs
        """
        if len(documents) != len(embeddings):
            raise ValueError(f"Documents length ({len(documents)}) must match embeddings length ({len(embeddings)})")
        
        # Track document IDs
        document_ids = []
        
        # Create batch
        with self.collection.batch.dynamic() as batch:
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                # Prepare metadata
                metadata = doc.metadata.copy() if doc.metadata else {}
                
                # Ensure required metadata fields
                if 'page' not in metadata:
                    metadata['page'] = 0
                if 'chunk_id' not in metadata:
                    metadata['chunk_id'] = i
                if 'source' not in metadata:
                    metadata['source'] = "unknown"
                if 'chunk_method' not in metadata:
                    metadata['chunk_method'] = "default"
                
                # Prepare properties
                properties = {
                    "content": doc.page_content,
                    "source": metadata.get("source"),
                    "page": metadata.get("page"),
                    "chunk_id": metadata.get("chunk_id"),
                    "chunk_method": metadata.get("chunk_method"),
                }
                
                # Add object to batch with embedding
                doc_id = batch.add_object(
                    properties=properties,
                    vector=embedding
                )
                document_ids.append(doc_id)
        
        logger.info(f"Added {len(documents)} documents to Weaviate collection {self.collection_name}")
        return document_ids

    def retrieve_relevant_chunks(self, query_vector):
        """Retrieve relevant chunks from vector database"""
        response = self.collection.query.near_vector(
            near_vector=query_vector,
            limit=20
        )
        results = response.objects
        # Convert to a format compatible with our extractor
        chunks = []
        for item in results:
            properties = item.properties
            chunk = Document(
                page_content=properties.get("content", ""),
                metadata={
                    "source": properties.get("source", ""),
                    "page": properties.get("page", 0),
                    "chunk_id": properties.get("chunk_id", 0),
                    "chunk_method": properties.get("chunk_method", ""),
                    "distance": item.metadata.distance,
                }
            )
            chunks.append(chunk)
        
        return chunks

    def similarity_search(
        self,
        query_vector: List[float],
        k: int = 4
    ) -> List[Document]:
        """Perform a similarity search using a query vector.
        
        Args:
            query_vector: The embedding vector to search with
            k: Number of results to return
            
        Returns:
            List of Document objects
        """
        # Perform the vector search
        response = self.collection.query.near_vector(
            near_vector=query_vector,
            limit=k
        )
        
        # Convert to documents
        documents = []
        results = response.objects
        
        for result in results:
            properties = result.properties
            
            doc = Document(
                page_content=properties.get("content", ""),
                metadata={
                    "source": properties.get("source", ""),
                    "page": properties.get("page", 0),
                    "chunk_id": properties.get("chunk_id", 0),
                    "chunk_method": properties.get("chunk_method", ""),
                    "distance": result.metadata.distance,
                }
            )
            documents.append(doc)
        
        return documents
    
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        try:
            self.client.collections.delete(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection {self.collection_name}: {str(e)}")
            raise
