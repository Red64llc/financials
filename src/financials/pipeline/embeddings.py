"""
Embeddings module for generating vector representations of text.

This module provides wrapper classes for various embedding models,
with a focus on document vectorization for financial text analysis.
"""
import os
import logging
from typing import List, Dict, Any, Optional, Union

from langchain_openai import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Service for generating embeddings from text using different models.
    
    This class provides a consistent interface to generate embeddings
    using various backend models, with OpenAI as the default.
    """
    
    def __init__(
        self,
        embedding_type: str = "openai",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """Initialize the embedding service.
        
        Args:
            embedding_type: Type of embedding model ("openai", "huggingface", etc.)
            model_name: Name of the specific model to use
            api_key: API key for the embedding service
            **kwargs: Additional arguments for the embedding model
        """
        self.embedding_type = embedding_type
        self.model = self._init_embedding_model(embedding_type, model_name, api_key, **kwargs)
    
    def _init_embedding_model(
        self,
        embedding_type: str,
        model_name: Optional[str],
        api_key: Optional[str],
        **kwargs
    ) -> Embeddings:
        """Initialize the embedding model based on the type.
        
        Args:
            embedding_type: Type of embedding model
            model_name: Specific model name
            api_key: API key for the service
            **kwargs: Additional model parameters
            
        Returns:
            Initialized embedding model
        """
        if embedding_type == "openai":
            # OpenAI embeddings (default)
            openai_api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OpenAI API key is required for OpenAI embeddings")
            
            model = OpenAIEmbeddings(
                openai_api_key=openai_api_key,
                model=model_name or "text-embedding-ada-002",
                **kwargs
            )
            logger.info(f"Initialized OpenAI embedding model: {model_name or 'text-embedding-ada-002'}")
            return model
        
        elif embedding_type == "huggingface":
            # For future implementation: HuggingFace embeddings
            from langchain.embeddings import HuggingFaceEmbeddings
            
            model = HuggingFaceEmbeddings(
                model_name=model_name or "sentence-transformers/all-mpnet-base-v2",
                **kwargs
            )
            logger.info(f"Initialized HuggingFace embedding model: {model_name or 'sentence-transformers/all-mpnet-base-v2'}")
            return model
        
        elif embedding_type == "cohere":
            # For future implementation: Cohere embeddings
            from langchain_cohere import CohereEmbeddings
            
            cohere_api_key = api_key or os.getenv("COHERE_API_KEY")
            if not cohere_api_key:
                raise ValueError("Cohere API key is required for Cohere embeddings")
            
            model = CohereEmbeddings(
                cohere_api_key=cohere_api_key,
                model=model_name or "embed-english-v3.0",
                **kwargs
            )
            logger.info(f"Initialized Cohere embedding model: {model_name or 'embed-english-v3.0'}")
            return model
            
        else:
            raise ValueError(f"Unsupported embedding type: {embedding_type}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents.
        
        Args:
            texts: List of text documents
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = self.model.embed_documents(texts)
            logger.info(f"Generated embeddings for {len(texts)} documents")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating document embeddings: {str(e)}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query string.
        
        Args:
            text: Query text
            
        Returns:
            Embedding vector
        """
        try:
            embedding = self.model.embed_query(text)
            logger.debug(f"Generated embedding for query: {text[:50]}...")
            return embedding
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise
