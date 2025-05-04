"""
Unit tests for the Weaviate vector store client.
"""
import pytest
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any

from src.financials.pipeline.weaviate_client import WeaviateVectorStore
from langchain.schema import Document


class TestWeaviateVectorStore:
    """Test suite for the WeaviateVectorStore class."""

    @patch('weaviate.connect_to_local')
    def test_init_local_client(self, mock_connect_local):
        """Test client initialization with local connection."""
        # Setup
        mock_client = MagicMock()
        mock_connect_local.return_value = mock_client
        
        # Execute
        store = WeaviateVectorStore(
            host="localhost",
            port=8080,
            grpc_port=50051
        )
        
        # Verify
        mock_connect_local.assert_called_once_with(
            host="localhost",
            port=8080,
            grpc_port=50051,
            headers=None
        )
        assert store.client == mock_client
        assert store.collection_name == "pdf_documents"

    @patch('weaviate.connect_to_weaviate_cloud')
    def test_init_cloud_client(self, mock_connect_cloud):
        """Test client initialization with cloud connection."""
        # Setup
        mock_client = MagicMock()
        mock_connect_cloud.return_value = mock_client
        
        # Execute
        store = WeaviateVectorStore(
            host="weaviate-cluster.example.com",
            port=443,
            grpc_port=443,
            auth_config={"X-OpenAI-Api-Key": "test-api-key"}
        )
        
        # Verify
        mock_connect_cloud.assert_called_once()
        assert store.client == mock_client

    @patch('weaviate.Client')
    def test_add_documents(self, mock_client):
        """Test adding documents with embeddings."""
        # Setup
        mock_collection = MagicMock()
        mock_batch = MagicMock()
        mock_collection.batch.dynamic.return_value.__enter__.return_value = mock_batch
        
        store = WeaviateVectorStore()
        store.collection = mock_collection
        
        # Test data
        docs = [
            Document(
                page_content="Test content 1",
                metadata={"source": "test1.pdf", "page": 1}
            ),
            Document(
                page_content="Test content 2",
                metadata={"source": "test2.pdf", "page": 2}
            )
        ]
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        
        # Execute
        result = store.add_documents(docs, embeddings)
        
        # Verify
        assert len(result) == 2
        assert mock_batch.add_object.call_count == 2

    @patch('weaviate.Client')
    def test_similarity_search(self, mock_client):
        """Test similarity search with query vector."""
        # Setup
        mock_collection = MagicMock()
        mock_results = MagicMock()
        
        # Create mock objects for the query results
        mock_obj1 = MagicMock()
        mock_obj1.properties = {"content": "Result 1", "source": "test1.pdf", "page": 1, "chunk_id": 1, "chunk_method": "test"}
        mock_obj1.metadata.distance = 0.1
        
        mock_obj2 = MagicMock()
        mock_obj2.properties = {"content": "Result 2", "source": "test2.pdf", "page": 2, "chunk_id": 2, "chunk_method": "test"}
        mock_obj2.metadata.distance = 0.2
        
        mock_results.objects = [mock_obj1, mock_obj2]
        mock_collection.query.near_vector.return_value = mock_results
        
        store = WeaviateVectorStore()
        store.collection = mock_collection
        
        # Test data
        query_vector = [0.1, 0.2, 0.3]
        
        # Execute
        results = store.similarity_search(query_vector, k=2)
        
        # Verify
        assert len(results) == 2
        assert results[0].page_content == "Result 1"
        assert results[0].metadata["source"] == "test1.pdf"
        assert results[0].metadata["distance"] == 0.1
        assert results[1].page_content == "Result 2"
        mock_collection.query.near_vector.assert_called_once_with(vector=query_vector, limit=2)

    @patch('weaviate.Client')
    def test_delete_collection(self, mock_client):
        """Test deleting a collection."""
        # Setup
        mock_client_instance = MagicMock()
        mock_collections = MagicMock()
        mock_client_instance.collections = mock_collections
        
        store = WeaviateVectorStore()
        store.client = mock_client_instance
        store.collection_name = "test_collection"
        
        # Execute
        store.delete_collection()
        
        # Verify
        mock_collections.delete.assert_called_once_with("test_collection")
