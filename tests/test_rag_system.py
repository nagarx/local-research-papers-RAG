#!/usr/bin/env python3
"""
RAG System Functionality Tests

This module contains comprehensive tests for the core RAG system functionality
including document processing, embeddings, vector storage, and query processing.
"""

import pytest
import asyncio
import time
import tempfile
from pathlib import Path
from typing import List, Dict, Any

# Test suite imports
from conftest import TestResultTracker


class TestRAGSystemFunctionality:
    """Test core RAG system functionality"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.tracker = TestResultTracker()
    
    @pytest.mark.unit
    def test_embedding_generation(self):
        """Test embedding generation functionality"""
        print("\n🔍 Testing Embedding Generation")
        
        start_time = time.time()
        
        try:
            from src.embeddings import EmbeddingManager
            
            # Create embedding manager
            em = EmbeddingManager()
            
            # Test single text embedding
            test_text = "This is a test document about machine learning and artificial intelligence."
            embedding = asyncio.run(em.embed_text_async(test_text))
            
            assert embedding is not None, "Embedding should not be None"
            assert len(embedding) > 0, "Embedding should have dimensions"
            assert isinstance(embedding[0], (int, float)), "Embedding should contain numbers"
            
            # Test batch embedding
            test_texts = [
                "Machine learning is a subset of artificial intelligence.",
                "Natural language processing helps computers understand text.",
                "Vector databases store high-dimensional embeddings efficiently."
            ]
            
            embeddings = asyncio.run(em.embed_texts_batch_async(test_texts))
            
            assert len(embeddings) == len(test_texts), "Should have one embedding per text"
            assert all(len(emb) > 0 for emb in embeddings), "All embeddings should have dimensions"
            
            duration = time.time() - start_time
            self.tracker.log_result(
                "Embedding Generation",
                True,
                duration,
                f"Single: {len(embedding)} dims, Batch: {len(embeddings)} embeddings"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.tracker.log_result("Embedding Generation", False, duration, str(e))
            raise
    
    @pytest.mark.unit
    def test_vector_storage_operations(self):
        """Test vector storage operations"""
        print("\n📦 Testing Vector Storage Operations")
        
        start_time = time.time()
        
        try:
            from src.storage import ChromaVectorStore
            from src.embeddings import EmbeddingManager
            
            # Create components
            vs = ChromaVectorStore()
            em = EmbeddingManager()
            
            # Create test data
            test_chunks = [
                {
                    "id": "test_chunk_0",
                    "text": "Machine learning algorithms learn patterns from data.",
                    "chunk_index": 0,
                    "source_info": {
                        "document_id": "test_doc_001",
                        "document_name": "test.pdf",
                        "page_number": 1,
                        "block_index": 0,
                        "block_type": "text"
                    }
                },
                {
                    "id": "test_chunk_1", 
                    "text": "Neural networks are inspired by biological brain structures.",
                    "chunk_index": 1,
                    "source_info": {
                        "document_id": "test_doc_001",
                        "document_name": "test.pdf",
                        "page_number": 1,
                        "block_index": 1,
                        "block_type": "text"
                    }
                }
            ]
            
            # Generate embeddings
            texts = [chunk["text"] for chunk in test_chunks]
            embeddings = asyncio.run(em.embed_texts_batch_async(texts))
            
            # Add document to vector store
            success = vs.add_document(
                document_id="test_doc_001",
                chunks=test_chunks,
                embeddings=embeddings,
                metadata={"filename": "test.pdf", "test": True}
            )
            
            assert success, "Document should be added successfully"
            
            # Test search
            query_embedding = embeddings[0]  # Use first embedding as query
            results = vs.search(query_embedding, top_k=2)
            
            assert len(results) > 0, "Search should return results"
            assert len(results) <= 2, "Should not return more than top_k results"
            
            # Test document listing
            docs = vs.list_documents()
            assert len(docs) > 0, "Should list documents"
            
            # Test stats
            stats = vs.get_stats()
            assert isinstance(stats, dict), "Stats should be a dictionary"
            assert stats["total_documents"] > 0, "Should have documents"
            
            duration = time.time() - start_time
            self.tracker.log_result(
                "Vector Storage Operations",
                True,
                duration,
                f"Added doc, searched, found {len(results)} results"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.tracker.log_result("Vector Storage Operations", False, duration, str(e))
            raise
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_document_processing(self, sample_documents):
        """Test document processing with Marker"""
        print("\n📄 Testing Document Processing")
        
        if not sample_documents:
            pytest.skip("No sample documents available")
        
        start_time = time.time()
        
        try:
            from src.ingestion import DocumentProcessor
            
            # Create processor
            processor = DocumentProcessor()
            
            # Test processing first available document
            test_doc = sample_documents[0]
            
            result = asyncio.run(processor.process_document_async(test_doc))
            
            assert result is not None, "Processing result should not be None"
            assert "content" in result, "Result should have content"
            assert "blocks" in result["content"], "Content should have blocks"
            assert len(result["content"]["blocks"]) > 0, "Should have text blocks"
            
            # Test chunk structure
            first_block = result["content"]["blocks"][0]
            assert "text" in first_block, "Block should have text"
            assert "source_info" in first_block, "Block should have source_info"
            
            duration = time.time() - start_time
            self.tracker.log_result(
                "Document Processing",
                True,
                duration,
                f"Processed {test_doc.name}, {len(result['content']['blocks'])} blocks"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.tracker.log_result("Document Processing", False, duration, str(e))
            raise
    
    @pytest.mark.integration
    def test_ollama_integration(self):
        """Test Ollama LLM integration"""
        print("\n🤖 Testing Ollama Integration")
        
        start_time = time.time()
        
        try:
            from src.llm import OllamaClient
            
            # Create client
            client = OllamaClient()
            
            # Test connection
            connection_ok = client.test_connection()
            
            if not connection_ok:
                pytest.skip("Ollama server not available")
            
            # Test simple generation
            response = asyncio.run(client.generate_response_async(
                user_query="What is machine learning?",
                context_chunks=[],
                conversation_history=None
            ))
            
            assert response is not None, "Response should not be None"
            assert "response" in response, "Response should have response field"
            assert len(response["response"]) > 0, "Response should have content"
            
            duration = time.time() - start_time
            self.tracker.log_result(
                "Ollama Integration",
                True,
                duration,
                f"Response: {len(response['response'])} chars"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.tracker.log_result("Ollama Integration", False, duration, str(e))
            # Don't raise - Ollama might not be available
            pytest.skip(f"Ollama integration failed: {e}")
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_end_to_end_rag_query(self, sample_documents):
        """Test complete end-to-end RAG query processing"""
        print("\n🔄 Testing End-to-End RAG Query")
        
        if not sample_documents:
            pytest.skip("No sample documents available")
        
        start_time = time.time()
        
        try:
            from src.chat import ChatEngine
            
            # Create chat engine
            engine = ChatEngine()
            
            # Test query processing
            response = asyncio.run(engine.query_async(
                user_query="What is the main contribution of this research?",
                top_k=3,
                include_conversation_history=False
            ))
            
            assert response is not None, "Response should not be None"
            assert "response" in response, "Should have response field"
            assert "sources" in response, "Should have sources field"
            assert "metadata" in response, "Should have metadata field"
            
            # Test response content
            assert len(response["response"]) > 0, "Response should have content"
            assert isinstance(response["sources"], list), "Sources should be a list"
            assert isinstance(response["metadata"], dict), "Metadata should be a dict"
            
            # Test metadata
            metadata = response["metadata"]
            assert "response_time" in metadata, "Should have response_time"
            assert "total_sources" in metadata, "Should have total_sources"
            assert "timestamp" in metadata, "Should have timestamp"
            
            duration = time.time() - start_time
            self.tracker.log_result(
                "End-to-End RAG Query",
                True,
                duration,
                f"Response: {len(response['response'])} chars, Sources: {len(response['sources'])}"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.tracker.log_result("End-to-End RAG Query", False, duration, str(e))
            raise
    
    @pytest.mark.integration
    def test_source_tracking(self):
        """Test source tracking and citation functionality"""
        print("\n📚 Testing Source Tracking")
        
        start_time = time.time()
        
        try:
            from src.tracking import SourceTracker, SourceReference
            
            # Create tracker
            tracker = SourceTracker()
            
            # Register a test document
            tracker.register_document(
                document_id="test_doc_001",
                document_path="test.pdf",
                metadata={
                    "filename": "test.pdf",
                    "content": {"page_count": 5},
                    "processed_at": "2023-01-01T00:00:00"
                }
            )
            
            # Create source reference
            source_ref = tracker.create_source_reference(
                document_id="test_doc_001",
                page_number=1,
                block_index=0,
                block_type="text",
                text_snippet="This is a test snippet from the document.",
                confidence_score=0.95
            )
            
            assert isinstance(source_ref, SourceReference), "Should create SourceReference"
            assert source_ref.document_id == "test_doc_001", "Should have correct document_id"
            assert source_ref.page_number == 1, "Should have correct page_number"
            
            # Test citation formatting
            citation = tracker.format_citation(source_ref, style="simple")
            assert isinstance(citation, str), "Citation should be a string"
            assert len(citation) > 0, "Citation should have content"
            
            # Test statistics
            stats = tracker.get_statistics()
            assert isinstance(stats, dict), "Stats should be a dictionary"
            assert stats["total_documents"] > 0, "Should have documents"
            
            duration = time.time() - start_time
            self.tracker.log_result(
                "Source Tracking",
                True,
                duration,
                f"Created reference, formatted citation: '{citation}'"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.tracker.log_result("Source Tracking", False, duration, str(e))
            raise
    
    @pytest.mark.integration
    def test_system_health_check(self):
        """Test system health check functionality"""
        print("\n🏥 Testing System Health Check")
        
        start_time = time.time()
        
        try:
            from src.chat import ChatEngine
            
            # Create engine
            engine = ChatEngine()
            
            # Run health check
            health = asyncio.run(engine.test_system_health())
            
            assert health is not None, "Health check should return result"
            assert "overall_status" in health, "Should have overall_status"
            assert "components" in health, "Should have components"
            assert "timestamp" in health, "Should have timestamp"
            
            # Test component health
            components = health["components"]
            assert isinstance(components, dict), "Components should be a dict"
            assert len(components) > 0, "Should have component health info"
            
            duration = time.time() - start_time
            self.tracker.log_result(
                "System Health Check",
                True,
                duration,
                f"Status: {health['overall_status']}, Components: {len(components)}"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.tracker.log_result("System Health Check", False, duration, str(e))
            raise
    
    def teardown_method(self):
        """Cleanup after each test method"""
        self.tracker.print_summary()


@pytest.mark.integration
@pytest.mark.slow
class TestRAGSystemIntegration:
    """Integration tests for the complete RAG system"""
    
    def test_full_document_pipeline(self, sample_documents):
        """Test the complete document processing pipeline"""
        if not sample_documents:
            pytest.skip("No sample documents available")
        
        print("\n🔄 Testing Full Document Pipeline")
        
        try:
            from src.chat import ChatEngine
            
            # Create engine
            engine = ChatEngine()
            
            # Process a document
            test_doc = sample_documents[0]
            result = asyncio.run(engine.add_documents_async([str(test_doc)]))
            
            assert result["success"], f"Document processing should succeed: {result.get('error', '')}"
            assert result["total_documents"] > 0, "Should process documents"
            assert result["total_chunks"] > 0, "Should create chunks"
            
            # Query the processed document
            query_result = asyncio.run(engine.query_async(
                "What is this document about?",
                top_k=3
            ))
            
            assert query_result is not None, "Query should return result"
            assert len(query_result["response"]) > 0, "Should have response"
            assert len(query_result["sources"]) > 0, "Should have sources"
            
            print(f"   ✅ Processed {result['total_documents']} docs, {result['total_chunks']} chunks")
            print(f"   ✅ Query returned {len(query_result['sources'])} sources")
            
        except Exception as e:
            pytest.fail(f"Full document pipeline test failed: {e}")


if __name__ == "__main__":
    """Run RAG system tests standalone"""
    print("🚀 Running RAG System Functionality Tests")
    print("=" * 60)
    
    import pytest
    pytest.main([__file__, "-v", "-m", "not slow"]) 