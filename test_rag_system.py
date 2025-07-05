#!/usr/bin/env python3
"""
Comprehensive RAG System Test Suite

This test suite validates the complete RAG system functionality including:
- Document processing with Marker
- Embedding generation
- Vector storage and retrieval
- Chat functionality with Ollama
- End-to-end query processing
"""

import os
import sys
import asyncio
import tempfile
import traceback
from pathlib import Path
from typing import Dict, List, Any

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from chat import ChatEngine
    from config import get_config, get_logger
    from ingestion import get_global_marker_models
    from utils import PerformanceUtils, LoggerUtils, clear_gpu_cache
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure you're in the correct directory and dependencies are installed")
    sys.exit(1)


class RAGSystemTester:
    """Comprehensive RAG system tester"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.config = get_config()
        self.test_results = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "test_details": []
        }
    
    def log_test_result(self, test_name: str, success: bool, duration: float, details: str = ""):
        """Log test result"""
        self.test_results["total_tests"] += 1
        
        if success:
            self.test_results["passed_tests"] += 1
            status = "✅ PASSED"
        else:
            self.test_results["failed_tests"] += 1
            status = "❌ FAILED"
        
        duration_str = PerformanceUtils.format_duration(duration)
        print(f"{status} {test_name} ({duration_str})")
        
        if details:
            print(f"   {details}")
        
        self.test_results["test_details"].append({
            "test": test_name,
            "success": success,
            "duration": duration,
            "details": details
        })
    
    async def test_marker_integration(self) -> bool:
        """Test Marker integration and global model caching"""
        print("\n🔧 Testing Marker Integration...")
        
        try:
            # Test 1: Global model loading
            start_time = time.time()
            models = get_global_marker_models()
            duration = time.time() - start_time
            
            self.log_test_result(
                "Global Model Loading",
                models is not None,
                duration,
                f"Models loaded: {len(models) if models else 0}"
            )
            
            # Test 2: Model reuse (should be instant)
            start_time = time.time()
            models_2 = get_global_marker_models()
            duration = time.time() - start_time
            
            self.log_test_result(
                "Model Reuse (Caching)",
                models_2 is models and duration < 0.1,
                duration,
                f"Same instance: {models_2 is models}, Duration: {duration:.3f}s"
            )
            
            return True
            
        except Exception as e:
            self.log_test_result("Marker Integration", False, 0, str(e))
            return False
    
    async def test_document_processing(self) -> bool:
        """Test document processing with sample PDFs"""
        print("\n📄 Testing Document Processing...")
        
        try:
            # Initialize chat engine (includes document processor)
            chat_engine = ChatEngine()
            
            # Test sample documents
            sample_dir = Path("sample_documents")
            if not sample_dir.exists():
                self.log_test_result("Document Processing", False, 0, "Sample documents not found")
                return False
            
            pdf_files = list(sample_dir.glob("*.pdf"))
            if not pdf_files:
                self.log_test_result("Document Processing", False, 0, "No PDF files found")
                return False
            
            # Test processing first PDF
            test_pdf = pdf_files[0]
            start_time = time.time()
            
            result = await chat_engine.document_processor.process_document_async(test_pdf)
            duration = time.time() - start_time
            
            success = result is not None and "content" in result
            details = f"File: {test_pdf.name}, Blocks: {len(result.get('content', {}).get('blocks', []))}"
            
            self.log_test_result("Document Processing", success, duration, details)
            
            return success
            
        except Exception as e:
            self.log_test_result("Document Processing", False, 0, str(e))
            return False
    
    async def test_embeddings_and_vector_storage(self) -> bool:
        """Test embedding generation and vector storage"""
        print("\n🔍 Testing Embeddings & Vector Storage...")
        
        try:
            chat_engine = ChatEngine()
            
            # Test embedding generation
            test_texts = [
                "This is a test document about machine learning.",
                "Another document about neural networks and AI.",
                "A third document discussing natural language processing."
            ]
            
            start_time = time.time()
            embeddings = await chat_engine.embedding_manager.embed_texts_batch_async(test_texts)
            duration = time.time() - start_time
            
            success = len(embeddings) == len(test_texts)
            details = f"Generated {len(embeddings)} embeddings"
            
            self.log_test_result("Embedding Generation", success, duration, details)
            
            # Test vector storage
            if success:
                start_time = time.time()
                
                # Create mock document chunks
                chunks = [
                    {
                        "id": f"test_chunk_{i}",
                        "text": text,
                        "chunk_index": i,
                        "source_info": {
                            "document_id": "test_doc",
                            "document_name": "test.pdf",
                            "page_number": 1,
                            "block_index": i
                        }
                    }
                    for i, text in enumerate(test_texts)
                ]
                
                # Add to vector store
                added = chat_engine.vector_store.add_document(
                    document_id="test_doc",
                    chunks=chunks,
                    embeddings=embeddings,
                    metadata={"test": True}
                )
                
                duration = time.time() - start_time
                
                self.log_test_result("Vector Storage", added, duration, f"Added {len(chunks)} chunks")
                
                # Test search
                if added:
                    start_time = time.time()
                    query_embedding = embeddings[0]  # Use first embedding as query
                    results = chat_engine.vector_store.search(query_embedding, top_k=2)
                    duration = time.time() - start_time
                    
                    search_success = len(results) > 0
                    details = f"Found {len(results)} similar chunks"
                    
                    self.log_test_result("Vector Search", search_success, duration, details)
                    
                    return search_success
            
            return success
            
        except Exception as e:
            self.log_test_result("Embeddings & Vector Storage", False, 0, str(e))
            return False
    
    async def test_ollama_integration(self) -> bool:
        """Test Ollama client integration"""
        print("\n🤖 Testing Ollama Integration...")
        
        try:
            chat_engine = ChatEngine()
            
            # Test connection
            start_time = time.time()
            connection_ok = chat_engine.ollama_client.test_connection()
            duration = time.time() - start_time
            
            self.log_test_result("Ollama Connection", connection_ok, duration)
            
            if connection_ok:
                # Test simple query
                start_time = time.time()
                response = await chat_engine.ollama_client.generate_response_async(
                    "What is machine learning?",
                    context_chunks=[]
                )
                duration = time.time() - start_time
                
                success = response is not None and len(response) > 0
                details = f"Response length: {len(response) if response else 0} chars"
                
                self.log_test_result("Ollama Response", success, duration, details)
                
                return success
            
            return connection_ok
            
        except Exception as e:
            self.log_test_result("Ollama Integration", False, 0, str(e))
            return False
    
    async def test_end_to_end_rag(self) -> bool:
        """Test complete end-to-end RAG pipeline"""
        print("\n🔄 Testing End-to-End RAG Pipeline...")
        
        try:
            chat_engine = ChatEngine()
            
            # Test complete RAG query
            start_time = time.time()
            response = await chat_engine.process_query_async(
                "What is the main contribution of the paper?",
                max_chunks=3
            )
            duration = time.time() - start_time
            
            success = response is not None and "answer" in response
            details = f"Response: {response.get('answer', 'No answer')[:100]}..."
            
            self.log_test_result("End-to-End RAG", success, duration, details)
            
            return success
            
        except Exception as e:
            self.log_test_result("End-to-End RAG", False, 0, str(e))
            return False
    
    async def test_system_health(self) -> bool:
        """Test overall system health"""
        print("\n🏥 Testing System Health...")
        
        try:
            chat_engine = ChatEngine()
            
            start_time = time.time()
            health = await chat_engine.test_system_health()
            duration = time.time() - start_time
            
            overall_healthy = health.get("overall_status") == "healthy"
            component_count = len(health.get("components", {}))
            
            self.log_test_result(
                "System Health Check",
                overall_healthy,
                duration,
                f"Components checked: {component_count}"
            )
            
            return overall_healthy
            
        except Exception as e:
            self.log_test_result("System Health", False, 0, str(e))
            return False
    
    def print_final_summary(self):
        """Print final test summary"""
        print("\n" + "="*60)
        print("🧪 COMPREHENSIVE RAG SYSTEM TEST SUMMARY")
        print("="*60)
        
        total = self.test_results["total_tests"]
        passed = self.test_results["passed_tests"]
        failed = self.test_results["failed_tests"]
        
        print(f"📊 Total Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"📈 Success Rate: {(passed/total)*100:.1f}%" if total > 0 else "N/A")
        
        if failed > 0:
            print("\n❌ Failed Tests:")
            for test in self.test_results["test_details"]:
                if not test["success"]:
                    print(f"   • {test['test']}: {test['details']}")
        
        print("\n" + "="*60)
        
        if failed == 0:
            print("🎉 ALL TESTS PASSED! Your RAG system is working perfectly!")
        else:
            print(f"⚠️  {failed} test(s) failed. Please review the issues above.")
        
        print("="*60)


async def main():
    """Run comprehensive RAG system tests"""
    print("🚀 Starting Comprehensive RAG System Tests")
    print("="*60)
    
    # Initialize tester
    tester = RAGSystemTester()
    
    # Pre-load global models to avoid timing issues
    print("📦 Pre-loading Marker models...")
    get_global_marker_models()
    
    # Run all tests
    tests = [
        tester.test_marker_integration(),
        tester.test_document_processing(),
        tester.test_embeddings_and_vector_storage(),
        tester.test_ollama_integration(),
        tester.test_end_to_end_rag(),
        tester.test_system_health()
    ]
    
    # Execute tests
    start_time = time.time()
    
    for test in tests:
        try:
            await test
        except Exception as e:
            print(f"❌ Test execution error: {e}")
        
        # Clear GPU cache between tests
        clear_gpu_cache()
        
        # Small delay between tests
        await asyncio.sleep(0.5)
    
    total_duration = time.time() - start_time
    
    # Print final summary
    tester.print_final_summary()
    
    print(f"\n⏱️  Total test execution time: {PerformanceUtils.format_duration(total_duration)}")


if __name__ == "__main__":
    asyncio.run(main()) 