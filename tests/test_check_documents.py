"""
Test script for check_documents functionality
"""

import pytest
from src.utils import check_documents_main
from src.utils.check_documents import main as direct_main, check_specific_document
from src.utils.document_status import DocumentStatusChecker


def test_check_documents_import():
    """Test that check_documents can be imported correctly"""
    # Should not raise any import errors
    assert check_documents_main is not None
    assert direct_main is not None
    assert check_specific_document is not None


def test_document_status_checker_creation():
    """Test that DocumentStatusChecker can be created"""
    checker = DocumentStatusChecker()
    assert checker is not None


def test_check_documents_main_callable():
    """Test that check_documents_main is callable"""
    # This should not raise an exception
    assert callable(check_documents_main)
    assert callable(direct_main)
    assert callable(check_specific_document)


def test_document_status_methods():
    """Test that DocumentStatusChecker has expected methods"""
    checker = DocumentStatusChecker()
    
    # Check that expected methods exist
    assert hasattr(checker, 'get_processed_documents')
    assert hasattr(checker, 'get_indexed_documents')
    assert hasattr(checker, 'get_permanent_documents')
    assert hasattr(checker, 'get_all_documents_status')
    assert hasattr(checker, 'print_documents_status')
    assert hasattr(checker, 'get_storage_usage')
    assert hasattr(checker, 'check_document_exists')
    assert hasattr(checker, 'check_document_exists_by_hash')
    assert hasattr(checker, 'get_duplicate_detection_report')


def test_get_all_documents_status_returns_dict():
    """Test that get_all_documents_status returns a dictionary"""
    checker = DocumentStatusChecker()
    status = checker.get_all_documents_status()
    
    assert isinstance(status, dict)
    assert "total_documents" in status
    assert "timestamp" in status
    assert "config_info" in status


def test_get_storage_usage_returns_dict():
    """Test that get_storage_usage returns a dictionary"""
    checker = DocumentStatusChecker()
    usage = checker.get_storage_usage()
    
    assert isinstance(usage, dict)
    # Should have size information or error
    assert "total_size" in usage or "error" in usage


def test_config_integration():
    """Test that the config system is properly integrated"""
    checker = DocumentStatusChecker()
    
    # Check that config paths are being used
    assert hasattr(checker, 'config')
    assert hasattr(checker, 'processed_dir')
    assert hasattr(checker, 'chroma_dir')
    assert hasattr(checker, 'embeddings_dir')
    
    # Check that paths are Path objects or strings
    assert checker.processed_dir is not None
    assert checker.chroma_dir is not None
    assert checker.embeddings_dir is not None


def test_duplicate_detection_report():
    """Test that duplicate detection report returns proper structure"""
    checker = DocumentStatusChecker()
    report = checker.get_duplicate_detection_report()
    
    assert isinstance(report, dict)
    
    if "error" not in report:
        assert "total_documents" in report
        assert "documents_with_hash" in report
        assert "unique_hashes" in report
        assert "duplicate_groups" in report
        assert "duplicate_documents" in report
        assert "duplicates" in report
        assert "timestamp" in report


def test_check_specific_document_functions():
    """Test specific document check functions"""
    checker = DocumentStatusChecker()
    
    # Test non-existent document by filename
    result = checker.check_document_exists("non_existent_file.pdf")
    assert result is None
    
    # Test non-existent document by hash
    result = checker.check_document_exists_by_hash("fake_hash_123")
    assert result is None


def test_check_specific_document_function():
    """Test the check_specific_document function"""
    # Test with no arguments
    result = check_specific_document()
    assert result is None
    
    # Test with non-existent filename
    result = check_specific_document(filename="non_existent.pdf")
    assert result is None
    
    # Test with non-existent hash
    result = check_specific_document(content_hash="fake_hash")
    assert result is None


def test_enhanced_status_report():
    """Test that the enhanced status report includes new fields"""
    checker = DocumentStatusChecker()
    status = checker.get_all_documents_status()
    
    # Check for new config_info section
    assert "config_info" in status
    config_info = status["config_info"]
    assert "processed_dir" in config_info
    assert "chroma_dir" in config_info
    assert "embeddings_dir" in config_info


if __name__ == "__main__":
    pytest.main([__file__]) 