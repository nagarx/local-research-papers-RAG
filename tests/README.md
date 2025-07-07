# ArXiv RAG System - Test Suite

This directory contains comprehensive tests for the ArXiv RAG system.

## Test Organization

### Test Files
- `test_imports.py` - Import and basic functionality tests
- `test_rag_system.py` - Core RAG functionality tests  
- `test_streamlit.py` - Streamlit UI tests
- `conftest.py` - Shared test configuration and fixtures
- `run_tests.py` - Test runner script

### Test Categories

**Unit Tests** (`@pytest.mark.unit`)
- Fast, isolated tests
- Test individual components
- No external dependencies

**Integration Tests** (`@pytest.mark.integration`)
- Test multiple components together
- May require external services (Ollama)
- Slower execution

**Slow Tests** (`@pytest.mark.slow`)
- Tests that take several seconds
- Document processing tests
- End-to-end workflows

## Running Tests

### Using the Test Runner (Recommended)

```bash
# Run all tests
python tests/run_tests.py all

# Run only fast tests
python tests/run_tests.py fast

# Run only import tests
python tests/run_tests.py imports

# Run RAG system tests
python tests/run_tests.py rag

# Run UI tests
python tests/run_tests.py ui

# Run unit tests only
python tests/run_tests.py unit

# Run integration tests only
python tests/run_tests.py integration

# Run slow tests only
python tests/run_tests.py slow
```

### Using pytest directly

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_imports.py

# Run tests with specific markers
pytest -m unit
pytest -m integration
pytest -m "not slow"

# Run with verbose output
pytest -v tests/

# Run specific test
pytest tests/test_imports.py::TestImports::test_core_module_imports
```

## Test Requirements

### Prerequisites
- All dependencies installed: `pip install -r requirements.txt`
- Ollama server running (for integration tests)
- Sample documents in `sample_documents/` directory

### Optional Dependencies
- `pytest-timeout` - For test timeouts
- `pytest-asyncio` - For async test support
- `pytest-mock` - For mocking

Install with:
```bash
pip install pytest pytest-timeout pytest-asyncio pytest-mock
```

## Test Structure

### Import Tests
- Verify all modules can be imported
- Test basic component creation
- Validate configuration loading
- Check directory structure

### RAG System Tests
- Embedding generation
- Vector storage operations
- Document processing with Marker
- Ollama integration
- End-to-end query processing
- Source tracking and citations
- System health checks

### UI Tests
- Streamlit app imports
- Component structure validation
- Configuration testing
- Error handling
- Integration with RAG system

## Configuration

Tests use the same configuration as the main application but with some test-specific settings:

- Tests run with warnings disabled by default
- Timeout set to 300 seconds for slow tests
- Fixtures provide test data and mock objects
- Automatic cleanup after tests

## Troubleshooting

### Common Issues

**Import Errors**
- Ensure you're in the project root directory
- Check that `src/` directory exists
- Verify dependencies are installed

**Ollama Connection Failures**
- Start Ollama server: `ollama serve`
- Pull required model: `ollama pull llama3.1:8b`
- Integration tests will skip if Ollama is unavailable

**Document Processing Failures**
- Ensure sample documents exist in `sample_documents/`
- Check that Marker dependencies are installed
- Some tests will skip if no documents are available

**Memory Issues**
- Run tests with `--disable-warnings` to reduce output
- Use `python tests/run_tests.py fast` to skip slow tests
- Close other applications to free memory

### Getting Help

If tests are failing:
1. Run import tests first: `python tests/run_tests.py imports`
2. Check the specific error messages
3. Verify your environment matches requirements
4. Try running individual test files to isolate issues

## Adding New Tests

When adding new functionality:
1. Add unit tests to `test_rag_system.py`
2. Add integration tests if components interact
3. Use appropriate markers (`@pytest.mark.unit`, etc.)
4. Add fixtures to `conftest.py` if needed
5. Update this README if new test categories are added 