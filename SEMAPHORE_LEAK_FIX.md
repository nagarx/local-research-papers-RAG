# Semaphore Leak Fix - Comprehensive Solution

## Problem Description

The system was experiencing semaphore leaks when processing documents with the Marker tool, resulting in warnings like:

```
/opt/anaconda3/lib/python3.12/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
```

## Root Cause Analysis

1. **Marker Tool Multiprocessing**: The Marker tool was creating multiprocessing resources internally
2. **ThreadPoolExecutor**: Using ThreadPoolExecutor was creating additional threading/multiprocessing contexts
3. **Insufficient Worker Limits**: Worker configuration wasn't comprehensive enough
4. **Resource Cleanup**: Existing cleanup wasn't aggressive enough for semaphore-specific resources

## Comprehensive Solution Implemented

### 1. Environment Variable Configuration

**File**: `src/ingestion/document_processor.py`

Added comprehensive environment variables to force single-threaded operation:

```python
# COMPREHENSIVE environment variables to force single-threaded operation
os.environ['MARKER_MAX_WORKERS'] = '1'
os.environ['MARKER_PARALLEL_FACTOR'] = '1'
os.environ['MARKER_WORKERS'] = '1'
os.environ['MARKER_NUM_WORKERS'] = '1'
os.environ['MARKER_DISABLE_MULTIPROCESSING'] = '1'

# PyTorch threading control
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Transformers/HuggingFace threading control
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# General Python threading control
os.environ['PYTHONHASHSEED'] = '0'

# Disable multiprocessing in common libraries
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Force single GPU if available
```

### 2. Enhanced Marker Configuration

**File**: `src/ingestion/document_processor.py` - `_setup_converter()` method

Added aggressive single-worker configuration:

```python
config = {
    # ... existing config ...
    
    # CRITICAL: ABSOLUTELY FORCE single-worker operation
    "workers": 1,
    "max_workers": 1,
    "parallel_factor": 1,
    "batch_size": 1,
    "num_workers": 1,
    "disable_multiprocessing": True,
    "single_threaded": True,
    "sequential": True,
    
    # Additional safety settings
    "max_parallel": 1,
    "cpu_count": 1,
    "pool_size": 1,
}
```

### 3. Eliminated ThreadPoolExecutor

**File**: `src/ingestion/document_processor.py` - `process_document_async()` method

**Before:**
```python
with ThreadPoolExecutor(max_workers=1) as executor:
    rendered = await loop.run_in_executor(executor, self._process_with_marker_sync, str(file_path))
```

**After:**
```python
# Direct synchronous processing to avoid any threading/multiprocessing issues
rendered = self._process_with_marker_sync(str(file_path))
```

### 4. Enhanced Resource Cleanup

**File**: `src/utils/resource_cleanup.py`

Added comprehensive multiprocessing resource cleanup:

```python
def cleanup_multiprocessing_resources():
    """Clean up multiprocessing resources to prevent semaphore leaks"""
    # - Force garbage collection
    # - Clean up multiprocessing contexts and semaphores
    # - Platform-specific cleanup (macOS semaphore handling)
    # - Thread cleanup

def cleanup_semaphore_leaks():
    """Specifically target semaphore leak cleanup"""
    # - Access multiprocessing.semaphore_tracker safely
    # - Clean up semaphore registry
    # - Remove thread-local semaphore references
```

### 5. Strategic Cleanup Points

**Added cleanup calls at multiple strategic points:**

1. **Before model loading** (global models)
2. **Before document processing**
3. **After successful processing**
4. **On processing errors**
5. **Between batch documents**
6. **Between processing batches**

### 6. Global Model Loading Cleanup

**File**: `src/ingestion/document_processor.py` - `get_global_marker_models()` function

Added cleanup before and after model loading to prevent initial semaphore leaks.

## Testing and Validation

### How to Test the Fix

1. **Monitor semaphore warnings**: Run document processing and check for warnings
2. **Process multiple documents**: Test batch processing to ensure no accumulation
3. **Check system resources**: Monitor semaphore usage with system tools

### Expected Results

- **No semaphore leak warnings** during document processing
- **Consistent performance** across multiple document processing sessions
- **Clean shutdown** without resource tracker warnings

## Usage Notes

### Performance Impact

- **Slightly slower processing**: Single-threaded operation may be slower than parallel
- **More reliable**: Eliminates resource leaks and improves long-term stability
- **Memory efficient**: Better memory management and cleanup

### Configuration Verification

Check that Marker is using single-worker configuration:

```python
# The log should show:
"Marker converter initialized with strict single-worker configuration"
```

### Monitoring

Watch for these log messages indicating proper cleanup:

```
"Starting direct Marker processing for: [filename]"
"Marker processing completed for: [filename]"
```

## Troubleshooting

### If semaphore leaks still occur:

1. **Check environment variables**: Ensure all threading env vars are set
2. **Verify Marker version**: Some versions might not respect all config options
3. **Platform-specific issues**: macOS and Linux handle semaphores differently
4. **System limits**: Check system semaphore limits with `ipcs -s`

### Emergency cleanup:

If semaphores accumulate, restart the Python process to clear all resources.

## Technical Details

### Why This Solution Works

1. **Single-threaded execution**: Eliminates multiprocessing altogether
2. **Comprehensive env vars**: Controls all threading libraries
3. **Aggressive cleanup**: Multiple cleanup points ensure no resource accumulation
4. **Direct execution**: Removes ThreadPoolExecutor layer that was creating additional contexts

### Platform Considerations

- **macOS**: Special handling for named semaphore cleanup
- **Linux**: Standard multiprocessing cleanup sufficient
- **Windows**: Environment variables handle most threading libraries

## Monitoring and Maintenance

### Regular Checks

1. Monitor log files for semaphore warnings
2. Check system resource usage during long processing sessions
3. Verify cleanup logs appear after each document

### Performance Monitoring

Track processing times to ensure single-threaded operation doesn't significantly impact performance.

## Version History

- **v1.0**: Initial semaphore leak fix implementation
- **Date**: January 2025
- **Status**: Active, monitoring for effectiveness