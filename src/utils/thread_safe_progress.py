"""
Thread-Safe Progress Tracking for Streamlit

This module provides a thread-safe way to track progress in background threads
without calling Streamlit components directly from those threads.
"""

import threading
import time
from typing import Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class ProgressState:
    """Thread-safe progress state"""
    current_step: int = 0
    total_steps: int = 0
    current_message: str = ""
    start_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    is_complete: bool = False
    error_message: Optional[str] = None
    
    def get_progress_ratio(self) -> float:
        """Get progress as a ratio (0.0 to 1.0)"""
        if self.total_steps <= 0:
            return 0.0
        return min(self.current_step / self.total_steps, 1.0)
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds"""
        return time.time() - self.start_time
    
    def get_eta(self) -> Optional[float]:
        """Get estimated time to completion"""
        if self.current_step <= 0:
            return None
        elapsed = self.get_elapsed_time()
        rate = self.current_step / elapsed
        remaining_steps = self.total_steps - self.current_step
        return remaining_steps / rate if rate > 0 else None


class ThreadSafeProgressTracker:
    """
    Thread-safe progress tracker that can be updated from background threads
    and read from the main Streamlit thread.
    """
    
    def __init__(self, operation_name: str = "Processing"):
        self.operation_name = operation_name
        self._lock = threading.Lock()
        self._state = ProgressState()
        self._callbacks = []
    
    def set_total_steps(self, total_steps: int):
        """Set the total number of steps"""
        with self._lock:
            self._state.total_steps = total_steps
            self._state.last_update = time.time()
    
    def update(self, current_step: int, message: str = "", is_complete: bool = False):
        """Update progress from any thread"""
        with self._lock:
            self._state.current_step = current_step
            self._state.current_message = message
            self._state.is_complete = is_complete
            self._state.last_update = time.time()
    
    def set_error(self, error_message: str):
        """Set error state"""
        with self._lock:
            self._state.error_message = error_message
            self._state.is_complete = True
            self._state.last_update = time.time()
    
    def get_state(self) -> ProgressState:
        """Get current progress state (thread-safe)"""
        with self._lock:
            # Return a copy to avoid thread safety issues
            return ProgressState(
                current_step=self._state.current_step,
                total_steps=self._state.total_steps,
                current_message=self._state.current_message,
                start_time=self._state.start_time,
                last_update=self._state.last_update,
                is_complete=self._state.is_complete,
                error_message=self._state.error_message
            )
    
    def is_complete(self) -> bool:
        """Check if operation is complete"""
        with self._lock:
            return self._state.is_complete
    
    def has_error(self) -> bool:
        """Check if operation has error"""
        with self._lock:
            return self._state.error_message is not None
    
    def get_progress_info(self) -> Dict[str, Any]:
        """Get comprehensive progress information"""
        state = self.get_state()
        return {
            "operation_name": self.operation_name,
            "progress_ratio": state.get_progress_ratio(),
            "current_step": state.current_step,
            "total_steps": state.total_steps,
            "current_message": state.current_message,
            "elapsed_time": state.get_elapsed_time(),
            "eta": state.get_eta(),
            "is_complete": state.is_complete,
            "error_message": state.error_message,
            "last_update": state.last_update
        }
    
    def create_callback(self):
        """Create a callback function that can be used in background threads"""
        def callback(message: str, processed_count: Optional[int] = None, total_count: Optional[int] = None):
            try:
                if total_count is not None:
                    self.set_total_steps(total_count)
                
                if processed_count is not None:
                    self.update(processed_count, message)
                else:
                    # If no processed_count, just update the message
                    current_state = self.get_state()
                    self.update(current_state.current_step, message)
            except Exception as e:
                # Don't let callback errors break the main processing
                pass
        
        return callback
    
    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """Wait for operation to complete"""
        start_time = time.time()
        
        while not self.is_complete():
            if timeout and (time.time() - start_time) > timeout:
                return False
            time.sleep(0.1)
        
        return True


class StreamlitProgressRenderer:
    """
    Helper class to render progress in Streamlit using the thread-safe tracker.
    This should only be called from the main Streamlit thread.
    """
    
    def __init__(self, tracker: ThreadSafeProgressTracker):
        self.tracker = tracker
        self.progress_bar = None
        self.status_text = None
        self.container = None
    
    def setup_ui(self, container=None):
        """Setup Streamlit UI components"""
        import streamlit as st
        
        if container:
            self.container = container
        
        with (self.container if self.container else st.container()):
            self.progress_bar = st.progress(0)
            self.status_text = st.empty()
    
    def update_ui(self):
        """Update Streamlit UI components with current progress"""
        if not self.progress_bar or not self.status_text:
            return
        
        try:
            info = self.tracker.get_progress_info()
            
            # Update progress bar
            self.progress_bar.progress(info["progress_ratio"])
            
            # Update status text
            if info["error_message"]:
                self.status_text.error(f"❌ {info['error_message']}")
            elif info["is_complete"]:
                self.status_text.success(f"✅ {info['operation_name']} completed!")
            else:
                elapsed = info["elapsed_time"]
                eta = info["eta"]
                eta_str = f" (ETA: {eta:.1f}s)" if eta else ""
                
                self.status_text.text(
                    f"🔄 {info['current_message']} "
                    f"({info['current_step']}/{info['total_steps']}) "
                    f"[{elapsed:.1f}s{eta_str}]"
                )
        except Exception as e:
            # Don't let UI update errors break anything
            pass
    
    def cleanup(self):
        """Clean up UI components"""
        if self.progress_bar:
            self.progress_bar.empty()
        if self.status_text:
            self.status_text.empty()
        
        self.progress_bar = None
        self.status_text = None 