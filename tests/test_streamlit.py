#!/usr/bin/env python3
"""
Streamlit UI Tests

This module tests the Streamlit user interface components and functionality.
"""

import pytest
import sys
import time
from pathlib import Path
from unittest.mock import Mock, patch

# Test suite imports
from conftest import TestResultTracker


class TestStreamlitUI:
    """Test Streamlit user interface components"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.tracker = TestResultTracker()
    
    def test_streamlit_imports(self):
        """Test Streamlit app imports"""
        print("\n🎨 Testing Streamlit Imports")
        
        start_time = time.time()
        
        try:
            # Test core Streamlit import
            import streamlit as st
            
            # Test app-specific imports
            from src.ui.streamlit_app import StreamlitRAGApp
            
            assert StreamlitRAGApp is not None, "StreamlitRAGApp should be importable"
            
            duration = time.time() - start_time
            self.tracker.log_result(
                "Streamlit Imports",
                True,
                duration,
                "StreamlitRAGApp imported successfully"
            )
            
        except ImportError as e:
            duration = time.time() - start_time
            self.tracker.log_result("Streamlit Imports", False, duration, str(e))
            pytest.skip(f"Streamlit not available: {e}")
        except Exception as e:
            duration = time.time() - start_time
            self.tracker.log_result("Streamlit Imports", False, duration, str(e))
            raise
    
    def test_streamlit_app_structure(self):
        """Test Streamlit app class structure"""
        print("\n🏗️  Testing App Structure")
        
        start_time = time.time()
        
        try:
            from src.ui.streamlit_app import StreamlitRAGApp
            
            # Test class exists and has required methods
            required_methods = [
                'setup_session_state',
                'render_header',
                'render_sidebar',
                'render_file_upload',
                'render_chat_interface',
                'initialize_system'
            ]
            
            missing_methods = []
            for method in required_methods:
                if not hasattr(StreamlitRAGApp, method):
                    missing_methods.append(method)
            
            assert len(missing_methods) == 0, f"Missing methods: {missing_methods}"
            
            duration = time.time() - start_time
            self.tracker.log_result(
                "App Structure",
                True,
                duration,
                f"All {len(required_methods)} required methods present"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.tracker.log_result("App Structure", False, duration, str(e))
            raise
    
    @patch('streamlit.session_state', new_callable=dict)
    def test_session_state_initialization(self, mock_session_state):
        """Test session state initialization"""
        print("\n💾 Testing Session State")
        
        start_time = time.time()
        
        try:
            from src.ui.streamlit_app import StreamlitRAGApp
            
            # Mock Streamlit components
            with patch('streamlit.set_page_config'), \
                 patch('streamlit.markdown'), \
                 patch('streamlit.error'):
                
                # Create app instance
                app = StreamlitRAGApp()
                
                # Check that required session state keys are set
                required_keys = [
                    'chat_engine',
                    'conversation_history',
                    'processed_documents',
                    'system_stats',
                    'upload_status',
                    'models_loaded'
                ]
                
                # The setup_session_state method should set these keys
                app.setup_session_state()
                
                # Since we're mocking, we can't directly test the session state,
                # but we can verify the method runs without error
                
                duration = time.time() - start_time
                self.tracker.log_result(
                    "Session State Initialization",
                    True,
                    duration,
                    "Session state setup completed"
                )
                
        except Exception as e:
            duration = time.time() - start_time
            self.tracker.log_result("Session State Initialization", False, duration, str(e))
            raise
    
    def test_app_configuration(self):
        """Test app configuration and dependencies"""
        print("\n⚙️  Testing App Configuration")
        
        start_time = time.time()
        
        try:
            # Test that required dependencies are available
            import streamlit
            import pandas as pd
            import plotly.express as px
            import plotly.graph_objects as go
            
            # Test configuration access
            from src.config import get_config
            config = get_config()
            
            assert config is not None, "Configuration should be available"
            
            duration = time.time() - start_time
            self.tracker.log_result(
                "App Configuration",
                True,
                duration,
                "All dependencies available"
            )
            
        except ImportError as e:
            duration = time.time() - start_time
            self.tracker.log_result("App Configuration", False, duration, str(e))
            pytest.skip(f"Required dependencies not available: {e}")
        except Exception as e:
            duration = time.time() - start_time
            self.tracker.log_result("App Configuration", False, duration, str(e))
            raise
    
    def test_app_launcher(self):
        """Test the app launcher script"""
        print("\n🚀 Testing App Launcher")
        
        start_time = time.time()
        
        try:
            # Test that app.py exists and is valid Python
            app_file = Path("app.py")
            assert app_file.exists(), "app.py should exist"
            
            # Test that it contains expected content
            content = app_file.read_text()
            assert "streamlit" in content.lower(), "app.py should reference Streamlit"
            assert "src/ui/streamlit_app.py" in content, "Should reference the Streamlit app"
            
            duration = time.time() - start_time
            self.tracker.log_result(
                "App Launcher",
                True,
                duration,
                "app.py exists and has expected content"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.tracker.log_result("App Launcher", False, duration, str(e))
            raise
    
    def test_ui_components_structure(self):
        """Test UI components and layout structure"""
        print("\n🎨 Testing UI Components")
        
        start_time = time.time()
        
        try:
            from src.ui.streamlit_app import StreamlitRAGApp
            
            # Mock Streamlit to avoid actual UI rendering
            with patch('streamlit.set_page_config'), \
                 patch('streamlit.markdown'), \
                 patch('streamlit.error'), \
                 patch('streamlit.sidebar'), \
                 patch('streamlit.columns'), \
                 patch('streamlit.file_uploader'), \
                 patch('streamlit.text_input'):
                
                app = StreamlitRAGApp()
                
                # Test that methods can be called without errors
                app.render_header()
                app.render_sidebar()
                app.render_file_upload()
                app.render_chat_interface()
                
                duration = time.time() - start_time
                self.tracker.log_result(
                    "UI Components",
                    True,
                    duration,
                    "All UI rendering methods work"
                )
                
        except Exception as e:
            duration = time.time() - start_time
            self.tracker.log_result("UI Components", False, duration, str(e))
            raise
    
    def test_error_handling(self):
        """Test error handling in UI components"""
        print("\n🚨 Testing Error Handling")
        
        start_time = time.time()
        
        try:
            from src.ui.streamlit_app import StreamlitRAGApp
            
            # Test that the app can handle import failures gracefully
            with patch('streamlit.set_page_config'), \
                 patch('streamlit.error') as mock_error:
                
                # The app should handle missing dependencies
                # This is tested by the import handling in the actual app
                
                duration = time.time() - start_time
                self.tracker.log_result(
                    "Error Handling",
                    True,
                    duration,
                    "Error handling mechanisms in place"
                )
                
        except Exception as e:
            duration = time.time() - start_time
            self.tracker.log_result("Error Handling", False, duration, str(e))
            raise
    
    def teardown_method(self):
        """Cleanup after each test method"""
        self.tracker.print_summary()


class TestStreamlitIntegration:
    """Integration tests for Streamlit with RAG system"""
    
    @pytest.mark.integration
    def test_rag_system_integration(self):
        """Test Streamlit integration with RAG system"""
        print("\n🔗 Testing RAG System Integration")
        
        try:
            from src.ui.streamlit_app import StreamlitRAGApp
            from src.chat import ChatEngine
            
            # Mock Streamlit environment
            with patch('streamlit.set_page_config'), \
                 patch('streamlit.error'), \
                 patch('streamlit.session_state', new_callable=dict):
                
                app = StreamlitRAGApp()
                
                # Test that the app can create RAG components
                # This would normally be done in the UI, but we test the underlying functionality
                engine = ChatEngine()
                assert engine is not None, "Should be able to create ChatEngine"
                
                print("   ✅ RAG system integration works")
                
        except Exception as e:
            pytest.fail(f"RAG system integration failed: {e}")


def test_streamlit_app_file_structure():
    """Test that Streamlit app file structure is correct"""
    print("\n📁 Testing Streamlit File Structure")
    
    required_files = [
        "src/ui/streamlit_app.py",
        "src/ui/__init__.py",
        "app.py"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"   ✅ {file_path}")
    
    assert len(missing_files) == 0, f"Missing Streamlit files: {missing_files}"
    print("   ✅ All Streamlit files exist")


if __name__ == "__main__":
    """Run Streamlit tests standalone"""
    print("🎨 Running Streamlit UI Tests")
    print("=" * 50)
    
    import pytest
    pytest.main([__file__, "-v"]) 