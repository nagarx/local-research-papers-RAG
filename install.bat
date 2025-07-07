@echo off
setlocal enabledelayedexpansion

REM ArXiv Paper RAG Assistant - Automated Installer (Windows)
REM This script sets up the complete environment for the RAG system

:: Color codes for output (using PowerShell for colors)
set "GREEN=[32m"
set "YELLOW=[33m"
set "RED=[31m"
set "BLUE=[34m"
set "NC=[0m"

:: Function to print colored output
call :print_header

:: Main installation function
call :main
goto :end

:print_header
echo.
echo =================================================================
echo 🤖 ArXiv Paper RAG Assistant - Automated Installer (Windows)
echo =================================================================
echo.
goto :eof

:print_status
echo [92m✅ %~1[0m
goto :eof

:print_warning
echo [93m⚠️  %~1[0m
goto :eof

:print_error
echo [91m❌ %~1[0m
goto :eof

:print_info
echo [94mℹ️  %~1[0m
goto :eof

:check_admin
net session >nul 2>&1
if !errorlevel! == 0 (
    call :print_warning "Running as administrator"
    echo This installer can run without administrator privileges.
    echo Press any key to continue or Ctrl+C to exit...
    pause >nul
)
goto :eof

:check_python
call :print_info "Checking Python installation..."

python --version >nul 2>&1
if !errorlevel! == 0 (
    for /f "tokens=2 delims= " %%a in ('python --version 2^>^&1') do set PYTHON_VERSION=%%a
    call :print_status "Python !PYTHON_VERSION! is installed"
    
    REM Check if version is 3.10+
    python -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
    if !errorlevel! == 0 (
        call :print_status "Python version meets requirements (3.10+)"
        goto :eof
    ) else (
        call :print_error "Python !PYTHON_VERSION! is installed but version 3.10+ is required"
        goto :install_python
    )
) else (
    call :print_warning "Python not found"
    goto :install_python
)
goto :eof

:install_python
call :print_info "Please install Python 3.10+ from python.org"
call :print_info "Opening Python download page..."
start https://www.python.org/downloads/
echo.
echo Press any key after installing Python to continue...
pause >nul

REM Check again after installation
call :check_python
goto :eof

:check_ollama
call :print_info "Checking Ollama installation..."

ollama list >nul 2>&1
if !errorlevel! == 0 (
    call :print_status "Ollama is installed and running"
    goto :eof
) else (
    call :print_warning "Ollama not found"
    goto :install_ollama
)
goto :eof

:install_ollama
call :print_info "Installing Ollama..."
call :print_info "Opening Ollama download page..."
start https://ollama.ai/download

echo.
echo Please download and install Ollama from the opened page.
echo Press any key after installing Ollama to continue...
pause >nul

REM Wait for Ollama to be ready
call :print_info "Waiting for Ollama to be ready..."
set /a counter=0
:ollama_wait_loop
if !counter! geq 30 (
    call :print_warning "Ollama setup timeout - continuing anyway"
    goto :eof
)
ollama list >nul 2>&1
if !errorlevel! == 0 (
    call :print_status "Ollama is ready"
    goto :eof
)
timeout /t 2 >nul
set /a counter+=1
goto :ollama_wait_loop

:create_venv
call :print_info "Creating Python virtual environment..."

if exist "venv" (
    call :print_warning "Virtual environment already exists, removing old one..."
    rmdir /s /q venv
)

python -m venv venv
if !errorlevel! neq 0 (
    call :print_error "Failed to create virtual environment"
    goto :error_exit
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip setuptools wheel
if !errorlevel! neq 0 (
    call :print_warning "Failed to upgrade pip"
) else (
    call :print_status "Virtual environment created and activated"
)
goto :eof

:install_dependencies
call :print_info "Installing Python dependencies..."

if not exist "requirements.txt" (
    call :print_error "requirements.txt not found!"
    goto :error_exit
)

REM Install dependencies with retry logic
set /a attempt=1
:install_retry
call :print_info "Installing dependencies (attempt !attempt!/3)..."
pip install -r requirements.txt
if !errorlevel! == 0 (
    call :print_status "Dependencies installed successfully"
    goto :eof
)

set /a attempt+=1
if !attempt! leq 3 (
    call :print_warning "Attempt failed, retrying..."
    timeout /t 5 >nul
    goto :install_retry
)

call :print_error "Failed to install dependencies after 3 attempts"
goto :error_exit

:setup_environment
call :print_info "Setting up environment..."

REM Create necessary directories
mkdir data\documents 2>nul
mkdir data\processed 2>nul
mkdir data\embeddings 2>nul
mkdir data\index 2>nul
mkdir data\cache 2>nul
mkdir data\logs 2>nul
mkdir temp_uploads 2>nul

REM Create .env file if it doesn't exist
if not exist ".env" (
    if exist "env_example.txt" (
        copy env_example.txt .env >nul
        call :print_status "Created .env configuration file"
    )
)

call :print_status "Environment setup complete"
goto :eof

:download_model
call :print_info "Downloading Ollama model..."

REM Check if model already exists
ollama list | findstr "llama3.2:latest" >nul
if !errorlevel! == 0 (
    call :print_status "Ollama model already downloaded"
    goto :eof
)

REM Download model
call :print_info "Downloading model (this may take a while)..."
ollama pull llama3.2:latest
if !errorlevel! == 0 (
    call :print_status "Ollama model downloaded successfully"
) else (
    call :print_warning "Model download failed"
    call :print_info "You can download it later with: ollama pull llama3.2:latest"
)
goto :eof

:run_tests
call :print_info "Running system tests..."

if exist "tests\test_imports.py" (
    python -m pytest tests\test_imports.py -v
    if !errorlevel! == 0 (
        call :print_status "Import tests passed"
    )
)

if exist "run_tests.py" (
    python run_tests.py
    if !errorlevel! == 0 (
        call :print_status "System tests completed"
    )
)
goto :eof

:create_launcher
call :print_info "Creating launcher script..."

REM Create start_rag.bat
(
echo @echo off
echo cd /d "%%~dp0"
echo call venv\Scripts\activate.bat
echo python run.py
echo pause
) > start_rag.bat

call :print_status "Launcher script created: start_rag.bat"
goto :eof

:main
REM Check if running as admin
call :check_admin

REM Check Python
call :check_python

REM Check Ollama
call :check_ollama

REM Create virtual environment
call :create_venv

REM Install dependencies
call :install_dependencies

REM Setup environment
call :setup_environment

REM Download model
call :download_model

REM Run tests
call :run_tests

REM Create launcher
call :create_launcher

echo.
echo =================================================================
call :print_status "Installation completed successfully!"
echo =================================================================
echo.
call :print_info "To start the application:"
echo   1. Double-click: start_rag.bat
echo   2. Or run: venv\Scripts\activate.bat && python run.py
echo   3. Open your browser to: http://localhost:8501
echo.
call :print_info "For troubleshooting, check the logs in data\logs\"
echo.
echo Press any key to exit...
pause >nul
goto :eof

:error_exit
call :print_error "Installation failed!"
echo.
echo Press any key to exit...
pause >nul
exit /b 1

:end
endlocal 