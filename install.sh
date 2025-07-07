#!/bin/bash

# ArXiv Paper RAG Assistant - Automated Installer (Unix/Linux/macOS)
# This script sets up the complete environment for the RAG system

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    elif [[ "$OSTYPE" == "cygwin" ]]; then
        OS="windows"
    else
        OS="unknown"
    fi
}

# Function to install Python if needed
install_python() {
    print_info "Checking Python installation..."
    
    if command_exists python3; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        REQUIRED_VERSION="3.10"
        
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
            print_status "Python $PYTHON_VERSION is installed and meets requirements"
            return 0
        else
            print_error "Python $PYTHON_VERSION is installed but version 3.10+ is required"
        fi
    else
        print_warning "Python 3 not found"
    fi
    
    print_info "Attempting to install Python 3.10+..."
    
    case $OS in
        "linux")
            if command_exists apt; then
                sudo apt update
                sudo apt install -y python3 python3-pip python3-venv python3-dev
            elif command_exists yum; then
                sudo yum install -y python3 python3-pip python3-venv python3-devel
            elif command_exists dnf; then
                sudo dnf install -y python3 python3-pip python3-venv python3-devel
            else
                print_error "Package manager not found. Please install Python 3.10+ manually"
                exit 1
            fi
            ;;
        "macos")
            if command_exists brew; then
                brew install python@3.11
            else
                print_error "Homebrew not found. Please install Python 3.10+ manually from python.org"
                exit 1
            fi
            ;;
        *)
            print_error "Unsupported OS for automatic Python installation"
            exit 1
            ;;
    esac
}

# Function to install Ollama
install_ollama() {
    print_info "Checking Ollama installation..."
    
    if command_exists ollama; then
        print_status "Ollama is already installed"
        return 0
    fi
    
    print_info "Installing Ollama..."
    
    case $OS in
        "linux"|"macos")
            curl -fsSL https://ollama.ai/install.sh | sh
            ;;
        *)
            print_error "Unsupported OS for automatic Ollama installation"
            print_info "Please install Ollama manually from https://ollama.ai"
            exit 1
            ;;
    esac
    
    # Wait for Ollama to be available
    print_info "Waiting for Ollama to be ready..."
    for i in {1..30}; do
        if ollama list >/dev/null 2>&1; then
            print_status "Ollama is ready"
            break
        fi
        sleep 2
    done
}

# Function to create virtual environment
create_venv() {
    print_info "Creating Python virtual environment..."
    
    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists, removing old one..."
        rm -rf venv
    fi
    
    python3 -m venv venv
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    print_status "Virtual environment created and activated"
}

# Function to install Python dependencies
install_dependencies() {
    print_info "Installing Python dependencies..."
    
    if [ ! -f "requirements.txt" ]; then
        print_error "requirements.txt not found!"
        exit 1
    fi
    
    # Install dependencies with retry logic
    for i in {1..3}; do
        if pip install -r requirements.txt; then
            print_status "Dependencies installed successfully"
            break
        else
            print_warning "Attempt $i failed, retrying..."
            sleep 5
        fi
        
        if [ $i -eq 3 ]; then
            print_error "Failed to install dependencies after 3 attempts"
            exit 1
        fi
    done
}

# Function to setup environment
setup_environment() {
    print_info "Setting up environment..."
    
    # Create necessary directories
    mkdir -p data/{documents,processed,embeddings,index,cache,logs}
    mkdir -p temp_uploads
    
    # Create .env file if it doesn't exist
    if [ ! -f ".env" ] && [ -f "env_example.txt" ]; then
        cp env_example.txt .env
        print_status "Created .env configuration file"
    fi
    
    print_status "Environment setup complete"
}

# Function to download Ollama model
download_model() {
    print_info "Downloading Ollama model..."
    
    # Check if model already exists
    if ollama list | grep -q "llama3.2:latest"; then
        print_status "Ollama model already downloaded"
        return 0
    fi
    
    # Download model with timeout
    timeout 1800 ollama pull llama3.2:latest || {
        print_warning "Model download failed or timed out"
        print_info "You can download it later with: ollama pull llama3.2:latest"
        return 1
    }
    
    print_status "Ollama model downloaded successfully"
}

# Function to run tests
run_tests() {
    print_info "Running system tests..."
    
    if [ -f "tests/test_imports.py" ]; then
        python -m pytest tests/test_imports.py -v
        print_status "Import tests passed"
    fi
    
    if [ -f "run_tests.py" ]; then
        python run_tests.py
        print_status "System tests completed"
    fi
}

# Function to create launcher script
create_launcher() {
    print_info "Creating launcher script..."
    
    cat > start_rag.sh << 'EOF'
#!/bin/bash
# ArXiv Paper RAG Assistant Launcher

cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Check if Ollama is running
if ! pgrep -f "ollama" > /dev/null; then
    echo "Starting Ollama..."
    ollama serve &
    sleep 5
fi

# Launch the application
python run.py
EOF

    chmod +x start_rag.sh
    print_status "Launcher script created: start_rag.sh"
}

# Main installation function
main() {
    echo "================================================================="
    echo "🤖 ArXiv Paper RAG Assistant - Automated Installer"
    echo "================================================================="
    
    # Detect OS
    detect_os
    print_info "Detected OS: $OS"
    
    # Check if running as root (not recommended)
    if [ "$EUID" -eq 0 ]; then
        print_warning "Running as root is not recommended"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    # Install system dependencies
    print_info "Installing system dependencies..."
    
    case $OS in
        "linux")
            # Update package manager
            if command_exists apt; then
                sudo apt update
                sudo apt install -y curl wget git build-essential
            elif command_exists yum; then
                sudo yum install -y curl wget git gcc gcc-c++ make
            elif command_exists dnf; then
                sudo dnf install -y curl wget git gcc gcc-c++ make
            fi
            ;;
        "macos")
            # Install Xcode command line tools if needed
            if ! command_exists git; then
                xcode-select --install
            fi
            ;;
    esac
    
    # Install Python
    install_python
    
    # Install Ollama
    install_ollama
    
    # Create virtual environment
    create_venv
    
    # Install dependencies
    install_dependencies
    
    # Setup environment
    setup_environment
    
    # Download model (non-blocking)
    download_model || print_warning "Model download failed, can be done later"
    
    # Run tests
    run_tests || print_warning "Some tests failed, but installation can continue"
    
    # Create launcher script
    create_launcher
    
    echo "================================================================="
    print_status "Installation completed successfully!"
    echo "================================================================="
    echo
    print_info "To start the application:"
    echo "  1. Run: ./start_rag.sh"
    echo "  2. Or run: source venv/bin/activate && python run.py"
    echo "  3. Open your browser to: http://localhost:8501"
    echo
    print_info "For troubleshooting, check the logs in data/logs/"
}

# Handle script interruption
trap 'echo -e "\n${RED}Installation interrupted${NC}"; exit 1' INT TERM

# Run main function
main "$@" 