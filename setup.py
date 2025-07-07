"""
Setup script for ArXiv Paper RAG Assistant
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements_path = this_directory / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, 'r', encoding='utf-8') as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith('#')
        ]

setup(
    name="arxiv-paper-rag",
    version="1.0.0",
    author="ArXiv RAG Assistant Team",
    author_email="contact@arxivrag.com",
    description="An intelligent RAG system for academic research papers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/arxiv-paper-rag",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Indexing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "arxiv-rag=src.ui.streamlit_app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "arxiv_rag": [
            "ui/static/*",
            "config/*.yaml",
            "config/*.json",
        ],
    },
    zip_safe=False,
    keywords="rag, pdf, academic, research, llm, ollama, marker, streamlit",
    project_urls={
        "Bug Reports": "https://github.com/your-org/arxiv-paper-rag/issues",
        "Documentation": "https://github.com/your-org/arxiv-paper-rag/docs",
        "Source": "https://github.com/your-org/arxiv-paper-rag",
    },
) 