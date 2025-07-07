#!/bin/bash
# Cleanup script for removing old Streamlit UI components
# Run this after confirming you want to use only the modern UI

echo "🧹 ArXiv RAG System - Old UI Cleanup Script"
echo "==========================================="
echo ""
echo "This script will remove the old Streamlit UI components."
echo "Make sure you have the modern UI working before proceeding!"
echo ""
read -p "Are you sure you want to remove the old UI? (y/N): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "❌ Cleanup cancelled."
    exit 1
fi

echo ""
echo "📁 Creating backup directory..."
mkdir -p old_ui_backup

echo "🔄 Backing up old UI files..."

# Backup old launchers
cp app.py old_ui_backup/ 2>/dev/null && echo "   ✓ Backed up app.py"
cp run.py old_ui_backup/ 2>/dev/null && echo "   ✓ Backed up run.py"

# Backup Streamlit UI
cp -r src/ui/streamlit_app.py old_ui_backup/ 2>/dev/null && echo "   ✓ Backed up streamlit_app.py"

# Backup migration script
cp migrate_to_chromadb.py old_ui_backup/ 2>/dev/null && echo "   ✓ Backed up migrate_to_chromadb.py"

# Backup tests
cp tests/test_streamlit.py old_ui_backup/ 2>/dev/null && echo "   ✓ Backed up test_streamlit.py"

echo ""
echo "🗑️  Removing old UI files..."

# Remove old launchers
rm -f app.py && echo "   ✓ Removed app.py"
rm -f run.py && echo "   ✓ Removed run.py"

# Remove Streamlit UI
rm -f src/ui/streamlit_app.py && echo "   ✓ Removed streamlit_app.py"

# Remove migration script
rm -f migrate_to_chromadb.py && echo "   ✓ Removed migrate_to_chromadb.py"

# Remove old tests
rm -f tests/test_streamlit.py && echo "   ✓ Removed test_streamlit.py"

echo ""
echo "📝 Updating requirements.txt..."

# Create a new requirements.txt without Streamlit dependencies
grep -v "streamlit\|plotly" requirements.txt > requirements_new.txt
mv requirements_new.txt requirements.txt
echo "   ✓ Removed Streamlit and Plotly from requirements.txt"

echo ""
echo "📝 Updating setup.py..."

# Remove console script entry
if [ -f setup.py ]; then
    sed -i.bak '/arxiv-rag=src.ui.streamlit_app:main/d' setup.py && rm setup.py.bak
    echo "   ✓ Removed Streamlit console script from setup.py"
fi

echo ""
echo "✅ Cleanup completed!"
echo ""
echo "📁 Old files backed up to: old_ui_backup/"
echo "🚀 Use 'python launch_modern_ui.py' to start the modern UI"
echo ""
echo "⚠️  Note: If you need to restore the old UI, you can find the files in old_ui_backup/"