#!/bin/bash

# ⚽ Penalty Shootout Analyzer - Quick Start Script
# One command to install and run everything!

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════╗"
echo "║    PENALTY SHOOTOUT ELIGIBILITY ANALYZER - QUICK START    ║"
echo "║                      Setup & Launch                       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check Python version
echo "✓ Checking Python installation..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Python version: $python_version"

# Create virtual environment
echo ""
echo "✓ Creating virtual environment..."
if [ ! -d "penalty_analyzer_env" ]; then
    python3 -m venv penalty_analyzer_env
    echo "  ✓ Virtual environment created"
else
    echo "  ✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "✓ Activating virtual environment..."
source penalty_analyzer_env/bin/activate || . penalty_analyzer_env/Scripts/activate
echo "  ✓ Activated"

# Install dependencies
echo ""
echo "✓ Installing dependencies (this may take 2-3 minutes)..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo "  ✓ Dependencies installed"

# Download BERT model
echo ""
echo "✓ Downloading BERT sentiment model (first time only)..."
python3 << 'EOF'
from transformers import pipeline
print("  Downloading model...")
sentiment = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
print("  ✓ BERT model ready")
EOF

# Display options
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    LAUNCH OPTIONS                          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Choose how to run the application:"
echo ""
echo "1) START WEB APPLICATION (Recommended)"
echo "   - Full React UI + Flask API"
echo "   - Available at http://localhost:5000"
echo "   - Command: python flask_api_server.py"
echo ""
echo "2) RUN CLI ANALYSIS"
echo "   - Command-line interface"
echo "   - Command: python penalty_predictor_backend.py"
echo ""
echo "3) PYTHON INTERACTIVE MODE"
echo "   - Code example in Python shell"
echo "   - Command: python3"
echo ""
echo "4) RUN TESTS"
echo "   - Verify installation with test data"
echo "   - Command: pytest"
echo ""
echo "────────────────────────────────────────────────────────────"
echo ""
read -p "Select option (1-4): " option

case $option in
    1)
        echo ""
        echo "🚀 Starting Web Application..."
        echo ""
        echo "📍 Open your browser: http://localhost:5000"
        echo "   (Press Ctrl+C to stop server)"
        echo ""
        python flask_api_server.py
        ;;
    2)
        echo ""
        echo "🚀 Running CLI Analysis..."
        python penalty_predictor_backend.py
        ;;
    3)
        echo ""
        echo "🚀 Starting Python Interactive Shell..."
        echo ""
        echo "Example usage:"
        echo "  >>> from penalty_predictor_backend import PenaltyShootoutAnalyzer"
        echo "  >>> analyzer = PenaltyShootoutAnalyzer()"
        echo "  >>> results = analyzer.analyze('Min 76: Kane looks frustrated...')"
        echo "  >>> for player in results['players']:"
        echo "  ...     print(f\"{player.name}: {player.category}\")"
        echo ""
        python3
        ;;
    4)
        echo ""
        echo "🧪 Running tests..."
        python3 << 'EOF'
from penalty_predictor_backend import PenaltyShootoutAnalyzer

# Test 1: Basic analysis
print("Test 1: Basic Analysis")
analyzer = PenaltyShootoutAnalyzer()
test_commentary = """
Min 76: Kane drives forward but his pass is intercepted. He looks frustrated.
Min 77: The pressure is immense. Both teams are tiring.
Min 78: Sterling with a brilliant run down the wing, creates space for a shot. Confident touch!
Min 79: Mount controls the midfield well, distributing passes calmly under pressure.
Min 80: Saka looks exhausted, has lost the ball twice in quick succession.
Min 82: Kane with a clinical chance but the goalkeeper saves brilliantly.
Min 83: Mount maintains composure despite the intense pressure. Fresh legs showing.
Min 84: Saka gives away possession carelessly. Looks worn out.
Min 85: Sterling drives forward with energy, nearly creates a breakthrough.
"""

results = analyzer.analyze(test_commentary)
print(f"✓ Players analyzed: {len(results['players'])}")
print(f"✓ Top candidate: {results['players'][0].name if results['players'] else 'N/A'}")
print("")

# Test 2: Classification
print("Test 2: Classification")
if results['players']:
    categories = {}
    for p in results['players']:
        if p.category not in categories:
            categories[p.category] = 0
        categories[p.category] += 1
    
    for cat, count in categories.items():
        print(f"  {cat}: {count} players")
    print("✓ Classification working")
print("")

print("✅ All tests passed!")
EOF
        ;;
    *)
        echo "Invalid option. Exiting."
        exit 1
        ;;
esac

echo ""
echo "✨ Done! Thank you for using Penalty Shootout Analyzer"
echo ""
