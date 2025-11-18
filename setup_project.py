"""
Setup Script for Credit Risk Scoring Model
Run this script to create all necessary project directories
Usage: python setup_project.py
"""

import os
from pathlib import Path

def create_project_structure():
    """Create all necessary directories for the project"""
    
    # Define directory structure
    directories = [
        "data/raw",
        "data/processed",
        "notebooks",
        "src",
        "app",
        "models",
        "outputs/figures",
        "outputs/metrics",
        "outputs/reports",
        "tests"
    ]
    
    print("🚀 Creating Credit Risk Model Project Structure...")
    print("=" * 60)
    
    # Create directories
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}/")
    
    # Create .gitkeep files in empty directories
    gitkeep_dirs = [
        "data/raw",
        "data/processed",
        "models",
        "outputs/figures",
        "outputs/metrics",
        "outputs/reports"
    ]
    
    print("\n📝 Creating .gitkeep files...")
    for directory in gitkeep_dirs:
        gitkeep_path = Path(directory) / ".gitkeep"
        gitkeep_path.touch()
        print(f"✅ Created: {directory}/.gitkeep")
    
    # Create __init__.py files in Python packages
    python_packages = ["src", "app", "tests"]
    
    print("\n🐍 Creating __init__.py files...")
    for package in python_packages:
        init_path = Path(package) / "__init__.py"
        init_path.write_text('"""Package initialization"""\n')
        print(f"✅ Created: {package}/__init__.py")
    
    print("\n" + "=" * 60)
    print("✨ Project structure created successfully!")
    print("\n📂 Directory tree:")
    print("""
credit-risk-model/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
├── app/
├── models/
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── reports/
└── tests/
    """)
    print("\n🎯 Next Steps:")
    print("1. Place your dataset in data/raw/")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Run data preprocessing: python src/data_preprocessing.py")
    print("4. Check config.py for configuration settings")
    print("\nHappy modeling! 🚀")

if __name__ == "__main__":
    create_project_structure()
