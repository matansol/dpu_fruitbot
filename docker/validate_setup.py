# Docker Setup Validator
# Run this before building to check if all required files exist

import os
from pathlib import Path

def check_file(path, required=True):
    """Check if a file exists."""
    exists = Path(path).exists()
    status = "✓" if exists else ("✗ MISSING" if required else "⚠ Optional")
    print(f"  {status} {path}")
    return exists

def check_dir(path, required=True):
    """Check if a directory exists."""
    exists = Path(path).is_dir()
    status = "✓" if exists else ("✗ MISSING" if required else "⚠ Optional")
    
    if exists:
        # Count items in directory
        items = list(Path(path).iterdir())
        print(f"  {status} {path}/ ({len(items)} items)")
    else:
        print(f"  {status} {path}/")
    
    return exists

def main():
    print("=" * 70)
    print("DOCKER SETUP VALIDATION")
    print("=" * 70)
    
    all_good = True
    
    # Required files
    print("\n📄 REQUIRED FILES:")
    print("-" * 70)
    all_good &= check_file("Dockerfile")
    all_good &= check_file("docker-compose.yml")
    all_good &= check_file(".dockerignore")
    all_good &= check_file("docker/requirements.txt")
    all_good &= check_file("app.py")
    all_good &= check_file("dpu_clf.py")
    all_good &= check_file("setup.py")
    all_good &= check_file("README.md")
    
    # Required directories
    print("\n📁 REQUIRED DIRECTORIES:")
    print("-" * 70)
    all_good &= check_dir("static")
    all_good &= check_dir("templates")
    all_good &= check_dir("models")
    all_good &= check_dir("procgen")
    
    # Optional files
    print("\n📋 OPTIONAL FILES:")
    print("-" * 70)
    check_file(".env", required=False)
    
    # Check static subdirectories
    print("\n🎨 STATIC FILES:")
    print("-" * 70)
    check_dir("static/css", required=False)
    check_dir("static/js", required=False)
    check_dir("static/images", required=False)
    
    # Check templates
    print("\n📝 TEMPLATES:")
    print("-" * 70)
    check_file("templates/index.html")
    
    # Check for at least one model
    print("\n🤖 MODELS:")
    print("-" * 70)
    models_path = Path("models/fruitbot")
    if models_path.exists():
        model_dirs = [d for d in models_path.iterdir() if d.is_dir()]
        print(f"  ✓ Found {len(model_dirs)} model(s) in models/fruitbot/")
        for model_dir in sorted(model_dirs)[:5]:  # Show first 5
            print(f"    - {model_dir.name}")
        if len(model_dirs) > 5:
            print(f"    ... and {len(model_dirs) - 5} more")
        
        if len(model_dirs) > 3:
            print(f"\n  ⚠ WARNING: {len(model_dirs)} models will increase Docker image size!")
            print(f"    Consider keeping only 1-2 production models.")
    else:
        print("  ✗ MISSING models/fruitbot/ directory")
        all_good = False
    
    # Summary
    print("\n" + "=" * 70)
    if all_good:
        print("✓ All required files and directories found!")
        print("=" * 70)
        print("\n🚀 READY TO BUILD!")
        print("\nNext steps:")
        print("  1. Run: docker-compose up --build")
        print("  2. Visit: http://localhost:8000")
        print("  3. Check health: http://localhost:8000/health")
    else:
        print("✗ Some required files or directories are missing!")
        print("=" * 70)
        print("\n⚠ FIX THE ISSUES ABOVE BEFORE BUILDING")
    
    print()

if __name__ == "__main__":
    main()
