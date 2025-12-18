# Model Size Analyzer
# Run this to see which models take up the most space

import os
from pathlib import Path

def get_dir_size(path):
    """Calculate total size of directory in bytes."""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    except Exception as e:
        print(f"Error accessing {path}: {e}")
    return total

def format_size(bytes):
    """Convert bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} TB"

def analyze_models(models_dir="models"):
    """Analyze model directory sizes."""
    models_path = Path(models_dir)
    
    if not models_path.exists():
        print(f"Models directory not found: {models_dir}")
        return
    
    print("=" * 70)
    print("MODEL SIZE ANALYSIS")
    print("=" * 70)
    
    # Analyze each game
    for game_dir in models_path.iterdir():
        if game_dir.is_dir():
            print(f"\n{game_dir.name.upper()}:")
            print("-" * 70)
            
            model_sizes = []
            
            # Analyze each model version
            for model_dir in game_dir.iterdir():
                if model_dir.is_dir():
                    size = get_dir_size(model_dir)
                    model_sizes.append((model_dir.name, size))
            
            # Sort by size (largest first)
            model_sizes.sort(key=lambda x: x[1], reverse=True)
            
            total_size = sum(size for _, size in model_sizes)
            
            for model_name, size in model_sizes:
                print(f"  {model_name:40} {format_size(size):>15}")
            
            print("-" * 70)
            print(f"  {'TOTAL':40} {format_size(total_size):>15}")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS:")
    print("=" * 70)
    print("1. Keep only your production model(s)")
    print("2. Edit .dockerignore to exclude unused models")
    print("3. This will significantly reduce Docker image size")
    print("=" * 70)

if __name__ == "__main__":
    analyze_models()
