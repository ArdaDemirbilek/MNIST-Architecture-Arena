"""
Convenience entry point for MNIST Architecture Arena.

Train individual models with:
    python -m scripts.train --model mlp --lr 0.01 --epochs 75
"""

from scripts.train import main


if __name__ == "__main__":
    main()