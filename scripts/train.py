import argparse
from datetime import datetime

import numpy as np

from utils.training import run_training
from utils.data import load_mnist
from utils.experiments import save_experiment
from utils.evaluation import confusion_matrix
from utils.plots import plot_training_history, plot_confusion_matrix

from models.softmax_regression import SoftmaxRegression, cross_entropy
from models.mlp import MLP



def main():
    parser = argparse.ArgumentParser(
        description="Train a model on MNIST."
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["mlp", "softmax"],
        required=True,
        help="Model architecture to train.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of training samples per batch.",
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Number of hidden neurons (MLP only).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()

    if args.lr <= 0:
        parser.error("--lr must be positive")

    if args.epochs <= 0:
        parser.error("--epochs must be positive")

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    if args.hidden_dim <= 0:
        parser.error("--hidden-dim must be positive")

    config = {
        "model": args.model,
        "input_dim": 784,
        "num_classes": 10,
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "seed": args.seed,
        "validation_size": 5000,
    }

    X_train, Y_train, X_val, Y_val, _, _ = load_mnist(
        validation_size=config["validation_size"],
        seed=config["seed"],
    )

    model, history, metrics, val_probs = run_training(
        config=config, 
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val
    )


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    experiment_name = (
        f"{config['model']}_"
        f"lr{config['learning_rate']}_"
        f"seed{config['seed']}_"
        f"{timestamp}"
    )

    experiment_dir = save_experiment(
        name=experiment_name,
        config=config,
        metrics=metrics,
        history=history,
    )

    model.save_weights(
        experiment_dir / "model_weights.npz"
    )

    plot_training_history(
        history,
        save_path=experiment_dir / "training_loss.png",
        show=False,
    )

    y_true = np.argmax(Y_val, axis=1)
    y_pred = np.argmax(val_probs, axis=1)

    cm = confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        num_classes=config["num_classes"],
    )

    plot_confusion_matrix(
        cm=cm,
        class_names=[str(i) for i in range(config["num_classes"])],
        title=f"Confusion Matrix of {config['model'].upper()}",
        save_path=experiment_dir / "confusion_matrix.png",
    )

    print(f"\nTraining completed: {config['model']}")
    print(f"Validation Accuracy: {metrics['validation_accuracy']:.2%}")
    print(f"Validation Loss: {metrics['validation_loss']:.4f}")
    print(f"Training Time: {metrics['training_time_seconds']:.2f}s")
    print(f"Saved to: {experiment_dir}")
    


if __name__ == "__main__":
    main()