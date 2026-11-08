from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np



def plot_training_history(history, save_path=None):
    """
    Plot training loss over completed epochs.
    """
    losses = history["train_loss"]
    epochs = range(1, len(losses) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        epochs,
        losses,
        color="#2563eb",
        linewidth=2,
        marker="o",
        markersize=4,
        label="Training Loss"
    )

    ax.set_title("Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")

    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    plt.show()


def plot_model_comparison(
    softmax_history_path,
    mlp_history_path,
    save_path="experiments/comparison/training_loss.png"
):
    """
    Compare the training loss curves of Softmax Regression
    and MLP using previously saved experiment histories.

    Args:
        softmax_history_path: Path to Softmax history.json.
        mlp_history_path: Path to MLP history.json.
        save_path: Destination for the comparison figure.
    """

    with open(softmax_history_path, "r", encoding="utf-8") as f:
        softmax_history = json.load(f)

    with open(mlp_history_path, "r", encoding="utf-8") as f:
        mlp_history = json.load(f)

    softmax_losses = softmax_history["train_loss"]
    mlp_losses = mlp_history["train_loss"]

    softmax_epochs = range(1, len(softmax_losses) + 1)
    mlp_epochs = range(1, len(mlp_losses) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        softmax_epochs,
        softmax_losses,
        label="Softmax Regression",
        color="#64748b",
        marker="o",
        linewidth=2.5,
    )

    ax.plot(
        mlp_epochs,
        mlp_losses,
        label="MLP (128 hidden neurons)",
        color="#2563eb",
        marker="o",
        linewidth=2.5,
    )

    ax.set_title("MNIST - Training Loss Comparison")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")

    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(save_path, dpi=180, bbox_inches="tight")

    plt.show()
    plt.close(fig)


def plot_confusion_matrix(
    cm,
    class_names=None,
    title="Confusion Matrix",
    save_path=None,
):
    """
    Visualize a confusion matrix using a heatmap.

    Parameters
    ----------
    cm : np.ndarray
        Square confusion matrix where rows represent
        true labels and columns represent predictions.

    class_names : list, optional
        Names of the classes.

    title : str
        Title of the plot.

    save_path : str or Path, optional
        Path where the figure should be saved.
    """

    cm = np.asarray(cm)

    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("Confusion matrix must be square.")

    num_classes = cm.shape[0]

    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    if len(class_names) != num_classes:
        raise ValueError(
            "Number of class names must match matrix dimensions."
        )

    fig, ax = plt.subplots(figsize=(9, 7))

    image = ax.imshow(
        cm,
        interpolation="nearest",
        cmap="Blues",
    )

    fig.colorbar(image, ax=ax)

    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted Label",
        ylabel="True Label",
        title=title,
    )

    threshold = cm.max() / 2

    for i in range(num_classes):
        for j in range(num_classes):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if cm[i, j] > threshold
                    else "black"
                ),
                fontsize=9,
            )

    ax.set_ylim(num_classes - 0.5, -0.5)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight",
        )

    plt.show()
    plt.close(fig)