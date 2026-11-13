import time
import numpy as np

from models.softmax_regression import SoftmaxRegression, cross_entropy
from models.mlp import MLP

from utils.experiments import save_experiment


def create_model(config: dict):
    if config["model"] == "softmax":
        return SoftmaxRegression(
            input_dim=784,
            num_classes=10,
            seed=config["seed"],
        )

    if config["model"] == "mlp":
        return MLP(
            input_dim=784,
            hidden_dim=config["hidden_dim"],
            num_classes=10,
            seed=config["seed"],
        )

    raise ValueError(f"Unsupported model: {config['model']}")


def run_training(
    config: dict,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
):
    model = create_model(config)

    initial_loss = cross_entropy(
        model.forward(X_train),
        Y_train,
    )

    start = time.perf_counter()

    history = model.fit(
        X_train,
        Y_train,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        X_val=X_val,
        Y_val=Y_val,
    )

    training_time = time.perf_counter() - start

    val_probs = model.forward(X_val)

    val_accuracy = np.mean(
        np.argmax(val_probs, axis=1)
        == np.argmax(Y_val, axis=1)
    )

    val_loss = cross_entropy(val_probs, Y_val)

    metrics = {
        "initial_training_loss": float(initial_loss),
        "final_training_loss": float(history["train_loss"][-1]),
        "validation_accuracy": float(val_accuracy),
        "validation_loss": float(val_loss),
        "training_time_seconds": float(training_time),
    }

    return model, history, metrics, val_probs