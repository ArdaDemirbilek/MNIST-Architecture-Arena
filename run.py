import numpy as np

from models.softmax_regression import SoftmaxRegression, cross_entropy
from models.mlp import MLP
from utils.data import load_mnist
from utils.experiments import save_experiment
from utils.evaluation import (
    confusion_matrix,
    per_class_accuracy,
    categorize_predictions,
)

from utils.plots import (
    plot_training_history,
    plot_model_comparison,
    plot_confusion_matrix,
    plot_per_class_accuracy,
    plot_prediction_examples,
)




def main():
    # 1. Load MNIST dataset
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_mnist()

    print(f"Training:   {X_train.shape}, {Y_train.shape}")
    print(f"Validation: {X_val.shape}, {Y_val.shape}")
    print(f"Test:       {X_test.shape}, {Y_test.shape}")

    # ============================================================
    # EXPERIMENT 1: SOFTMAX REGRESSION
    # ============================================================

    softmax_config = {
        "model": "SoftmaxRegression",
        "input_dim": 784,
        "num_classes": 10,
        "epochs": 10,
        "batch_size": 64,
        "learning_rate": 0.1,
        "seed": 42,
        "train_samples": 55000,
        "validation_samples": 5000,
    }

    softmax_model = SoftmaxRegression(
        input_dim=softmax_config["input_dim"],
        num_classes=softmax_config["num_classes"],
    )

    softmax_initial_loss = float(
        cross_entropy(softmax_model.forward(X_train), Y_train)
    )

    softmax_history = softmax_model.fit(
        X_train,
        Y_train,
        epochs=softmax_config["epochs"],
        batch_size=softmax_config["batch_size"],
        learning_rate=softmax_config["learning_rate"],
    )

    softmax_val_probs = softmax_model.forward(X_val)

    softmax_val_accuracy = float(
        np.mean(
            np.argmax(softmax_val_probs, axis=1)
            == np.argmax(Y_val, axis=1)
        )
    )

    softmax_val_loss = float(
        cross_entropy(softmax_val_probs, Y_val)
    )

    softmax_metrics = {
        "initial_training_loss": softmax_initial_loss,
        "final_training_loss": float(softmax_history["train_loss"][-1]),
        "validation_accuracy": softmax_val_accuracy,
        "validation_loss": softmax_val_loss,
    }

    print("\n--- Softmax Regression Results ---")
    print(f"Initial Training Loss: {softmax_initial_loss:.4f}")
    print(f"Final Training Loss: {softmax_metrics['final_training_loss']:.4f}")
    print(f"Validation Accuracy: {softmax_val_accuracy:.2%}")
    print(f"Validation Loss: {softmax_val_loss:.4f}")

    softmax_dir = save_experiment(
        name="softmax_baseline",
        config=softmax_config,
        metrics=softmax_metrics,
        history=softmax_history,
    )

    plot_training_history(
        softmax_history,
        save_path=softmax_dir / "training_loss.png",
    )

    y_true = np.argmax(Y_val, axis=1)
    softmax_pred = np.argmax(softmax_val_probs, axis=1)

    softmax_cm = confusion_matrix(
        y_true,
        softmax_pred,
        num_classes=10,
    )

    assert softmax_cm.sum() == len(Y_val)

    softmax_per_class = per_class_accuracy(softmax_cm)

    plot_confusion_matrix(
        cm=softmax_cm,
        class_names=[str(i) for i in range(10)],
        title="Confusion Matrix of Softmax Regression",
        save_path=softmax_dir / "confusion_matrix.png",
    )

    # ============================================================
    # EXPERIMENT 2: MULTILAYER PERCEPTRON
    # ============================================================

    mlp_config = {
        "model": "MLP",
        "input_dim": 784,
        "hidden_dim": 128,
        "num_classes": 10,
        "epochs": 10,
        "batch_size": 64,
        "learning_rate": 0.01,
        "seed": 42,
        "train_samples": 55000,
        "validation_samples": 5000,
    }

    mlp_model = MLP(
        input_dim=mlp_config["input_dim"],
        hidden_dim=mlp_config["hidden_dim"],
        num_classes=mlp_config["num_classes"],
    )

    mlp_initial_loss = float(
        cross_entropy(mlp_model.forward(X_train), Y_train)
    )

    mlp_history = mlp_model.fit(
        X_train,
        Y_train,
        epochs=mlp_config["epochs"],
        batch_size=mlp_config["batch_size"],
        learning_rate=mlp_config["learning_rate"],
    )

    mlp_val_probs = mlp_model.forward(X_val)

    mlp_val_accuracy = float(
        np.mean(
            np.argmax(mlp_val_probs, axis=1)
            == np.argmax(Y_val, axis=1)
        )
    )

    mlp_val_loss = float(
        cross_entropy(mlp_val_probs, Y_val)
    )

    mlp_metrics = {
        "initial_training_loss": mlp_initial_loss,
        "final_training_loss": float(mlp_history["train_loss"][-1]),
        "validation_accuracy": mlp_val_accuracy,
        "validation_loss": mlp_val_loss,
    }

    print("\n--- MLP Results ---")
    print(f"Initial Training Loss: {mlp_initial_loss:.4f}")
    print(f"Final Training Loss: {mlp_metrics['final_training_loss']:.4f}")
    print(f"Validation Accuracy: {mlp_val_accuracy:.2%}")
    print(f"Validation Loss: {mlp_val_loss:.4f}")

    mlp_dir = save_experiment(
        name="mlp_128",
        config=mlp_config,
        metrics=mlp_metrics,
        history=mlp_history,
    )

    plot_training_history(
        mlp_history,
        save_path=mlp_dir / "training_loss.png",
    )

    mlp_pred = np.argmax(mlp_val_probs, axis=1)

    mlp_cm = confusion_matrix(
        y_true,
        mlp_pred,
        num_classes=10,
    )

    assert mlp_cm.sum() == len(Y_val)

    mlp_per_class = per_class_accuracy(mlp_cm)

    plot_confusion_matrix(
        cm=mlp_cm,
        class_names=[str(i) for i in range(10)],
        title="Confusion Matrix of MLP",
        save_path=mlp_dir / "confusion_matrix.png",
    )

    # ============================================================
    # FINAL COMPARISON
    # ============================================================

    print("\n========== MODEL COMPARISON ==========")
    print(
        f"Softmax Regression: "
        f"Accuracy = {softmax_val_accuracy:.2%}, "
        f"Loss = {softmax_val_loss:.4f}"
    )
    print(
        f"MLP (128 hidden):   "
        f"Accuracy = {mlp_val_accuracy:.2%}, "
        f"Loss = {mlp_val_loss:.4f}"
    )

    plot_model_comparison(
        softmax_history_path="experiments/softmax_baseline/history.json",
        mlp_history_path="experiments/mlp_128/history.json",
    )

    assert np.isclose(
        np.trace(softmax_cm) / softmax_cm.sum(),
        softmax_val_accuracy,
    )

    assert np.isclose(
        np.trace(mlp_cm) / mlp_cm.sum(),
        mlp_val_accuracy,
    )

    plot_per_class_accuracy(
        softmax_accuracies=softmax_per_class,
        mlp_accuracies=mlp_per_class,
        save_path="experiments/comparison/per_class_accuracy.png",
    )

    print("\n  Softmax   |   MLP")
    for i in range(len(softmax_per_class)):
        print(i, end=": ")
        print(
            np.round(softmax_per_class[i], decimals=2), 
            np.round(mlp_per_class[i], decimals=2), 
            sep="      "
        )

    print("----------------------\n\n")

    categories = categorize_predictions(
        y_true,
        softmax_pred,
        mlp_pred,
    )

    corrected_eights = categories["mlp_corrected"][
        y_true[categories["mlp_corrected"]] == 8
    ]

    plot_prediction_examples(
        X=X_val,
        y_true=y_true,
        softmax_pred=softmax_pred,
        mlp_pred=mlp_pred,
        indices=corrected_eights,
        title="Digit 8 - Errors Corrected by MLP",
        save_path="experiments/comparison/corrected_eights.png",
        max_samples=16,
    )

    print("\nExperiments saved successfully.")


if __name__ == "__main__":
    main()