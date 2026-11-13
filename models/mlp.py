import numpy as np
from models.softmax_regression import softmax, cross_entropy


def relu(X: np.ndarray) -> np.ndarray:
    """
    Apply the ReLU activation element-wise, replacing
    negative values with zero.

    Args:
        X: Input array of any shape.

    Returns:
        Array with the same shape as X and all values >= 0.
    """
    return  np.maximum(0, X)


class MLP:
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        num_classes: int,
        seed: int = 42,
    ):
        self.rng = np.random.default_rng(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.W1 = self.rng.normal(
            loc=0.0,
            scale=np.sqrt(2 / input_dim),
            size=(input_dim, hidden_dim)
            )
        self.W2 = self.rng.normal(
            loc=0.0,
            scale=np.sqrt(2 / hidden_dim),
            size=(hidden_dim, num_classes)
            )
        self.b1 = np.zeros((hidden_dim, ))
        self.b2 = np.zeros((num_classes, ))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Compute class probabilities by passing the input through
        a ReLU hidden layer and a softmax output layer.

        Args:
            X: Input batch of shape (batch_size, input_dim).

        Returns:
            Class probabilities of shape (batch_size, num_classes).
        """
        z1 = X @ self.W1 + self.b1
        l1 = relu(z1)
        z2 = l1 @ self.W2 + self.b2
        l2 = softmax(z2)
        return l2
 
    def compute_gradients(
            self, X: np.ndarray, Y: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute gradients of mean cross-entropy loss with respect
        to the weights and biases of both layers using backpropagation.

        Args:
            X: Input batch of shape (batch_size, input_dim).
            Y: One-hot labels of shape (batch_size, num_classes).

        Returns:
            Tuple containing dW1, db1, dW2 and db2, each with the
            same shape as the parameter it belongs to.
        """
        m = X.shape[0]
        z1 = X @ self.W1 + self.b1
        l1 = relu(z1)
        z2 = l1 @ self.W2 + self.b2
        l2 = softmax(z2)

        dz2 = l2 - Y
        dW2 = (l1.T @ dz2) / m
        db2 = np.sum(dz2, axis=0) / m

        dz1 = (dz2 @ self.W2.T) * (z1 > 0)
        dW1 = (X.T @ dz1) / m
        db1 = np.sum(dz1, axis=0) / m
        return dW1, db1, dW2, db2

    def fit(self, X_train, Y_train, epochs=20, batch_size=64, learning_rate=0.01):
        """
        Train the model with mini-batch gradient descent. The training
        data is shuffled at the start of every epoch, and the loss on
        the full training set is recorded after each epoch.

        Args:
            X_train: Training inputs of shape (num_samples, input_dim).
            Y_train: One-hot labels of shape (num_samples, num_classes).
            epochs: Number of passes over the training data.
            batch_size: Number of samples per gradient update.
            learning_rate: Step size for the parameter updates.

        Returns:
            Dictionary with a "train_loss" list holding the mean
            cross-entropy loss after each epoch.
        """
        history = {
            "train_loss": []
        }
        num_exp = X_train.shape[0]

        for _ in range(epochs):
            permutation_indices = self.rng.permutation(num_exp)
            X_shuffled, Y_shuffled = X_train[permutation_indices], Y_train[permutation_indices]

            for start in range(0, num_exp, batch_size):
                end = min(start + batch_size, num_exp)
                X_batch, Y_batch = X_shuffled[start:end], Y_shuffled[start:end]
                dW1, db1, dW2, db2 = self.compute_gradients(X_batch, Y_batch)
                self.W1 -= learning_rate * dW1
                self.b1 -= learning_rate * db1
                self.W2 -= learning_rate * dW2
                self.b2 -= learning_rate * db2

            probs = self.forward(X_train)
            loss = cross_entropy(probs, Y_train)
            history["train_loss"].append(float(loss))

        return history

    def save_weights(self, path):
        """
        Save the model's learned parameters to an NPZ file.

        Args:
            path: Destination path of the NPZ file.
        """
        np.savez(
            path,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
        )

    def load_weights(self, path):
        """
        Load model parameters after validating their shapes.
        """
        with np.load(path, allow_pickle=False) as data:
            weights = {}

            for name in ("W1", "b1", "W2", "b2"):
                saved = data[name]
                current = getattr(self, name)

                if saved.shape != current.shape:
                    raise ValueError(
                        f"Shape mismatch for {name}: "
                        f"expected {current.shape}, got {saved.shape}"
                    )

                weights[name] = saved.copy()

        for name, values in weights.items():
            setattr(self, name, values)