import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    """
    Compute numerically stable softmax probabilities row-wise.
    Each row represents the logits of one sample. The maximum
    logit is subtracted before exponentiation to prevent possible overflows.

    Args:
        z (np.ndarray): Logits with shape (batch_size, num_classes).

    Returns:
        np.ndarray: Class probabilities with the same shape as z.
                    Each row sums to approximately 1.
    """
    z_new = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_new)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy(probs: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute the mean categorical cross-entropy loss.

    Args:
        probs: Predicted class probabilities, shape (batch_size, num_classes).
        targets: One-hot encoded labels with the same shape.

    Returns:
        Mean cross-entropy loss across the batch.
    """
    safe_probs = np.clip(probs, 1e-12, 1.0)
    return - np.mean(np.sum(targets * np.log(safe_probs), axis=1))


class SoftmaxRegression:
    def __init__(
        self, 
        input_dim: int, 
        num_classes: int, 
        seed: int = 42
    ):
        self.rng = np.random.default_rng(seed)
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.W = np.zeros((self.input_dim, self.num_classes))
        self.b = np.zeros(self.num_classes)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Compute class probabilities by applying softmax
        to the linear logits XW + b.

        Args:
            X: Input batch of shape (batch_size, input_dim).

        Returns:
            Class probabilities of shape (batch_size, num_classes).
        """
        logits = X @ self.W + self.b
        return softmax(logits)

    def compute_gradients(self, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:        
        """
        Compute gradients of mean cross-entropy loss with respect
        to the model weights and biases.

        Args:
            X: Input batch of shape (batch_size, input_dim).
            Y: One-hot labels of shape (batch_size, num_classes).

        Returns:
            Tuple containing weight and bias gradients.
        """
        m = X.shape[0]
        P = self.forward(X)
        error = P - Y        
        dW = (X.T @ error) / m
        db = np.sum(error, axis=0) / m        
        return dW, db
        
    def fit(self, X_train, Y_train, epochs=20, batch_size=64, learning_rate=0.01, X_val=None, Y_val=None):
        """
        Train the model with mini-batch gradient descent. The training
        data is shuffled at the start of every epoch, and the loss on
        the full training set is recorded after each epoch. If
        validation data is given, the validation loss is recorded too.

        Args:
            X_train: Training inputs of shape (num_samples, input_dim).
            Y_train: One-hot labels of shape (num_samples, num_classes).
            epochs: Number of passes over the training data.
            batch_size: Number of samples per gradient update.
            learning_rate: Step size for the parameter updates.
            X_val: Optional validation inputs of shape (num_val, input_dim).
            Y_val: Optional one-hot validation labels of shape
                (num_val, num_classes). Must be given together with X_val.

        Returns:
            Dictionary with "train_loss" and "val_loss" lists holding the
            mean cross-entropy loss after each epoch. "val_loss" stays
            empty when no validation data is given.

        Raises:
            ValueError: If only one of X_val and Y_val is provided.
        """
        if (X_val is None) != (Y_val is None):
            raise ValueError("X_val and Y_val must be provided together.")
        
        history = {
            "train_loss": [],
            "val_loss": [],
        }
        num_exp = X_train.shape[0]

        for _ in range(epochs):
            permutation_indices = self.rng.permutation(num_exp)
            X_shuffled, Y_shuffled = X_train[permutation_indices], Y_train[permutation_indices]

            for start in range(0, num_exp, batch_size):
                end = min(start + batch_size, num_exp)
                X_batch, Y_batch = X_shuffled[start:end], Y_shuffled[start:end]
                dW, db = self.compute_gradients(X_batch, Y_batch)
                self.W -= learning_rate * dW
                self.b -= learning_rate * db
                
            s = self.forward(X_train)
            loss = cross_entropy(s, Y_train)
            history["train_loss"].append(float(loss))

            if X_val is not None and Y_val is not None:
                val_probs = self.forward(X_val)
                val_loss = cross_entropy(val_probs, Y_val)
                history["val_loss"].append(float(val_loss))
            
        return history

    def save_weights(self, path):
        """
        Save the model's learned parameters to an NPZ file.
    
        Args:
            path: Destination path of the NPZ file.
        """
        np.savez(
            path,
            W=self.W,
            b=self.b,
        )
    
    def load_weights(self, path):
        """
        Load model parameters after validating their shapes.
        """
        with np.load(path, allow_pickle=False) as data:
            weights = {}

            for name in ("W", "b"):
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