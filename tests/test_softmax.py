import numpy as np
import pytest

from models.softmax_regression import softmax, cross_entropy
from models.softmax_regression import SoftmaxRegression 



class TestSoftmax:
    def test_softmax_probability_normalization(self):
        """Checks that each row of softmax output sums to 1."""
        logits = np.array([
            [2.0, 1.0, 0.0],
            [0.5, 1.5, 3.0]
        ])
        probabilities = softmax(logits)
        row_sums = np.sum(probabilities, axis=1)

        assert np.allclose(row_sums, 1.0)

    def test_softmax_shape_prevention(self):
        """Checks that softmax output has the same shape as the input logits."""
        logits = np.array([
            [1000.1, 1000.2, 1000.4],
            [550.9, 999.9, 1000.0],
            [1500.0, 0.0, 250.5]
        ])

        probabilities = softmax(logits)

        assert np.shape(logits) == np.shape(probabilities)

    def test_softmax_numerical_stability(self):
        """Checks that very large logits give finite probabilities between 0 and 1, with no overflow or NaN."""
        logits = np.array([
            [1000.1, 1000.2, 1000.4],
            [550.9, 999.9, 1000.0],
            [1500.0, 0.0, 250.5]
        ])
        probabilities = softmax(logits)

        assert np.all(np.isfinite(probabilities))
        assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))

    def test_softmax_shift_variance(self):
        """Checks that adding the same constant to every logit leaves the softmax output unchanged."""
        logits = np.array([
            [3.5, 1000.2, 4.4],
            [0.5, 7.8, 6.3],
            [3.0, 0.0, 9.9]
        ])
        logits_added = logits + 1.0

        probabilities_normal = softmax(logits)
        probabilities_added = softmax(logits_added)

        assert np.allclose(probabilities_normal, probabilities_added)


class TestCrossEntropy:
    def test_cross_entropy_known_loss_value(self):
        """Checks that the loss for a small hand-computed example matches the expected value."""
        test_probs1 = np.array([
            [0.1, 0.8, 0.1],
            [0.2, 0.4, 0.3]
        ])
        test_targets1 = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        test_probs2 = np.array([
            [0.4, 0.3, 0.3],
            [0.2, 0.4, 0.4]
        ])
        test_targets2 = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0]
        ])

        assert np.isclose(cross_entropy(test_probs1, test_targets1), 0.713558)
        assert np.isclose(cross_entropy(test_probs2, test_targets2), 1.060132)

    def test_cross_entropy_numerical_stability(self):
        """Checks that the loss stays finite when a predicted probability is 0 or very close to 0."""
        test_probs = np.array([
            [0.4, 0.0, 0.6],
            [0.2, 0.5, 0.3]
        ])
        test_targets = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])

        assert np.isfinite(cross_entropy(test_probs, test_targets))

    def test_cross_entropy_mathematical_properties(self):
        """Checks that the loss is never negative and gets larger as the true class's probability gets smaller."""
        test_probs1 = np.array([
            [0.09, 0.01, 0.09],
            [0.2, 0.4, 0.3]
        ])
        test_probs2 = np.array([
            [0.09, 0.005, 0.09],
            [0.2, 0.4, 0.1]
        ])
        test_targets = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])

        assert cross_entropy(test_probs1, test_targets) >= 0.0
        assert cross_entropy(test_probs2, test_targets) > cross_entropy(test_probs1, test_targets)

    def test_cross_entropy_perfect_prediction(self):
        """Checks that the loss is (close to) 0 when the true class gets probability 1."""
        test_probs = np.array([[0.0, 1.0, 0.0]])
        test_targets = np.array([[0.0, 1.0, 0.0]])

        assert np.isclose(cross_entropy(test_probs, test_targets), 0.0)

class TestSoftmaxRegression:
    def test_gradient_values(self):
        """Checks that the gradients at zero-initialized parameters match hand-computed values for W and b."""
        X = np.array([
            [1.0, 2.0],
            [-1.0, 0.0]
        ])
        Y = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

        dW_target = np.array([
            [0.5, -0.5, 0.0],
            [1/3, -2/3, 1/3]
        ])
        db_target = np.array([-1/6, -1/6, 1/3])

        test_model = SoftmaxRegression(2, 3)
        dW_calc, db_calc = test_model.compute_gradients(X, Y)

        assert np.allclose(dW_calc, dW_target)
        assert np.allclose(db_calc, db_target)

    def test_checkpoint_mechanism(self, tmp_path):
        """
        Verify that saving and loading model weights
        preserves predictions after training.
        """
        rng = np.random.default_rng(123)
    
        X_train = rng.standard_normal((16, 5))
        Y_train = np.eye(3)[rng.integers(0, 3, size=16)]
    
        original_model = SoftmaxRegression(
            input_dim=5, 
            num_classes=3, 
            seed=42
        )
    
        original_model.fit(
            X_train,
            Y_train,
            epochs=2,
            batch_size=4,
            learning_rate=0.01,
        )
    
        original_predictions = original_model.forward(X_train)
    
        weights_path = tmp_path / "model_weights.npz"
    
        original_model.save_weights(weights_path)
    
        restored_model = SoftmaxRegression(
            input_dim=5, 
            num_classes=3, 
            seed=32,
        )
    
        restored_model.load_weights(weights_path)
    
        restored_predictions = restored_model.forward(X_train)
    
        np.testing.assert_allclose(
            original_predictions,
            restored_predictions,
            rtol=0,
            atol=1e-12,
        )
    
        for name in ("W", "b"):
            np.testing.assert_array_equal(
                getattr(original_model, name),
                getattr(restored_model, name),
            )

    def test_checkpoint_shape_mismatch(self, tmp_path):
        """Checks that loading weights saved from a model with a different input dimension raises a ValueError."""
        original = SoftmaxRegression(
            input_dim=5,
            num_classes=3,
        )

        path = tmp_path / "weights.npz"
        original.save_weights(path)

        incompatible = SoftmaxRegression(
            input_dim=8,
            num_classes=3,
        )

        with pytest.raises(ValueError, match="Shape mismatch"):
            incompatible.load_weights(path)

