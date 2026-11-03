import numpy as np

from models.mlp import relu, MLP
from models.softmax_regression import cross_entropy


class TestRelu:
    def test_relu_mathematical_property(self):
        rng = np.random.default_rng()
        arr1 = rng.standard_normal((10,8))

        assert np.all(relu(arr1) >= 0.0)


class TestMLP:
    def test_shape_check(self):
        mlp1 = MLP(input_dim=784, hidden_dim=128, num_classes=10)

        assert  (mlp1.b1.shape == (mlp1.hidden_dim, )) and \
            (mlp1.W1.shape == (mlp1.input_dim, mlp1.hidden_dim))
        assert  (mlp1.b2.shape == (mlp1.num_classes, )) and \
            (mlp1.W2.shape == (mlp1.hidden_dim, mlp1.num_classes))

    def test_known_forward_value(self):
        test_mlp = MLP(input_dim=3, hidden_dim=2, num_classes=2)
        test_mlp.W1 = np.array([[1.0, 2.0], [-1.0, 1.0], [0.0, 3.0]])
        test_mlp.W2 = np.array([[1.0, -1.0], [2.0, -2.0]])

        X_train = np.array([[1.0, -1.0, 2.0], [2.0, 3.0, -1.0]])
        test_probs = test_mlp.forward(X=X_train)
        test_targets=np.array([[1.0, 0.0], [1.0, 0.0]])

        assert np.allclose(test_probs, test_targets, rtol=0, atol=1/100)

    def test_gradient_check(self):
        rng = np.random.default_rng(0)
        test_mlp = MLP(input_dim=5, hidden_dim=4, num_classes=3)
        X_train = rng.standard_normal((6, 5))
        Y_train = np.eye(3)[rng.integers(0, 3, size=6)]
        eps = 1e-6

        grads = test_mlp.compute_gradients(X_train, Y_train)
        params = [test_mlp.W1, test_mlp.b1, test_mlp.W2, test_mlp.b2]

        for param, grad in zip(params, grads):
            numerical_grad = np.zeros_like(param)
            for idx in np.ndindex(param.shape):
                original = param[idx]
                param[idx] = original + eps
                loss_plus = cross_entropy(test_mlp.forward(X_train), Y_train)
                param[idx] = original - eps
                loss_minus = cross_entropy(test_mlp.forward(X_train), Y_train)
                param[idx] = original
                numerical_grad[idx] = (loss_plus - loss_minus) / (2 * eps)

            assert np.allclose(grad, numerical_grad, rtol=0, atol=1e-7)