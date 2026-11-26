import numpy as np
import pytest

from losses import mse, mae, bce, cce


class TestMSE:
    def test_basic(self):
        labels = np.array([1,2,3])
        predictions = np.array([1,2,4])

        assert mse(labels,predictions) == pytest.approx(1/3)

    def test_perfect_prediction(self):
        labels = np.array([1,2,3])
        assert mse(labels, labels) == pytest.approx(0)

    @pytest.mark.parametrize("seed", [1,42, 130, 400])
    def test_non_negative(self, seed):
        rng = np.random.default_rng(seed)
        labels = rng.integers(0,2, size=50)
        predictions = rng.random(size=50)

        assert bce(labels, predictions) >= 0


class TestMAE:
    def test_basic(self):
        labels = np.array([1, 2, 3])
        predictions = np.array([2, 2, 5])

        assert mae(labels, predictions) == pytest.approx(1.0)

    def test_perfect_prediction(self):
        labels = np.array([1, 2, 3])
        assert mae(labels, labels) == pytest.approx(0)

    @pytest.mark.parametrize("seed", [1, 42, 130, 400])
    def test_non_negative(self, seed):
        rng = np.random.default_rng(seed)
        labels = rng.integers(-100, 100, size=100)
        predictions = rng.integers(-100, 100, size=100)

        assert mae(labels, predictions) >= 0

    

class TestBCE:
    def test_basic(self):
        labels = np.array([1, 0])
        predictions = np.array([0.9, 0.1])

        assert bce(labels, predictions) == pytest.approx(-np.log(0.9))

    def test_perfect_prediction(self):
        labels = np.array([1, 0])
        predictions = np.array([1.0, 0.0])

        assert bce(labels, predictions) < 1e-5

    @pytest.mark.parametrize("seed", [1,42, 130, 400])
    def test_non_negative(self, seed):
        rng = np.random.default_rng(seed)
        labels = rng.integers(0,2, size=50)
        predictions = rng.random(size=50)

        assert bce(labels, predictions) >= 0

class TestCCE:
    def test_basic(self):
        labels = np.array([
            [1, 0, 0],
            [0, 1, 0]
        ])
        predictions = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1]
        ])

        expected = -(np.log(0.7) + np.log(0.8)) / 2
        assert cce(labels, predictions) == pytest.approx(expected)

    def test_perfect_prediction(self):
        labels = np.eye(3)
        predictions = np.eye(3)

        assert cce(labels, predictions) < 1e-5

    @pytest.mark.parametrize("seed", [1, 42, 130, 400])
    def test_non_negative(self, seed):
        rng = np.random.default_rng(seed)

        labels = rng.integers(0, 3, size=100)
        labels = np.eye(3)[labels]

        predictions = rng.random(size=(100, 3))
        predictions = predictions / predictions.sum(axis=1, keepdims=True)

        assert cce(labels, predictions) >= 0
