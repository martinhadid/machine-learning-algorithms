import numpy as np

class BaseRegression:

	def __init__(self, learning_rate=0.001, n_iters=1000):
		self.lr = learning_rate
		self.n_iters = n_iters
		self.weights = None
		self.bias = None

	def fit(self, X, y):
		n_samples, n_features = X.shape

		# init parameters
		self.weights = np.zeros(n_features)
		self.bias = 0

		# gradient descent
		for _ in range(self.n_iters):
			y_pred = self._approximation(X, self.weights, self.bias)

			# compute gradients - cost function is MSE
			dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
			db = (1 / n_samples) * np.sum(y_pred - y)

			# update parameters
			self.weights -= self.lr * dw
			self.bias -= self.lr * db

	def predict(self, X):
		return self._predict(X, self.weights, self.bias)

	def _predict(self, X, w, b):
		raise NotImplementedError()


class LinearRegression(BaseRegression):

	def _approximation(self, X, w, b):
		return np.dot(X, w) + b

	def _predict(self, X, w, b):
		return np.dot(X, w) + b


class LogisticRegression(BaseRegression):

	def _approximation(self, X, w, b):
		linear_model = np.dot(X, w) + b
		return self._sigmoid(linear_model)

	def _predict(self, X, w, b):
		linear_model = np.dot(X, w) + b
		y_pred = self._sigmoid(linear_model)
		y_pred_cls = [1 if i > 0.5 else 0 for i in y_pred]
		return np.array(y_pred_cls)

	def _sigmoid(self, x):
		return 1 / (1 + np.exp(-x))
