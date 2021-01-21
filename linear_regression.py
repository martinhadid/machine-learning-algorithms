import numpy as np

class LinearRegression:

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
			y_pred = np.dot(X, self.weights) + self.bias

			# compute gradients
			dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
			db = (1 / n_samples) * np.sum(y_pred - y)

			# update parameters
			self.weights -= self.lr * dw
			self.bias -= self.lr * db

	def predict(self, X):
		return np.dot(X, self.weights) + self.bias