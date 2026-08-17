
import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.1, seed=42):
        self.learning_rate = learning_rate
        rng = np.random.default_rng(seed)
        sizes = [input_size] + hidden_layers + [output_size]
        self.weights, self.biases = [], []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            self.weights.append(rng.normal(0, np.sqrt(2/fan_in), (fan_in, fan_out)))
            self.biases.append(np.zeros((1, fan_out)))

    @staticmethod
    def sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)

    def forward(self, X):
        A = X
        activations = [X]
        zs = []
        for i in range(len(self.weights)-1):
            Z = A @ self.weights[i] + self.biases[i]
            zs.append(Z)
            A = self.relu(Z)
            activations.append(A)
        Z = A @ self.weights[-1] + self.biases[-1]
        zs.append(Z)
        A = self.sigmoid(Z)
        activations.append(A)
        return activations, zs

    @staticmethod
    def loss(y, p):
        p = np.clip(p, 1e-8, 1-1e-8)
        return float(-np.mean(y*np.log(p) + (1-y)*np.log(1-p)))

    def train(self, X, y, epochs=3000):
        losses = []
        for _ in range(epochs):
            A, Z = self.forward(X)
            losses.append(self.loss(y, A[-1]))

            delta = A[-1] - y
            gw = [None]*len(self.weights)
            gb = [None]*len(self.biases)

            gw[-1] = A[-2].T @ delta / len(X)
            gb[-1] = np.mean(delta, axis=0, keepdims=True)

            for i in range(len(self.weights)-2, -1, -1):
                delta = (delta @ self.weights[i+1].T) * self.relu_derivative(Z[i])
                gw[i] = A[i].T @ delta / len(X)
                gb[i] = np.mean(delta, axis=0, keepdims=True)

            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * gw[i]
                self.biases[i] -= self.learning_rate * gb[i]
        return losses

    def predict_proba(self, X):
        return self.forward(X)[0][-1]

    def predict(self, X):
        return (self.predict_proba(X) >= .5).astype(int)
