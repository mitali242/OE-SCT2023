import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights1 = np.random.randn(self.input_size, self.hidden_size)
        self.weights2 = np.random.randn(self.hidden_size, self.output_size)
        self.bias1 = np.zeros((1, self.hidden_size))
        self.bias2 = np.zeros((1, self.output_size))

    def forward(self, X):
        self.hidden_layer = sigmoid(np.dot(X, self.weights1) + self.bias1)
        self.output_layer = sigmoid(np.dot(self.hidden_layer, self.weights2) + self.bias2)
        return self.output_layer

    def backward(self, X, y, learning_rate):
        error = y - self.output_layer
        d_output = error * sigmoid_derivative(self.output_layer)
        error_hidden = d_output.dot(self.weights2.T)
        d_hidden = error_hidden * sigmoid_derivative(self.hidden_layer)
        self.weights2 += self.hidden_layer.T.dot(d_output) * learning_rate
        self.bias2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
        self.weights1 += X.T.dot(d_hidden) * learning_rate
        self.bias1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        for i in range(epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)

    def predict(self, X):
        return np.round(self.forward(X))

X = np.array([[1, -1], [-1, 1], [-1, -1], [1, 1]])
y = np.array([[1], [1], [-1], [-1]])

mlp = MLP(2, 2, 1)
mlp.train(X, y, 100000, 0.1)

print(mlp.predict(X))
