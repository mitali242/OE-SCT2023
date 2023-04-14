import numpy as np
from sklearn.linear_model import Perceptron

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

y = np.array([0, 0, 0, 1])

perceptron = Perceptron(max_iter=100, eta0=0.1, random_state=0)

perceptron.fit(X, y)

test_data = np.array([[1, 1], [0, 0], [0, 1], [1, 0]])
predictions = perceptron.predict(test_data)

print(predictions)
