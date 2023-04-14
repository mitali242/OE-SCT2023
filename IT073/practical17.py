import numpy as np

train_inputs = np.array([[0.5, 5], [0.8, 7], [1.2, 8], [1.5, 10]])
train_outputs = np.array([0, 0, 1, 1])

def perceptron(inputs, weights):
    weighted_sum = np.dot(inputs, weights)

    if weighted_sum > 0:
        activation = 1
    else:
        activation = 0

    return activation


weights = np.random.rand(2)

learning_rate = 0.1
num_epochs = 100

for epoch in range(num_epochs):
    for inputs, output in zip(train_inputs, train_outputs):

        prediction = perceptron(inputs, weights)
        error = output - prediction
        weights += learning_rate * error * inputs

test_inputs = np.array([[1.7, 9], [0.6, 6]])
for inputs in test_inputs:
    prediction = perceptron(inputs, weights)
    if prediction == 1:
        print("Water level is high")
    else:
        print("Water level is low")
