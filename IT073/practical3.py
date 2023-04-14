import numpy as np

# Define training data for letters A and H
train_A = np.array([[1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1]])

train_H = np.array([[1, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1]])

# Flatten training data to 1D arrays
train_A = train_A.flatten()
train_H = train_H.flatten()

# Define Hebb learning rule function
def hebb_learning_rule(train_data):
    n = len(train_data)
    weights = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                weights[i][j] = 0
            else:
                weights[i][j] += train_data[i] * train_data[j]
    return weights

# Train the network with the training data
weights = hebb_learning_rule(train_A)
weights += hebb_learning_rule(train_H)

# Define test data for letter A and H
test_A = np.array([1, 1, 1, 0, 1,
                   1, 0, 0, 1, 1,
                   1, 1, 1, 1, 1,
                   1, 0, 0, 0, 1,
                   1, 0, 0, 0, 1])

test_H = np.array([1, 0, 0, 0, 1,
                   1, 0, 0, 0, 1,
                   1, 1, 1, 1, 1,
                   1, 0, 0, 0, 1,
                   1, 0, 0, 0, 1])

# Flatten test data to 1D arrays
test_A = test_A.flatten()
test_H = test_H.flatten()

# Test the network with the test data
output_A = np.dot(weights, test_A)
output_H = np.dot(weights, test_H)

# Determine the predicted letter based on the output
if np.greater(np.any(output_A) ,np.any(output_H)):
    print("Predicted letter is A")
else:
    print("Predicted letter is H")
