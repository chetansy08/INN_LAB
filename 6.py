import numpy as np
import matplotlib.pyplot as plt

def quadratic_function(x):
    return x**2-2*x

x_train = np.linspace(-1, 1, 100).reshape(-1, 1)
y_train = quadratic_function(x_train)

np.random.seed(42)
weights_output = np.random.randn(1, 1)
biases_output = np.zeros((1, 1))
learning_rate = 0.01
epochs = 1000

for epoch in range(epochs):
    output_layer_input = np.dot(x_train, weights_output) + biases_output
    loss = np.mean((output_layer_input - y_train)**2)
    output_error = output_layer_input - y_train
    output_delta = output_error
    weights_output -= learning_rate * np.dot(x_train.T, output_delta)
    biases_output -= learning_rate * np.sum(output_delta, axis=0, keepdims=True)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

predictions = np.dot(x_train, weights_output) + biases_output
plt.scatter(x_train, y_train, label='Training Data')
plt.plot(x_train, predictions, color='red', label='Model Prediction')
plt.legend()
plt.show()
