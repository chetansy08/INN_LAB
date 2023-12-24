import numpy as np

class Neuron:
    def __init__(self, input_size, activation_function='sigmoid'):
        # Initialize weights and bias randomly
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()

        # Choose activation function
        if activation_function == 'sigmoid':
            self.activation_function = self.sigmoid
            self.activation_derivative = self.sigmoid_derivative
        elif activation_function == 'step':
            self.activation_function = self.step_function
            self.activation_derivative = self.step_derivative
        else:
            raise ValueError("Unsupported activation function. Choose 'sigmoid' or 'step'.")

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def step_function(self, x):
        return 1 if x >= 0 else 0

    def step_derivative(self, x):
        return 0  # Derivative is not defined at the step points

    def forward(self, inputs):
        # Sum of weighted inputs plus bias
        weighted_sum = np.dot(inputs, self.weights) + self.bias

        # Apply activation function
        output = self.activation_function(weighted_sum)

        return output

# Example usage
input_size = 3  # Number of input features
neuron = Neuron(input_size, activation_function='step')

# Random input values for demonstration
inputs = np.random.rand(input_size)

# Forward pass through the neuron
output = neuron.forward(inputs)

# Display the results
print(f"Inputs: {inputs}")
print(f"Weights: {neuron.weights}")
print(f"Bias: {neuron.bias}")
print(f"Output: {output}")
