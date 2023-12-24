import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, max_epochs=1000):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def predict(self, inputs):
        # Sum of weighted inputs plus bias
        weighted_sum = np.dot(inputs, self.weights) + self.bias

        # Apply step function as activation
        return 1 if weighted_sum >= 0 else 0

    def train(self, training_data, labels):
        for epoch in range(self.max_epochs):
            errors = 0
            for inputs, label in zip(training_data, labels):
                # Forward pass
                prediction = self.predict(inputs)

                # Update weights and bias based on prediction error
                update = self.learning_rate * (label - prediction)
                self.weights += update * inputs
                self.bias += update

                errors += int(update != 0)

            # Check if all predictions are correct, if yes, stop training
            if errors == 0:
                print(f"Training converged after {epoch + 1} epochs.")
                break

        if errors > 0:
            print(f"Training did not converge after {self.max_epochs} epochs.")

# Example usage
input_size = 2  # Number of input features
perceptron = Perceptron(input_size)

# Training data (replace with your actual data)
training_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([0, 0, 0, 1])  # Binary labels for AND gate

# Train the perceptron
perceptron.train(training_data, labels)

# Test the trained perceptron
test_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predictions = [perceptron.predict(inputs) for inputs in test_inputs]

# Display the results
print("Test predictions:", predictions)
