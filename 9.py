import numpy as np
import matplotlib.pyplot as plt

class kohenen_network:

    def __init__(self, input_size, output_size, learning_rate,epochs):

        self.input_size = input_size
        self.output_size = output_size

        self.weights = np.random.rand(self.output_size, self.input_size)
        self.data_points = np.random.rand(100, self.input_size)

        self.learning_rate= learning_rate
        self.epochs=epochs

    def train_model(self):

        for epoch in range(self.epochs):
            for data_point in self.data_points:

                self.winner_index = np.argmin(np.linalg.norm(self.weights - data_point, axis=1))
                self.weights[self.winner_index] += self.learning_rate * (data_point - self.weights[self.winner_index])

        return self.data_points, self.weights

input_size = 2
output_size = 10

learning_rate=0.1
epochs=100

kh=kohenen_network(input_size,output_size,learning_rate,epochs)

data_points, weights = kh.train_model()

plt.scatter(data_points[:, 0], data_points[:, 1], label='Data Points')
plt.scatter(weights[:, 0], weights[:, 1], marker='x', s=100, label='Kohonen Neurons')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title("Kohenen Neurons")
plt.show()
