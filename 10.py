import numpy as np
import matplotlib.pyplot as plt
np.random.rand(1000)


class Som:

    def __init__(self, epochs):

        self.weights = np.random.rand(grid_size, grid_size, 2)
        self.data_points = np.random.randn(500, 2)
        self.epochs=epochs

    def train(self):

        for epoch in range(self.epochs):
            for data_point in self.data_points:
                winner = np.argmin(np.linalg.norm(self.weights - data_point, axis=2))
                neighborhood = np.exp(-np.linalg.norm(np.indices((grid_size, grid_size)).T - np.unravel_index(winner, (grid_size, grid_size)), axis=2) / 2)
                self.weights += 0.1 * neighborhood[:, :, np.newaxis] * (data_point - self.weights)
        return self.data_points, self.weights

grid_size = 10
epochs=100
som = Som(epochs)
weights = som.train()
plt.scatter(som.data_points[:, 0], som.data_points[:, 1], label='Data Points')
plt.scatter(som.weights[:, :, 0], som.weights[:, :, 1], marker='x', s=100, label='SOM Neurons')
plt.legend()
plt.show()
