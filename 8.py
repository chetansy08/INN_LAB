import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def create_datasets():
    x_train = np.linspace(-1, 1, 100).reshape(-1, 1)
    y_train = x_train**2 + np.random.normal(0, 0.1, size=x_train.shape)
    return x_train, y_train

def train_model(x_train, y_train):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(x_train, y_train, epochs=1000, verbose=0)
    return model

x_train, y_train=create_datasets()
model=train_model(x_train, y_train)

x_test = np.linspace(-1, 1, 100).reshape(-1, 1)
y_pred = model.predict(x_test)

plt.scatter(x_train, y_train, label='Training Data')
plt.plot(x_test, y_pred, color='red', label='Model Prediction')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title("Cover's Theorem")
plt.show()
