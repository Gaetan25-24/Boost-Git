import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0, 100, 0.1)
data = np.sin(0.1 * t) + np.random.randn(len(t)) * 0.1
x = np.array([data[i:i+10] for i in range(len(data)-10)])
y = data[10:]

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(32, input_shape=(10, 1), return_sequences=False),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
x = x.reshape((x.shape[0], x.shape[1], 1))
model.fit(x, y, epochs=20, verbose=0)

predictions = model.predict(x)
plt.plot(y, label='Vraie valeur')
plt.plot(predictions, label='Prédiction')
plt.title("Prédiction de série temporelle")
plt.legend()
plt.show()