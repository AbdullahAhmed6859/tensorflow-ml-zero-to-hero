import tensorflow as tf
import numpy as np
from tensorflow import keras

print("started")

model = keras.Sequential([
    keras.Input(shape=(1,)),
    keras.layers.Dense(units=1)
])

model.compile(optimizer='sgd', loss='mean_squared_error')
print("compiled")

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

print("training")
model.fit(xs, ys, epochs=500, verbose=1)

print("predicting")
print(model.predict(np.array([10.0])))
