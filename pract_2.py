import numpy as np
from keras.models import Sequential
import keras.layers
from keras.src.layers import Dense

# Define the model
model = Sequential()
model.add(Dense(units=2, activation="relu", input_dim=2))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the model summary
print(model.summary())

# Print the model weights
print(model.get_weights())

# Define the input data
X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y = np.array([0., 1., 1., 0.])

# Train the model
model.fit(X, Y, epochs=1000, batch_size=4)

# Print the updated model weights
print(model.get_weights())

# Predict the output
print(model.predict(X, batch_size=4))
