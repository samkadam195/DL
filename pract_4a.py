from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Generate dataset
X, Y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)

# Scale the dataset
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

# Define the model
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=500)

# Generate new data for prediction
X_new, Y_real = make_blobs(n_samples=3, centers=2, n_features=2, random_state=1)
X_new = scaler.transform(X_new)

# Predict the class for new data
Y_new = np.argmax(model.predict(X_new), axis=1)
for i in range(len(X_new)):
    print("X=%s, Predicted=%s, Desired=%s" % (X_new[i], Y_new[i], Y_real[i]))
