from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# Load the dataset
dataset = loadtxt('C:/Users/rohan/PycharmProjects/DL_PRACTICAL/pima-indians-diabetes.csv', delimiter=',')

# Split into input (X) and output (Y) variables
X = dataset[:, 0:8]
Y = dataset[:, 8]

# Print the input data
print(X)
print(Y)

# Define the model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

## Train the model
model.fit(X, Y, epochs=150, batch_size=10)

accuracy=model.evaluate(X,Y)
print("Accuracy of model is", (accuracy*100))

# Predict the output for the training data
prediction = model.predict(X, batch_size=4)

# Print the predictions
print(prediction)

exec("for i in range(5):print(X[i].tolist(),prediction[i], Y[i])")