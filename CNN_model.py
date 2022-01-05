# -*- coding: utf-8 -*-
"""

@ author:   Harry Thomas Chirayil
This    module    contain  python specific lines    which    enables    
the    user    to    generate   convolutional neural network model 
without fine-tuning and freezing the sequence layers of the pre-trained model. 


"""

# Import the required packages and libraries

import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the pre-trained model
model = keras.models.load_model('GB1_keras_clf_cnn_many3_model3.h5')
model.trainable =  False

# Setup the deep learning model with freezing the sequence layers
inputs = keras.Input(shape=(290,553))
layer2 = keras.layers.Conv1D(553, kernel_size=18, strides= 5, activation="relu")(inputs)
model.summary()

outputs = model(layer2, training=False)

model2 = keras.Model(inputs, outputs)

model = model2

model.summary()

# Load the dataset
print ('loading data')

x = np.load('x.npy')
yold = np.load('y1_harry_test.npy')

# Clean the dataset 
floatArray = np.asarray(yold, dtype = float)
B = np.where(floatArray > 0.2, 1, 0)
print (B)

scaler = StandardScaler()
x = scaler.fit_transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)

# Split the dataset into training and validation sets
X_train_full, X_test, y_train_full, y_test = train_test_split(x, B,test_size=0.2)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full,test_size=0.2)

print('training set: ',X_train.shape)
print('test set: ',X_test.shape)
print('validation set: ',X_valid.shape)

# Compile the model
model.compile(loss="binary_crossentropy", optimizer="Adamax", metrics=["accuracy"])


# Fit the model
history = model.fit(X_train, y_train, epochs=100,
                    validation_data=(X_valid, y_valid))
 
                    
# Plot the accuracy and loss curves 
vhistory = history 
accuracy = vhistory.history['accuracy']
val_accuracy = vhistory.history['val_accuracy']
loss = vhistory.history['loss']
val_loss = vhistory.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()





