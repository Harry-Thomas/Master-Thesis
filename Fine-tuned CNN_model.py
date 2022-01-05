# -*- coding: utf-8 -*-
"""


@author: Harry PC
"""


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt




# Unfreeze the base model
model.trainable = True

model.summary()



# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
)

# Fit the model
history = model.fit(X_train, y_train, epochs=100,
                    validation_data=(X_valid, y_valid))




# Plot the accuracy and loss curves 
vhistory = history 
accuracy = vhistory.history['binary_accuracy']
val_accuracy = vhistory.history['val_binary_accuracy']
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


############################################################