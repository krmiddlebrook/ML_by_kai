'''
author: Kai Middlebrook
Kaggle team: Kai, Jared, Nico, Angelo
'''


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import models
from tensorflow.keras import layers
from scipy.misc import toimage
import os

# Rules: Don't touch the test data. Design your convolutional neural network using only the training data. You can do whatever you want with the training data, but I recommend breaking it up into a reduced training set and a validation set. Another option to consider is using cross-validation when tuning hyperparameters in your CNN.

# Once you submit your code, I'll test your model on the test data and see what accuracy you get.

#%%
def show_imgs(X):
    plt.figure(1)
    k = 0
    for i in range(0,4):
        for j in range(0,4):
            plt.subplot2grid((4,4),(i,j))
            plt.imshow(toimage(X[k]))
            k = k+1
    # show the plot
    plt.show()

#%%
def load_cifar10(train_size):
    '''
    loads the cifar dataset and then processes it into an acceptable format for CNN
    returns:
        (1) X_train, Y_train: X matrix and Y vector containing onehot encoded values for each training example
        (2) X_test, Y_test: X matrix and Y vector containing onehot encoded values for each test example
    '''
    
    (X_train,Y_train), (X_test, Y_test) = cifar10.load_data() 
    

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    
    X_train = (X_train - X_train.mean())/ (X_train.max() - X_train.min())
    X_test = (X_test - X_test.mean())/ (X_test.max() - X_test.min())
    
    # The to_categorical function does one-hot encoding for us.
    Y_train = tf.keras.utils.to_categorical(Y_train)
    Y_test = tf.keras.utils.to_categorical(Y_test)
    
    X_train_partial = X_train[0:train_size,:,:,:]
    Y_train_partial = Y_train[0:train_size,:]
    X_val = X_train[train_size:,:,:,:]
    Y_val = Y_train[train_size:,:]
    
    return (X_train_partial, Y_train_partial), (X_val, Y_val), (X_test, Y_test)

#%%
(X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = load_cifar10(train_size=40000)

show_imgs(X_train[0:16])

#%%
def create_model():

    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', input_shape=(32,32,3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(layers.AveragePooling2D(pool_size=(2,2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(layers.AveragePooling2D(pool_size=(2,2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, kernel_size=(3,3), activation='tanh', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, kernel_size=(3,3), activation='tanh', padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))
    
    model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy',\
                    metrics = ['accuracy'])
    
    return model


#checkpoint_path = "/home4/krmiddlebrook/projects/cifar10/models/cp-.ckpt"
#checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
#cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
#                                                 save_weights_only=True,
#                                                 verbose=1)


model = create_model()
print(model.summary())

#%%
numTrainExamples = X_train.shape[0]
# Visualize a randomly selected training example:
plt.figure
i = np.random.randint(numTrainExamples)
plt.imshow(X_train[i,:,:],cmap=plt.get_cmap('gray'))

#%%
history = model.fit(X_train,Y_train, epochs = 50, batch_size = 128,
                      validation_data = (X_val, Y_val))
#model.save('cnn_7_dropout.h5')

print(history.history)
#%%

test_loss, test_acc = model.evaluate(X_test, Y_test)
print('test accuracy: {}  test loss: {}'.format(test_acc, test_loss))










