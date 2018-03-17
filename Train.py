
# coding: utf-8

# In[1]:


import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import optimizers
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[2]:


train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen  = ImageDataGenerator(rescale=1./255)
train_dir     = '../OCT2017/train/'
test_dir      = '../OCT2017/test/'
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(299, 299),     
                                                batch_size=128, class_mode='categorical')
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(299, 299),     
                                                batch_size=128, class_mode='categorical')


# In[3]:


# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)
# Get the output layer from the pre-trained Inception V3 model
x = base_model.output

# Now, add new layers that will be trained with our data
# These layers will be randomly initialized
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)

# Get the final Model to train
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers from the original base model so that we don't update the weights
for layer in base_model.layers:
    layer.trainable = False


# In[4]:


adam = optimizers.adam(lr=0.001)
# Compile the new model
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])


# In[5]:


# Setup a callback to save the best model
callbacks = [keras.callbacks.ModelCheckpoint('model.{epoch:02d}-{val_acc:.2f}.hdf5',
             monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)]

# Fit the data and output the history
history = model.fit_generator(train_generator, verbose=1, steps_per_epoch=len(train_generator), epochs=10,  
    validation_data=test_generator, validation_steps=len(test_generator), callbacks=callbacks)


# In[6]:


def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure()
    plt.title('Training and validation accuracy')
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', color='red', label='Validation acc')
    plt.legend()
    plt.show()

    plt.figure()
    plt.title('Training and validation loss')
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', color='red', label='Validation loss')
    plt.legend()
    plt.show()
    return acc, val_acc, loss, val_loss

acc, val_acc, loss, val_loss = plot_history(history)

