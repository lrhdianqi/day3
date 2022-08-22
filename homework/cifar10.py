# %%
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from keras.datasets import cifar10

# %%
cifar_10 = tf.keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar_10.load_data()
print('Datatype: X:%s, y:%s' % (train_images.dtype,train_labels.dtype))
print('Train: X=%s, y=%s' % (train_images.shape, train_labels.shape))
print('Test: X=%s, y=%s' % (test_images.shape, test_labels.shape))
# %%
# 共10类图片，0-9如下：
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# %%
from matplotlib import pyplot
for i in range(9):
	# define subplot
	pyplot.subplot(330+1+i) 
	# plot raw pixel data
	pyplot.imshow(train_images[i])
# show the figure
pyplot.show()
print( train_labels[:8])

#%%
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
rain_images = train_images / 255.0
test_images = test_images / 255.0
print(train_labels[:8])
# %%
import tensorflow as tf
from tensorflow.python.keras import layers, models
import matplotlib.pyplot as plt

model = models.Sequential()
# 32 kernels of size (3,3)
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.summary()
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
model.summary()

# %%
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
#%%
history=model.fit(train_images, train_labels, epochs=20, batch_size=32)
#%%
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)



































































# %%
loss, acc = model.evaluate(test_images, test_labels, verbose=0)
print('the accuracy of test set is: %.3f' % (acc * 100.0))
# %%
import sys # 涉及系统中文件
#%%
# plot loss
pyplot.subplot(211)
pyplot.title('Cross Entropy Loss')
pyplot.plot(history.history['loss'], color='blue', label='train')
pyplot.plot(history.history['val_loss'], color='orange', label='test')
#%%
# plot accuracy
pyplot.subplot(212)
pyplot.title('Classification Accuracy')
pyplot.plot(history.history['accuracy'], color='blue', label='train')
pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
# pyplot.legend
pyplot.show



# %%
