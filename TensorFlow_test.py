import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  #归一化
#print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
y=tf.one_hot(y_train,depth=10)  #depth ont-hot 编码的长度
#print(y)

#######将图片和对应的号码显示出来
# def show_single_image(img_arr, label):
#     plt.imshow(img_arr, cmap='binary')
#     plt.title('%i' % label)
#     plt.show()
#
#
# image_id = 8
# show_single_image(x_train[image_id], y_train[image_id])
model = models.Sequential([layers.Conv2D(filters=6, kernel_size=3, strides=1, input_shape=(28, 28, 1)),
                           layers.MaxPooling2D(pool_size=2, strides=2),
                           layers.ReLU(),
                           layers.Conv2D(filters=16, kernel_size=3, strides=1),
                           layers.MaxPooling2D(pool_size=2, strides=2),
                           layers.ReLU(),
                           layers.Flatten(),
                           layers.Dense(120, activation='relu'),
                           layers.Dropout(0.5),
                           layers.Dense(84, activation='relu'),
                           layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])
model.fit(x_train, y_train, epochs=5,validation_data=(x_test, y_test),validation_freq=1)
model.summary()
test_loss, test_acc=model.evaluate(x_test,  y_test)

#print(test_loss, test_acc)
model.save('my_mnist.h5')